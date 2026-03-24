#Here is an improved algorithm based on the **jSO (Self-Adaptive Differential Evolution with Linear Population Reduction)** architecture, which is a refined version of L-SHADE. 
#
#**Key Improvements over the previous solution:**
#1.  **Refined Parameter Adaptation (jSO strategies):** It uses a specific decay function for the `p` parameter (controlling greediness of the mutation) and a specific weighting mechanism for updating memory values (giving more weight to high-performing improvements).
#2.  **Boundary Reflection:** Instead of simple clipping (which clumps solutions at the bounds), it uses reflection. If a mutation shoots past a bound, it "bounces" back into the search space. This preserves diversity near the edges.
#3.  **Local Search Polishing (Coordinate Descent):** When the global search stagnates or time is running out (last 10%), the algorithm triggers a lightweight local search (Coordinate Descent) on the best-found solution. This helps fine-tune the decimal precision that global heuristics often miss.
#4.  **Vectorized Operations:** Maximizes usage of NumPy for mutation and crossover to minimize Python interpreter overhead.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Implements a jSO-inspired algorithm (an improved L-SHADE variant) 
    hybridized with a lightweight Coordinate Descent Local Search.
    
    1. Dynamic Population Reduction (Linear).
    2. Weighted Parameter Adaptation (Memory F and CR).
    3. Dynamic 'p' for current-to-pbest mutation.
    4. Boundary Reflection.
    5. Final Stage Local Polishing.
    """
    start_time = time.time()
    
    # --- Helper: Boundary Reflection ---
    def apply_bound_reflection(x, min_b, max_b):
        """
        Reflects particles that go out of bounds back into the search space.
        If reflection still results in OOB (rare), it clips.
        """
        # Lower bound reflection
        under = x < min_b
        if np.any(under):
            # x = min + (min - x) = 2*min - x
            x[under] = 2 * min_b[under] - x[under]
            # If still under (was way out), clip
            x[under] = np.maximum(x[under], min_b[under])
            
        # Upper bound reflection
        over = x > max_b
        if np.any(over):
            # x = max - (x - max) = 2*max - x
            x[over] = 2 * max_b[over] - x[over]
            # If still over, clip
            x[over] = np.minimum(x[over], max_b[over])
        return x

    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    
    # Population Strategy: Start large to explore, reduce to exploit
    # Capped at 300 to ensure iteration speed on slower functions
    pop_size_init = int(max(30, min(300, 25 * dim)))
    pop_size_min = 4
    
    # Memory for L-SHADE
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Archive
    archive = []
    arc_rate = 2.0 # Archive size = arc_rate * pop_size
    
    # Initial Population
    pop = min_b + np.random.rand(pop_size_init, dim) * (max_b - min_b)
    fitness = np.full(pop_size_init, float('inf'))
    
    best_val = float('inf')
    best_idx = -1
    best_vec = np.zeros(dim)

    # Initial Evaluation
    # We check time strictly
    for i in range(pop_size_init):
        if time.time() - start_time >= max_time:
            # Return best found so far (if any) or 0 if completely failed
            return best_val if best_val != float('inf') else 0.0
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i
            best_vec = pop[i].copy()

    # --- Main Loop ---
    curr_pop_size = pop_size_init
    
    # Local search state
    local_search_triggered = False
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed >= max_time:
            return best_val
            
        progress = elapsed / max_time
        
        # --- 1. Population Reduction (L-SHADE) ---
        next_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        next_pop_size = max(pop_size_min, next_pop_size)
        
        if next_pop_size < curr_pop_size:
            # Sort and kill worst
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:next_pop_size]]
            fitness = fitness[sorted_idx[:next_pop_size]]
            curr_pop_size = next_pop_size
            best_idx = np.argmin(fitness) # Recalculate best index relative to new pop
            
            # Reduce archive size if necessary
            curr_arc_size = int(curr_pop_size * arc_rate)
            if len(archive) > curr_arc_size:
                import random
                random.shuffle(archive)
                archive = archive[:curr_arc_size]

        # --- 2. Parameter Generation (jSO) ---
        # p (for p-best) decreases linearly from p_max to p_min
        p_max, p_min = 0.25, 0.05
        p = p_max - (p_max - p_min) * progress
        
        r_idx = np.random.randint(0, H, curr_pop_size)
        
        # Generate CR (Normal Distribution)
        m_cr = mem_cr[r_idx]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        # Inherit special case from jSO: if m_cr is -1 (terminal), fix CR to 0
        # (omitted here for standard L-SHADE stability)

        # Generate F (Cauchy Distribution)
        m_f = mem_f[r_idx]
        # jSO logic: for initial stages, keep F higher? 
        # We stick to standard Cauchy but retry if <= 0
        f = np.zeros(curr_pop_size)
        for i in range(curr_pop_size):
            while True:
                f_val = m_f[i] + 0.1 * np.random.standard_cauchy()
                if f_val > 0:
                    f[i] = min(f_val, 1.0)
                    break
        
        # --- 3. Mutation (current-to-pbest/1) ---
        # Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        num_pbest = max(1, int(p * curr_pop_size))
        pbest_candidates = sorted_indices[:num_pbest]
        
        pbest_indices = np.random.choice(pbest_candidates, curr_pop_size)
        x_pbest = pop[pbest_indices]
        
        # r1: random distinct from current
        r1_indices = np.random.randint(0, curr_pop_size, curr_pop_size)
        # Fix collisions
        collision = (r1_indices == np.arange(curr_pop_size))
        while np.any(collision):
            r1_indices[collision] = np.random.randint(0, curr_pop_size, np.sum(collision))
            collision = (r1_indices == np.arange(curr_pop_size))
        x_r1 = pop[r1_indices]
        
        # r2: random distinct from current and r1, from Union(Pop, Archive)
        if len(archive) > 0:
            pool = np.vstack((pop, np.array(archive)))
        else:
            pool = pop
        
        r2_indices = np.random.randint(0, len(pool), curr_pop_size)
        
        # Fix collisions logic for r2
        # r2 must not be i (current) AND r2 must not be r1
        # Note: r1 is index in 'pop', r2 is index in 'pool'
        
        # Current index in pool is just 0..curr_pop_size-1
        # r1 index in pool is just r1_indices
        
        bad_r2 = (r2_indices == np.arange(curr_pop_size)) | (r2_indices == r1_indices)
        while np.any(bad_r2):
            r2_indices[bad_r2] = np.random.randint(0, len(pool), np.sum(bad_r2))
            bad_r2 = (r2_indices == np.arange(curr_pop_size)) | (r2_indices == r1_indices)
            
        x_r2 = pool[r2_indices]
        
        # Compute Mutant V
        f_col = f[:, None]
        # v = x + F*(xp - x) + F*(xr1 - xr2)
        mutants = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # --- 4. Crossover (Binomial) ---
        rand_vals = np.random.rand(curr_pop_size, dim)
        j_rand = np.random.randint(0, dim, curr_pop_size)
        
        mask = (rand_vals < cr[:, None])
        mask[np.arange(curr_pop_size), j_rand] = True
        
        trials = np.where(mask, mutants, pop)
        
        # --- 5. Bound Handling (Reflection) ---
        trials = apply_bound_reflection(trials, min_b, max_b)
        
        # --- 6. Selection ---
        succ_mask = np.zeros(curr_pop_size, dtype=bool)
        diff_fitness = np.zeros(curr_pop_size)
        
        # Prepare archive candidates
        replaced_parents = []
        
        for i in range(curr_pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            t_val = func(trials[i])
            
            if t_val < fitness[i]:
                # Improvement
                succ_mask[i] = True
                diff_fitness[i] = fitness[i] - t_val
                
                replaced_parents.append(pop[i].copy())
                pop[i] = trials[i]
                fitness[i] = t_val
                
                if t_val < best_val:
                    best_val = t_val
                    best_vec = trials[i].copy()
                    best_idx = i
            elif t_val == fitness[i]:
                # Neutral move, accept but don't count as success for parameter adaptation
                pop[i] = trials[i]

        # Update Archive
        for parent in replaced_parents:
            if len(archive) < int(curr_pop_size * arc_rate):
                archive.append(parent)
            else:
                idx = np.random.randint(0, len(archive))
                archive[idx] = parent
                
        # --- 7. Memory Update (Weighted Lehmer Mean) ---
        if np.any(succ_mask):
            s_cr = cr[succ_mask]
            s_f = f[succ_mask]
            w_fit = diff_fitness[succ_mask]
            
            # Weighted mean
            w = w_fit / (np.sum(w_fit) + 1e-15)
            
            # Update M_CR
            if np.max(s_cr) == 0:
                mean_cr = 0
            else:
                mean_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
            
            # Update M_F (Lehmer Mean: sum(w*f^2) / sum(w*f))
            sum_wf = np.sum(w * s_f)
            if sum_wf > 1e-10:
                mean_f = np.sum(w * (s_f ** 2)) / sum_wf
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
            
            k_mem = (k_mem + 1) % H

        # --- 8. Local Search Polishing (Coordinate Descent) ---
        # Trigger logic: If we are in the last 15% of time OR std dev is very low (convergence)
        # We only run this sparsely to avoid wasting time on global search
        
        std_fit = np.std(fitness)
        near_end = progress > 0.85
        converged = std_fit < 1e-8
        
        if (near_end or converged) and not local_search_triggered:
            # Run a coordinate descent on the best vector
            # This helps fine-tune the decimals that DE might miss
            local_search_triggered = True # Don't get stuck in this loop forever
            
            current_best = best_vec.copy()
            current_score = best_val
            
            # Step size: proportional to domain width, decreasing
            step_sizes = (max_b - min_b) * 0.005 
            
            # Limit LS steps to avoid timeout
            ls_max_steps = 50 
            ls_steps = 0
            
            improved = True
            while improved and ls_steps < ls_max_steps:
                improved = False
                for d in range(dim):
                    if time.time() - start_time >= max_time:
                        return best_val
                        
                    # Try negative direction
                    orig_val = current_best[d]
                    current_best[d] -= step_sizes[d]
                    # Bound check
                    if current_best[d] < min_b[d]: current_best[d] = min_b[d]
                    
                    new_val = func(current_best)
                    if new_val < current_score:
                        current_score = new_val
                        best_val = new_val
                        best_vec = current_best.copy()
                        improved = True
                    else:
                        # Try positive direction
                        current_best[d] = orig_val + step_sizes[d]
                        if current_best[d] > max_b[d]: current_best[d] = max_b[d]
                        
                        new_val = func(current_best)
                        if new_val < current_score:
                            current_score = new_val
                            best_val = new_val
                            best_vec = current_best.copy()
                            improved = True
                        else:
                            # Revert
                            current_best[d] = orig_val
                            
                # Refine step size if no improvement in a full pass
                if not improved:
                    step_sizes *= 0.5
                    # If step size is tiny, stop LS
                    if np.max(step_sizes) > 1e-9:
                        improved = True # Continue with smaller steps
                    
                ls_steps += 1
                
            # Inject improved best back into population
            if best_idx != -1 and best_idx < curr_pop_size:
                pop[best_idx] = best_vec
                fitness[best_idx] = best_val

    return best_val
