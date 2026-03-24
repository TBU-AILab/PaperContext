#To improve upon the previous algorithm, we will implement **L-SHADE-E (L-SHADE with Epitaxial Local Search)**. This approach addresses the stagnation issue by incorporating two critical enhancements:
#
#1.  **Robust L-SHADE Core**: An optimized implementation of the L-SHADE algorithm (Success-History based Adaptive Differential Evolution with Linear Population Reduction). This is the current state-of-the-art for continuous "black-box" optimization. It dynamically adapts mutation (`F`) and crossover (`CR`) rates and manages diversity using an external archive.
#2.  **Coordinate Descent Local Search (Polishing)**: The previous result of ~1.46 suggests the algorithm found a good basin of attraction but failed to descend to the exact global minimum (likely 0.0). We inject a high-precision Coordinate Descent (MTS-LS1 style) phase that activates when the population stagnates or before a restart. This "polishes" the best solution found so far.
#3.  **Restart Mechanism**: If the population converges (standard deviation drops to zero) or stagnates, the algorithm triggers a restart to escape local optima, carrying over the global best solution to the next run.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Epitaxial Local Search and Restarts.
    
    Optimizes a black-box function by combining:
    1. L-SHADE: Adaptive Differential Evolution with Linear Population Size Reduction.
    2. Coordinate Descent: For fine-tuning (polishing) the best solution.
    3. Restart Strategy: To handle multimodal landscapes effectively within the time limit.
    """
    start_time = time.time()
    
    # --- Configuration & Pre-calculation ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    diff_b = ub - lb
    
    # Global state to track best across restarts
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Local Search: Coordinate Descent ---
    def local_search_polish(current_sol, current_val, time_limit):
        """
        Performs Coordinate Descent (MTS-LS1 style) to polish the solution.
        Crucial for driving 1.46 -> 0.0.
        """
        sol = current_sol.copy()
        val = current_val
        
        # Start with a small step relative to domain, shrink iteratively
        step_size = diff_b * 0.01
        min_step = 1e-15
        
        # Continue until step size is negligible
        while np.max(step_size) > min_step:
            improved = False
            # Randomize dimension order to avoid bias
            dims = np.random.permutation(dim)
            
            for d in dims:
                if time.time() - start_time >= time_limit:
                    return sol, val
                
                original_x = sol[d]
                step = step_size[d]
                
                # Try negative direction
                sol[d] = np.clip(original_x - step, lb[d], ub[d])
                new_val = func(sol)
                
                if new_val < val:
                    val = new_val
                    improved = True
                else:
                    # Try positive direction
                    sol[d] = np.clip(original_x + step, lb[d], ub[d])
                    new_val = func(sol)
                    
                    if new_val < val:
                        val = new_val
                        improved = True
                    else:
                        # Revert if no improvement
                        sol[d] = original_x
            
            # If we improved, we might keep the step size (greedy), 
            # otherwise we refine the precision.
            if not improved:
                step_size *= 0.5
            
        return sol, val

    # --- Main Restart Loop ---
    while True:
        # Check if enough time remains for a meaningful run (at least 5% of time left)
        elapsed = time.time() - start_time
        if elapsed >= max_time * 0.95:
            break

        # --- L-SHADE Initialization ---
        # Initial population size: 18 * dim is a standard heuristic for L-SHADE
        pop_size = int(18 * dim)
        pop_size_min = 4
        
        # Initialize population within bounds
        pop = lb + np.random.rand(pop_size, dim) * diff_b
        fitness = np.zeros(pop_size)
        
        # Evaluate initial population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return global_best_val
            fitness[i] = func(pop[i])
            if global_best_sol is None or fitness[i] < global_best_val:
                global_best_val = fitness[i]
                global_best_sol = pop[i].copy()
        
        # L-SHADE Memory (History)
        H = 5 # Memory size
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive (maintains diversity)
        arc_rate = 2.0 # Archive size relative to pop_size
        max_arc_size = int(pop_size * arc_rate)
        archive = np.empty((max_arc_size, dim))
        n_arc = 0
        
        # LPSR State
        initial_pop_size = pop_size
        
        # Stagnation Detection
        last_best_val = np.min(fitness)
        stagnation_count = 0
        
        # --- Evolutionary Generation Loop ---
        while True:
            # Time check
            curr_time = time.time()
            if curr_time - start_time >= max_time:
                return global_best_val
            
            # Linear Population Size Reduction (LPSR)
            # We map the reduction to the remaining time or convergence profile
            progress = (curr_time - start_time) / max_time
            
            # Calculate target population size
            next_pop_size = int(round((pop_size_min - initial_pop_size) * progress + initial_pop_size))
            next_pop_size = max(pop_size_min, next_pop_size)
            
            # Apply reduction
            if next_pop_size < pop_size:
                # Sort by fitness and keep top
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:next_pop_size]]
                fitness = fitness[sort_idx[:next_pop_size]]
                pop_size = next_pop_size
                
                # Resize archive capacity
                max_arc_size = int(pop_size * arc_rate)
                if n_arc > max_arc_size:
                    # Randomly remove excess
                    keep_idx = np.random.choice(n_arc, max_arc_size, replace=False)
                    archive[:max_arc_size] = archive[keep_idx]
                    n_arc = max_arc_size
            
            # --- Parameter Adaptation ---
            # Select memory slots
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate CR (Normal dist)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy dist)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Retry if F <= 0, clip if F > 1
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                f[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
            f = np.clip(f, 0, 1)
            
            # --- Mutation Strategy: current-to-pbest/1 ---
            # Sort population to find p-best
            sorted_indices = np.argsort(fitness)
            p_rate = 0.11 # Top 11%
            num_p_best = max(2, int(pop_size * p_rate))
            p_best_indices = sorted_indices[:num_p_best]
            
            # Randomly select p-best for each individual
            p_best_sel = np.random.choice(p_best_indices, pop_size)
            x_pbest = pop[p_best_sel]
            
            # Select r1 (random from pop)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_idx]
            
            # Select r2 (random from Union(pop, archive))
            if n_arc > 0:
                pool = np.vstack((pop, archive[:n_arc]))
            else:
                pool = pop
            r2_idx = np.random.randint(0, len(pool), pop_size)
            x_r2 = pool[r2_idx]
            
            # Calculate mutant vectors V
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f[:, None]
            v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, pop_size)
            rand_vals = np.random.rand(pop_size, dim)
            mask = rand_vals < cr[:, None]
            mask[np.arange(pop_size), j_rand] = True
            
            u = np.where(mask, v, pop)
            
            # --- Constraint Handling (Midpoint back-projection) ---
            mask_l = u < lb
            if np.any(mask_l):
                u[mask_l] = (pop[mask_l] + lb[np.where(mask_l)[1]]) / 2.0
            mask_u = u > ub
            if np.any(mask_u):
                u[mask_u] = (pop[mask_u] + ub[np.where(mask_u)[1]]) / 2.0
                
            # --- Evaluation and Selection ---
            # Tracking successful parameters for memory update
            diff_f = []
            succ_f = []
            succ_cr = []
            
            for i in range(pop_size):
                # Check time strictly inside evaluation loop
                if time.time() - start_time >= max_time:
                    return global_best_val
                
                new_val = func(u[i])
                
                # Selection
                if new_val <= fitness[i]:
                    # If strictly better, store in archive and record success
                    if new_val < fitness[i]:
                        if n_arc < max_arc_size:
                            archive[n_arc] = pop[i].copy()
                            n_arc += 1
                        else:
                            # Replace random archive member
                            rand_k = np.random.randint(0, n_arc)
                            archive[rand_k] = pop[i].copy()
                            
                        diff_f.append(fitness[i] - new_val)
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                    
                    # Update Population
                    pop[i] = u[i]
                    fitness[i] = new_val
                    
                    if new_val < global_best_val:
                        global_best_val = new_val
                        global_best_sol = u[i].copy()
                        
            # --- Memory Update ---
            if len(diff_f) > 0:
                diff_f = np.array(diff_f)
                succ_f = np.array(succ_f)
                succ_cr = np.array(succ_cr)
                
                # Weighted Lehmer Mean
                weights = diff_f / np.sum(diff_f)
                
                mean_f = np.sum(weights * (succ_f ** 2)) / np.sum(weights * succ_f)
                mean_cr = np.sum(weights * succ_cr)
                
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                k_mem = (k_mem + 1) % H
            
            # --- Convergence / Restart Check ---
            # 1. Stagnation check
            current_best = np.min(fitness)
            if current_best < last_best_val:
                last_best_val = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # 2. Population Diversity check
            std_fit = np.std(fitness)
            
            # Conditions to trigger Restart:
            # - Population has collapsed (std approx 0)
            # - Stagnated for too long (30 generations)
            if std_fit < 1e-9 or stagnation_count > 30:
                # Before restarting, POLISH the best solution found
                if global_best_sol is not None:
                    global_best_sol, global_best_val = local_search_polish(global_best_sol, global_best_val, max_time)
                
                # Break inner loop to trigger full restart
                break
                
    return global_best_val
