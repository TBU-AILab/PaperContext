#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 1.4239293629960912
#   Algorithm: L-SHADE-R (L-SHADE with Adaptive Restarts)
#
#The following algorithm implements **L-SHADE-R-E (L-SHADE with Restarts and Elite Injection)**.
#
#**Key Improvements:**
#1.  **Time-Aware Population Sizing**: The initial population size for each restart is dynamically scaled based on the problem dimension and capped to ensure responsiveness. If the remaining time is critical (< 2s), the population size is further reduced to guarantee that the algorithm can perform a sufficient number of generations to converge, rather than spending all time on a single generation of a large population.
#2.  **Elitism with Gaussian Injection**: Upon restarting, the algorithm doesn't just inject the single global best solution. It also injects mutated clones (Gaussian perturbations) of the global best. This creates a "micro-swarm" around the best known solution to refine it locally, while the rest of the randomly initialized population explores the global space.
#3.  **Tighter Restart Triggers**: The stagnation detection is made more robust. If the population variance drops below a strict tolerance or the best fitness does not improve for a dynamically calculated number of generations (based on dimension), a restart is triggered immediately to avoid wasting computational budget on a converged basin.
#4.  **Refined Midpoint-Target Boundary Handling**: The boundary handling places out-of-bounds solutions exactly halfway between the parent and the bound. This proven strategy preserves the search trajectory better than random re-initialization or clipping.
#5.  **Robust Vectorization**: All evolutionary operators (mutation, crossover, selection) are fully vectorized using NumPy, maximizing the number of evaluations performed per second.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-R-E (Linear Population Reduction SHADE with Restarts & Elite Injection).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Parameters & Configuration
    # -------------------------------------------------------------------------
    # Initial Population Size Configuration
    # We use a linear scaling with dimension, clipped to robust limits for Python.
    base_pop_mult = 20
    min_pop_init = 20
    max_pop_init = 180  # Cap to prevent slow generations in high dims
    
    # Minimum population size for LPSR
    pop_size_min = 5
    
    # SHADE Parameters
    H = 6                 # Memory size
    arc_rate = 2.6        # Archive size relative to population
    
    # Restart Triggers
    stop_tol = 1e-6       # Variance tolerance
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')
    global_best_vec = None
    
    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def get_remaining_seconds():
        return max_time - (datetime.now() - start_time).total_seconds()
    
    # -------------------------------------------------------------------------
    # Main Optimization Loop (Restarts)
    # -------------------------------------------------------------------------
    while True:
        rem_time = get_remaining_seconds()
        
        # Safety buffer: if time is nearly up, return best found
        if rem_time < 0.05:
            return global_best_val
            
        # ---------------------------------------------------------------------
        # Epoch Setup
        # ---------------------------------------------------------------------
        epoch_start_time = datetime.now()
        
        # Dynamic Population Sizing based on Remaining Time
        # If time is tight, reduce population to ensure we get iterations done.
        current_pop_mult = base_pop_mult
        if rem_time < 1.0:
            current_pop_mult = 10  # Aggressive reduction for final sprint
            
        pop_size_init = int(np.clip(current_pop_mult * dim, min_pop_init, max_pop_init))
        
        # Initialize Population
        pop_size = pop_size_init
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism & Elite Injection
        # If we have a global best, inject it AND some local perturbations
        if global_best_vec is not None:
            # 1. Inject exact global best
            pop[0] = global_best_vec.copy()
            fitness[0] = global_best_val
            
            # 2. Inject Gaussian perturbations (Local Search around best)
            # We replace a few random individuals with mutants of the best
            n_inject = min(3, pop_size - 1)
            for k in range(1, 1 + n_inject):
                # Scale perturbation by domain size, small factor
                pert = np.random.normal(0, 0.01, dim) * diff_b
                candidate = global_best_vec + pert
                candidate = np.clip(candidate, min_b, max_b)
                pop[k] = candidate
                # fitness[k] will be evaluated in the loop
        
        # SHADE Memories (Reset for new basin exploration)
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Local tracking
        epoch_best_val = float('inf')
        stall_counter = 0
        stall_limit = max(40, 5 * dim)
        
        # ---------------------------------------------------------------------
        # Initial Evaluation (Epoch)
        # ---------------------------------------------------------------------
        for i in range(pop_size):
            # Check time frequently
            if i % 10 == 0:
                if (datetime.now() - start_time) >= time_limit:
                    return global_best_val
            
            # Skip if already set (Elitism)
            if fitness[i] == float('inf'):
                val = func(pop[i])
                fitness[i] = val
            
            if fitness[i] < epoch_best_val:
                epoch_best_val = fitness[i]
                
            if fitness[i] < global_best_val:
                global_best_val = fitness[i]
                global_best_vec = pop[i].copy()
                
        # ---------------------------------------------------------------------
        # Generations Loop
        # ---------------------------------------------------------------------
        while True:
            # 1. Time Check
            now = datetime.now()
            elapsed_total = (now - start_time).total_seconds()
            if elapsed_total >= max_time:
                return global_best_val
            
            # 2. Linear Population Size Reduction (LPSR)
            # We scale the reduction to finish within the remaining time allocated to this epoch.
            # We assume the epoch can use the rest of the available time.
            epoch_elapsed = (now - epoch_start_time).total_seconds()
            
            # Progress relative to the budget we saw at start of restart
            progress = epoch_elapsed / rem_time
            if progress > 1.0: progress = 1.0
            
            target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Reduce Population (Keep Best)
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices[:target_size]]
                fitness = fitness[sort_indices[:target_size]]
                pop_size = target_size
                
                # Resize Archive
                curr_arc_cap = int(pop_size * arc_rate)
                if len(archive) > curr_arc_cap:
                    # Remove random elements
                    to_remove_count = len(archive) - curr_arc_cap
                    # Create a boolean mask to keep elements
                    keep_mask = np.ones(len(archive), dtype=bool)
                    remove_idxs = np.random.choice(len(archive), to_remove_count, replace=False)
                    keep_mask[remove_idxs] = False
                    # Rebuild archive
                    new_archive = [archive[idx] for idx in range(len(archive)) if keep_mask[idx]]
                    archive = new_archive
            
            # 3. Restart Triggers
            # Trigger A: Stagnation
            current_best = np.min(fitness)
            if current_best < epoch_best_val - 1e-12:
                epoch_best_val = current_best
                stall_counter = 0
            else:
                stall_counter += 1
                
            if stall_counter >= stall_limit:
                break # Restart
                
            # Trigger B: Convergence (Low Variance)
            if np.std(fitness) < stop_tol:
                break # Restart
                
            # Trigger C: Min Size & Stalled
            if pop_size <= pop_size_min and stall_counter > 15:
                break
                
            # 4. Parameter Adaptation
            # p decreases linearly from 0.2 to 0.05
            p_val = 0.2 - (0.15 * progress)
            p_val = max(0.05, p_val)
            
            # Generate CR and F (Vectorized)
            r_idxs = np.random.randint(0, H, pop_size)
            mu_cr = mem_cr[r_idxs]
            mu_f = mem_f[r_idxs]
            
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            f[f > 1] = 1.0
            
            # Repair negative F
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                n_neg = np.sum(mask_neg)
                f[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(n_neg)
                f[f > 1] = 1.0
                
            # 5. Mutation: current-to-pbest/1
            # Select p-best indices
            sorted_idx = np.argsort(fitness)
            n_pbest = max(2, int(p_val * pop_size))
            pbest_pool = sorted_idx[:n_pbest]
            
            x_pbest = pop[np.random.choice(pbest_pool, pop_size)]
            
            # Select r1 (!= i)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            # Fix collisions
            col_i = (r1_idxs == np.arange(pop_size))
            r1_idxs[col_i] = (r1_idxs[col_i] + 1) % pop_size
            x_r1 = pop[r1_idxs]
            
            # Select r2 (!= i, != r1) from Pop + Archive
            if len(archive) > 0:
                pop_all = np.vstack((pop, np.array(archive)))
            else:
                pop_all = pop
            n_all = len(pop_all)
            
            r2_idxs = np.random.randint(0, n_all, pop_size)
            # Collision fixing loop
            for k in range(pop_size):
                while r2_idxs[k] == k or r2_idxs[k] == r1_idxs[k]:
                    r2_idxs[k] = np.random.randint(0, n_all)
            x_r2 = pop_all[r2_idxs]
            
            # Compute Mutant
            f_v = f[:, None]
            mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
            
            # 6. Crossover (Binomial)
            rand_cr = np.random.rand(pop_size, dim)
            mask_cross = rand_cr < cr[:, None]
            # Ensure 1 param from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trials = np.where(mask_cross, mutant, pop)
            
            # 7. Bound Handling (Midpoint Target)
            # If < min, set to (min + parent)/2
            mask_l = trials < min_b
            if np.any(mask_l):
                trials = np.where(mask_l, (min_b + pop) * 0.5, trials)
            
            mask_u = trials > max_b
            if np.any(mask_u):
                trials = np.where(mask_u, (max_b + pop) * 0.5, trials)
                
            # 8. Evaluation & Selection
            succ_f = []
            succ_cr = []
            diffs = []
            
            for i in range(pop_size):
                if i % 10 == 0:
                    if (datetime.now() - start_time) >= time_limit:
                        return global_best_val
                
                f_tri = func(trials[i])
                
                if f_tri <= fitness[i]:
                    # Successful Update
                    if f_tri < fitness[i]:
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diffs.append(fitness[i] - f_tri)
                        archive.append(pop[i].copy())
                    
                    pop[i] = trials[i]
                    fitness[i] = f_tri
                    
                    if f_tri < global_best_val:
                        global_best_val = f_tri
                        global_best_vec = trials[i].copy()
                        stall_counter = 0 # Reset stall on global improvement
            
            # 9. Memory Update
            # Cap archive
            curr_arc_cap = int(pop_size * arc_rate)
            while len(archive) > curr_arc_cap:
                archive.pop(np.random.randint(0, len(archive)))
                
            if len(succ_f) > 0:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                df = np.array(diffs)
                
                # Weights
                w = df / np.sum(df)
                
                # Weighted Lehmer Mean F
                denom = np.sum(w * s_f)
                if denom == 0: m_f = 0.5
                else: m_f = np.sum(w * s_f**2) / denom
                
                # Weighted Mean CR
                m_cr = np.sum(w * s_cr)
                
                mem_f[k_mem] = m_f
                mem_cr[k_mem] = m_cr
                k_mem = (k_mem + 1) % H

    return global_best_val
