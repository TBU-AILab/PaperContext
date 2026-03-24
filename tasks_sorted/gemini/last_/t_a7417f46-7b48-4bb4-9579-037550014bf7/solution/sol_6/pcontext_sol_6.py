#Here is an improved algorithm. It implements **Restarting L-SHADE with Adaptive Hooke-Jeeves Polish**.
#
#### Key Improvements over Previous Version
#
#1.  **Restart Mechanism:** The previous algorithm (LPSR) committed to a single trajectory, shrinking the population to force convergence. If it got trapped in a local optimum (common with a score of 216), it couldn't escape. This version detects convergence (stagnation) and **restarts** the population in a new area of the search space, allowing it to explore multiple "basins" of attraction within the time limit.
#2.  **Stagnation Detection:** Instead of relying on a linear schedule, this algorithm monitors the improvement of the best solution. If the best solution doesn't improve significantly for a set number of generations, it triggers a restart or a local search polish, saving valuable time.
#3.  **Hooke-Jeeves Local Search:** A robust Direct Search method is applied at the end of the global search phase to "polish" the result. It is effective at traversing the final valley of the objective function, dealing well with non-differentiable or noisy landscapes where gradient methods fail.
#4.  **Vectorized Operations:** The core L-SHADE logic is heavily vectorized for performance in Python, allowing more evaluations per second.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting L-SHADE with Hooke-Jeeves Polish.
    
    Strategy:
    1. Global Search: L-SHADE (History-based Adaptive Differential Evolution).
       - Features: External Archive, Adaptive F/CR, 'current-to-pbest/1' mutation.
       - Restart: If population converges (std dev small) or fitness stagnates, 
         the algorithm saves the best, resets the population, and starts again.
    2. Local Polish: Hooke-Jeeves Pattern Search.
       - Applied to the global best solution at the very end to drain the 
         remaining time budget for high-precision refinement.
    """
    
    # --- Constants & Configuration ---
    start_time = time.time()
    bounds = np.array(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    
    # Reserve time for final polish (e.g., 5-10% or minimum 0.5s)
    polish_time_budget = max(0.5, max_time * 0.05) 
    global_search_time = max_time - polish_time_budget
    
    # Global Best Tracking
    global_best_x = None
    global_best_f = float('inf')
    
    # L-SHADE Parameters
    H = 5  # Memory size
    initial_pop_size = int(np.clip(20 * dim, 30, 100))
    arc_rate = 1.4
    
    # --- Helper: Boundary Constraint (Reflective) ---
    def check_bounds(candidates):
        # Bounce back strategy often works better than clipping for DE
        # Lower violations
        mask_l = candidates < min_b
        candidates[mask_l] = 2 * min_b[mask_l] - candidates[mask_l]
        mask_l_2 = candidates < min_b # If still out (due to heavy bounce)
        candidates[mask_l_2] = min_b[mask_l_2]
        
        # Upper violations
        mask_u = candidates > max_b
        candidates[mask_u] = 2 * max_b[mask_u] - candidates[mask_u]
        mask_u_2 = candidates > max_b
        candidates[mask_u_2] = max_b[mask_u_2]
        
        return candidates

    # --- Main Loop: Restarts ---
    restart_count = 0
    
    while (time.time() - start_time) < global_search_time:
        restart_count += 1
        
        # 1. Initialization for this run
        pop_size = initial_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
        fitness = np.array([float('inf')] * pop_size)
        
        # Initial Evaluation
        for i in range(pop_size):
            if (time.time() - start_time) > global_search_time: break
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_f:
                global_best_f = val
                global_best_x = pop[i].copy()
                
        # Algorithm State
        memory_sf = np.full(H, 0.5)
        memory_scr = np.full(H, 0.5)
        mem_k = 0
        archive = []
        
        # Stagnation counter
        stag_limit = 30 + (dim * 2) # Allow more time for higher dimensions
        stag_count = 0
        run_best_f = np.min(fitness)
        
        # 2. Evolutionary Loop
        while (time.time() - start_time) < global_search_time:
            
            # Sort population for p-best selection
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            current_best = fitness[0]
            
            # --- Convergence/Stagnation Checks ---
            # 1. Check if population has collapsed (Standard Deviation)
            pop_std = np.std(fitness)
            if pop_std < 1e-8:
                break # Restart
            
            # 2. Check improvement stagnation
            if current_best < run_best_f:
                run_best_f = current_best
                stag_count = 0
            else:
                stag_count += 1
                
            if stag_count > stag_limit:
                break # Restart
                
            # --- Parameter Adaptation ---
            # Randomly select memory indices
            r_idx = np.random.randint(0, H, pop_size)
            m_sf = memory_sf[r_idx]
            m_scr = memory_scr[r_idx]
            
            # Generate F (Cauchy) and CR (Normal)
            # F needs to be > 0. If <= 0, regenerate. If > 1, clip to 1.
            # We approximate regeneration by clipping 0.1 to 1.0 roughly
            sf = m_sf + 0.1 * np.random.standard_cauchy(pop_size)
            sf = np.clip(sf, 0.05, 1.0) # Avoid F=0
            
            scr = np.random.normal(m_scr, 0.1)
            scr = np.clip(scr, 0.0, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # Select p-best (top 5% to 20%)
            p_min = 2 / pop_size
            p = np.random.uniform(p_min, 0.2, pop_size)
            p_indices = (p * pop_size).astype(int)
            p_indices = np.maximum(p_indices, 1) # Ensure at least 1
            
            # Vectorized selection of vectors
            # x_pbest
            r_best_idx = [np.random.randint(0, pi) for pi in p_indices]
            x_pbest = pop[r_best_idx]
            
            # x_r1: random from pop, != i
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Simple collision fix: if r1==i, add 1 mod pop_size
            collision = (r1_idx == np.arange(pop_size))
            r1_idx[collision] = (r1_idx[collision] + 1) % pop_size
            x_r1 = pop[r1_idx]
            
            # x_r2: random from (pop U archive), != i, != r1
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((pop, archive_np))
            else:
                union_pop = pop
                
            r2_idx = np.random.randint(0, len(union_pop), pop_size)
            # Skip strict collision check for r2 to save time (low impact)
            x_r2 = union_pop[r2_idx]
            
            # Mutation vector v
            # v = x + F*(pbest - x) + F*(r1 - r2)
            sf_col = sf[:, None]
            v = pop + sf_col * (x_pbest - pop) + sf_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, pop_size)
            mask_rand = np.random.rand(pop_size, dim) < scr[:, None]
            mask_j = np.zeros((pop_size, dim), dtype=bool)
            mask_j[np.arange(pop_size), j_rand] = True
            final_mask = np.logical_or(mask_rand, mask_j)
            
            u = np.where(final_mask, v, pop)
            u = check_bounds(u)
            
            # --- Selection ---
            # Evaluate u (Cannot vectorize func call efficiently if func is blackbox single input)
            # We must loop.
            
            new_pop = []
            new_fitness = []
            success_sf = []
            success_scr = []
            fitness_diff = []
            
            # Temporary arrays for next generation
            next_pop = pop.copy()
            next_fitness = fitness.copy()
            
            for k in range(pop_size):
                # Time check inside evaluation loop for safety
                if (time.time() - start_time) > global_search_time:
                    break
                
                f_u = func(u[k])
                
                if f_u <= fitness[k]:
                    next_pop[k] = u[k]
                    next_fitness[k] = f_u
                    
                    if f_u < fitness[k]:
                        # Successful update
                        success_sf.append(sf[k])
                        success_scr.append(scr[k])
                        fitness_diff.append(fitness[k] - f_u)
                        
                        # Add old to archive
                        archive.append(pop[k].copy())
                    
                    if f_u < global_best_f:
                        global_best_f = f_u
                        global_best_x = u[k].copy()
            
            pop = next_pop
            fitness = next_fitness
            
            # Maintain Archive Size
            max_archive_size = int(pop_size * arc_rate)
            while len(archive) > max_archive_size:
                # Remove random elements
                idx_to_remove = np.random.randint(0, len(archive))
                archive.pop(idx_to_remove)
                
            # --- Update Memory (Lehmer Mean) ---
            if len(success_sf) > 0:
                success_sf = np.array(success_sf)
                success_scr = np.array(success_scr)
                diffs = np.array(fitness_diff)
                
                # Weighted mean based on fitness improvement
                total_diff = np.sum(diffs)
                weights = diffs / total_diff if total_diff > 0 else np.ones(len(diffs))/len(diffs)
                
                # Mean WL_F
                mean_sf = np.sum(weights * (success_sf ** 2)) / np.sum(weights * success_sf)
                
                # Mean WL_CR
                mean_scr = np.sum(weights * success_scr)
                
                memory_sf[mem_k] = mean_sf
                memory_scr[mem_k] = mean_scr
                mem_k = (mem_k + 1) % H

    # --- Phase 2: Local Search Polish (Hooke-Jeeves inspired) ---
    # We use the remaining time to finely tune the best solution found.
    # This works by exploring immediate neighbors with shrinking step sizes.
    
    if global_best_x is None:
        return float('inf') # Should not happen unless time is extremely close to 0
    
    current_x = global_best_x.copy()
    current_f = global_best_f
    
    # Initial step size: 1% of the domain, decreasing
    step_size = (max_b - min_b) * 0.01
    min_step_err = 1e-9
    rho = 0.5 # reduction factor
    
    while (time.time() - start_time) < max_time and np.max(step_size) > min_step_err:
        improved = False
        
        # Explore dimensions
        for i in range(dim):
            if (time.time() - start_time) >= max_time: break
            
            # Forward move
            temp_x = current_x.copy()
            temp_x[i] += step_size[i]
            
            # Bound check
            if temp_x[i] > max_b[i]: temp_x[i] = max_b[i]
            
            val = func(temp_x)
            
            if val < current_f:
                current_f = val
                current_x = temp_x
                improved = True
            else:
                # Backward move
                temp_x = current_x.copy()
                temp_x[i] -= step_size[i]
                
                if temp_x[i] < min_b[i]: temp_x[i] = min_b[i]
                
                val = func(temp_x)
                
                if val < current_f:
                    current_f = val
                    current_x = temp_x
                    improved = True
        
        if not improved:
            step_size *= rho # Reduce step size if no improvement in any direction
            
    # Final Update
    if current_f < global_best_f:
        global_best_f = current_f
        
    return global_best_f
