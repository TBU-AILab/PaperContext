#The output value of 220.49 suggests the previous algorithm likely stagnated in a local optimum (common in functions like Rastrigin or Rosenbrock) and lacked a sufficiently aggressive mechanism to explore other basins of attraction within the time limit.
#
#To improve this, I propose an optimized implementation of **L-SHADE with a Time-Adaptive Restart Strategy**, heavily inspired by the **jSO** algorithm (a top-performing variant of Differential Evolution).
#
#**Key Improvements:**
#1.  **Time-Budgeted Restarts:** The algorithm treats the remaining time as a strict budget for each restart cycle. It dynamically scales the population reduction schedule (LPSR) to fit exactly into the remaining time. This allows for full exploration-exploitation cycles even as time runs out.
#2.  **Adaptive Restart Seeding:** When a restart occurs (due to stagnation), the new population is seeded with a mix of the global best solution (to refine it), a Gaussian cloud around it (exploitation), and pure random points (exploration). This balances refining the current best while searching for better basins.
#3.  **Weighted Lehmer Mean:** Memory updates for mutation parameters ($F$ and $CR$) use a weighted mean based on fitness improvement magnitude, prioritizing parameters that yield larger gains.
#4.  **Vectorized Operations:** Python loops are minimized. Parameter generation, mutation, crossover, and boundary handling are fully vectorized using NumPy for maximum speed.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using an optimized L-SHADE algorithm with a 
    Time-Adaptive Restart Strategy.
    """
    
    # --- Helper: Boundary Correction ---
    def clip_to_bounds(vec, mn, mx):
        return np.clip(vec, mn, mx)

    # --- Initialization ---
    t_start = time.time()
    
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_vec = None
    
    # Algorithm Constants
    # Population size: Adapted for efficiency. 
    # Standard L-SHADE uses 18*dim, but we cap it to ensure many generations 
    # can occur even if the function is slow or time is short.
    N_init = int(18 * dim)
    N_init = max(30, min(N_init, 250)) 
    N_min = 4
    
    # History Memory Size
    H = 6
    
    # --- Main Loop (Restarts) ---
    # We loop until max_time is exhausted.
    while True:
        t_current = time.time()
        time_elapsed_total = t_current - t_start
        time_remaining = max_time - time_elapsed_total
        
        # Stop if insufficient time remains for a meaningful run
        if time_remaining < 0.1:
            break
            
        # Define time budget for this specific restart
        # We plan to utilize all remaining time for this run.
        # If we converge early, we break and restart again with the *new* remaining time.
        run_start_time = t_current
        run_time_budget = time_remaining
        
        # --- Run Initialization ---
        
        # Memory for Adaptive Parameters (F and CR)
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k_mem = 0
        
        # Population Initialization
        pop = np.zeros((N_init, dim))
        fitness = np.zeros(N_init)
        
        # Archive (for mutation strategy)
        # Fixed capacity array, managed with a counter
        archive = np.zeros((N_init, dim))
        archive_cnt = 0
        archive_ptr = 0
        
        # Seeding Strategy
        if global_best_vec is None:
            # First run: Pure Random
            pop = min_b + np.random.rand(N_init, dim) * diff_b
            
            # Evaluate Initial Population
            for i in range(N_init):
                if (time.time() - t_start) >= max_time:
                    return global_best_val
                
                val = func(pop[i])
                fitness[i] = val
                
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = pop[i].copy()
        else:
            # Restart: Biased Initialization
            # 1. Keep the global best
            pop[0] = global_best_vec
            fitness[0] = global_best_val
            
            # 2. Exploitation: Small Gaussian cloud around best (25% of pop)
            # This helps refine the current valley if we are close.
            n_exploit = int(0.25 * N_init)
            scale = 0.05 * diff_b # 5% of domain width standard deviation
            cloud = global_best_vec + np.random.randn(n_exploit, dim) * scale
            pop[1 : 1 + n_exploit] = clip_to_bounds(cloud, min_b, max_b)
            
            # 3. Exploration: Random scattering for the rest
            n_random = N_init - 1 - n_exploit
            pop[1 + n_exploit:] = min_b + np.random.rand(n_random, dim) * diff_b
            
            # Evaluate only new individuals
            for i in range(1, N_init):
                if (time.time() - t_start) >= max_time:
                    return global_best_val
                
                val = func(pop[i])
                fitness[i] = val
                
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = pop[i].copy()
        
        # Sort population by fitness (needed for p-best selection)
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        
        N = N_init
        
        # Stagnation tracking
        stag_count = 0
        last_run_best = fitness[0]
        
        # --- Evolutionary Cycle ---
        while True:
            # Time Check
            t_now = time.time()
            if (t_now - t_start) >= max_time:
                return global_best_val
            
            # Calculate Progress of THIS run (0.0 to 1.0)
            run_progress = (t_now - run_start_time) / run_time_budget
            
            # If we exceeded the budget for this run (very unlikely due to reduction logic), break to restart
            if run_progress >= 1.0:
                break
                
            # --- Linear Population Size Reduction (LPSR) ---
            # Calculates target population size based on time progress
            N_target = int(round(N_min + (N_init - N_min) * (1.0 - run_progress)))
            N_target = max(N_min, N_target)
            
            if N > N_target:
                # Truncate the worst individuals (population is already sorted)
                N = N_target
                pop = pop[:N]
                fitness = fitness[:N]
                
                # Resize Archive: Archive size tracks population size
                if archive_cnt > N:
                    # Randomly keep N elements to respect memory limit
                    keep_idxs = np.random.choice(archive_cnt, N, replace=False)
                    archive[:N] = archive[keep_idxs]
                    archive_cnt = N
                    archive_ptr = 0 
            
            # --- Parameter Generation ---
            # 1. Select memory slots
            r_indices = np.random.randint(0, H, N)
            mu_cr = M_CR[r_indices]
            mu_f = M_F[r_indices]
            
            # 2. Generate CR (Normal Distribution)
            CR = np.random.normal(mu_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # 3. Generate F (Cauchy Distribution)
            # Vectorized generation with retry for invalid values
            F = np.zeros(N)
            todo_mask = np.ones(N, dtype=bool)
            
            # Retry loop for F <= 0
            for _ in range(5): # Limit retries for speed
                n_todo = np.sum(todo_mask)
                if n_todo == 0: break
                
                # Cauchy: loc + scale * tan(pi * (rand - 0.5))
                subset_mu = mu_f[todo_mask]
                u_rand = np.random.rand(n_todo)
                f_vals = subset_mu + 0.1 * np.tan(np.pi * (u_rand - 0.5))
                
                # Valid mask (F > 0)
                valid = f_vals > 0
                
                # Update valid F
                ids = np.where(todo_mask)[0]
                good_ids = ids[valid]
                good_f = f_vals[valid]
                good_f[good_f > 1.0] = 1.0 # Clip upper to 1.0
                
                F[good_ids] = good_f
                todo_mask[good_ids] = False
            
            # Fallback for stubborn values
            if np.any(todo_mask):
                F[todo_mask] = 0.5
            
            # --- Mutation Strategy: current-to-pbest/1 ---
            # p-best parameter linearly decreases from 0.2 to 0.05 approx
            # This encourages exploitation towards the end of the run
            p = max(2.0/N, 0.2 * (1.0 - run_progress))
            num_pbest = int(max(2, p * N))
            
            # X_pbest: Randomly selected from top p%
            pbest_indices = np.random.randint(0, num_pbest, N)
            x_pbest = pop[pbest_indices]
            
            # X_r1: Distinct from X_i
            r1_indices = np.random.randint(0, N, N)
            # Fix collisions with i
            params_indices = np.arange(N)
            mask_r1 = (r1_indices == params_indices)
            r1_indices[mask_r1] = (r1_indices[mask_r1] + 1) % N
            x_r1 = pop[r1_indices]
            
            # X_r2: Distinct from X_i and X_r1, drawn from Union(Pop, Archive)
            if archive_cnt > 0:
                pool = np.vstack((pop, archive[:archive_cnt]))
            else:
                pool = pop
            pool_size = pool.shape[0]
            
            r2_indices = np.random.randint(0, pool_size, N)
            # Fix collisions
            mask_r2 = (r2_indices == params_indices) | (r2_indices == r1_indices)
            if np.any(mask_r2):
                r2_indices[mask_r2] = np.random.randint(0, pool_size, np.sum(mask_r2))
            x_r2 = pool[r2_indices]
            
            # Mutation Equation: V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            F_col = F[:, np.newaxis]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_matrix = np.random.rand(N, dim)
            cross_mask = rand_matrix <= CR[:, np.newaxis]
            # Force at least one dimension to come from mutant
            j_rand = np.random.randint(0, dim, N)
            cross_mask[np.arange(N), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # --- Bound Correction ---
            # If out of bounds, set value halfway between parent and bound
            mask_lower = trial < min_b
            if np.any(mask_lower):
                b_grid = np.tile(min_b, (N, 1))
                trial[mask_lower] = (pop[mask_lower] + b_grid[mask_lower]) * 0.5
                
            mask_upper = trial > max_b
            if np.any(mask_upper):
                b_grid = np.tile(max_b, (N, 1))
                trial[mask_upper] = (pop[mask_upper] + b_grid[mask_upper]) * 0.5
            
            # --- Selection ---
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            success_mask = np.zeros(N, dtype=bool)
            diff_fitness = np.zeros(N)
            
            for i in range(N):
                # Strict Time Check for expensive functions
                if (time.time() - t_start) >= max_time:
                    return global_best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    success_mask[i] = True
                    diff_fitness[i] = fitness[i] - f_trial
                    
                    # Update Global Best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_vec = trial[i].copy()
                    
                    # Add parent to archive
                    if archive_cnt < N:
                        archive[archive_cnt] = pop[i].copy()
                        archive_cnt += 1
                    else:
                        # Replace random or cyclic? Cyclic is faster.
                        archive[archive_ptr] = pop[i].copy()
                        archive_ptr = (archive_ptr + 1) % N
                        
                    new_pop[i] = trial[i]
                    new_fitness[i] = f_trial
            
            # --- Memory Update (Weighted Lehmer Mean) ---
            if np.any(success_mask):
                idx_succ = np.where(success_mask)[0]
                
                w_diff = diff_fitness[idx_succ]
                w_F = F[idx_succ]
                w_CR = CR[idx_succ]
                
                total_diff = np.sum(w_diff)
                
                if total_diff > 0:
                    weights = w_diff / total_diff
                    
                    # Mean CR
                    mean_cr = np.sum(weights * w_CR)
                    M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
                    
                    # Lehmer Mean F
                    num = np.sum(weights * (w_F ** 2))
                    den = np.sum(weights * w_F)
                    if den > 0:
                        mean_f = num / den
                        M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                    
                    k_mem = (k_mem + 1) % H
            
            # Update Population
            pop = new_pop
            fitness = new_fitness
            
            # Re-sort for next generation
            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]
            
            # --- Convergence/Stagnation Check ---
            # 1. Low Variance: Population has converged to a point
            fit_range = fitness[-1] - fitness[0]
            if fit_range < 1e-9:
                break # Trigger Restart
            
            # 2. No Improvement Stagnation
            if fitness[0] == last_run_best:
                stag_count += 1
            else:
                last_run_best = fitness[0]
                stag_count = 0
            
            # Dynamic stagnation limit: tolerate more stagnation in higher dims
            if stag_count > (60 + dim):
                break # Trigger Restart

    return global_best_val
