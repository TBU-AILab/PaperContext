#The following code implements an improved optimization algorithm based on **jSO** (a specific improvement of L-SHADE) combined with **Latin Hypercube Sampling (LHS)** initialization and a **Restart Mechanism**.
#
#**Key Improvements:**
#1.  **Latin Hypercube Sampling (LHS)**: Replaces random initialization to ensure the initial population covers the search space more evenly, reducing the chance of starting in a poor basin of attraction.
#2.  **Restart Strategy**: If the population converges (standard deviation of fitness becomes negligible) before the time limit, the algorithm triggers a "soft restart." It keeps the best solution found so far, generates a fresh diverse population using LHS, and continues searching. This prevents the algorithm from idling if it gets stuck in a local optimum early.
#3.  **Boundary Reflection**: Instead of clipping values to bounds (which piles solutions on the edges), it uses reflection. This helps maintain diversity near boundaries.
#4.  **Dynamic Parameters (jSO style)**: The `p` parameter (for selecting the top best individuals) linearly decreases over time, shifting focus from exploration (high `p`) to exploitation (low `p`).
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using an enhanced L-SHADE algorithm (jSO-like)
    with Latin Hypercube Sampling initialization and a Restart strategy.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Helper: Check time budget
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # Helper: Latin Hypercube Sampling for better initial coverage
    def lhs_init(n_samples, n_dim, bounds_arr):
        lower_bounds = bounds_arr[:, 0]
        upper_bounds = bounds_arr[:, 1]
        diff = upper_bounds - lower_bounds
        
        # Generate stratified samples
        grid = np.arange(n_samples).reshape(-1, 1)
        perm_grid = np.zeros((n_samples, n_dim))
        
        for d in range(n_dim):
            perm_grid[:, d] = np.random.permutation(grid[:, 0])
            
        jitter = np.random.rand(n_samples, n_dim)
        samples = (perm_grid + jitter) / n_samples
        return lower_bounds + samples * diff

    # Helper: Reflective Bound Handling (better than clipping)
    def apply_bounds_reflect(x, low, high):
        # Reflect lower violations
        bad_low = x < low
        x[bad_low] = 2 * low[bad_low] - x[bad_low]
        # If still bad (double bounce), clip
        x[x < low] = low[x < low]
        
        # Reflect upper violations
        bad_high = x > high
        x[bad_high] = 2 * high[bad_high] - x[bad_high]
        x[x > high] = high[x > high]
        return x

    # --- Configuration ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Population sizing
    # Start large for exploration, reduce linearly
    initial_pop_size = int(max(30, min(200, 18 * dim)))
    min_pop_size = 4
    
    history_size = 5 # Size of memory for F and CR
    
    # Global Best Tracking
    best_global_val = float('inf')
    best_global_vec = None
    
    # --- Main Optimization Loop (supports Restarts) ---
    # We loop until time runs out. The inner loop handles the evolution.
    # If convergence is detected, we break inner loop and restart.
    
    restart_count = 0
    
    while not check_time():
        
        # --- Initialization Phase (Restart or First Start) ---
        pop_size = initial_pop_size
        
        # Reduce initial pop size slightly on restarts to save evaluations
        if restart_count > 0:
            pop_size = max(min_pop_size * 2, int(initial_pop_size * 0.7))
            
        pop = lhs_init(pop_size, dim, bounds_np)
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return best_global_val
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_global_val:
                best_global_val = val
                best_global_vec = pop[i].copy()
        
        # Inject global best if restarting (elitism)
        if restart_count > 0 and best_global_vec is not None:
            pop[0] = best_global_vec
            fitness[0] = best_global_val

        # Memory for parameters (L-SHADE specific)
        memory_sf = np.full(history_size, 0.5)
        memory_scr = np.full(history_size, 0.5)
        mem_k = 0
        
        archive = []
        
        # --- Evolution Phase ---
        while not check_time():
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate progress based on TIME, not generations
            elapsed_sec = (datetime.now() - start_time).total_seconds()
            progress = min(1.0, elapsed_sec / max_time)
            
            # Dynamic reduction target
            target_pop_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
            target_pop_size = max(min_pop_size, target_pop_size)
            
            if pop_size > target_pop_size:
                # Reduce population: keep best
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:target_pop_size]]
                fitness = fitness[sort_idx[:target_pop_size]]
                pop_size = target_pop_size
                
                # Resize archive
                if len(archive) > pop_size:
                    import random
                    random.shuffle(archive)
                    archive = archive[:pop_size]

            # 2. Convergence / Stagnation Check (Trigger Restart)
            # If population fitness variance is extremely low, we are stuck.
            if pop_size >= min_pop_size:
                fit_std = np.std(fitness)
                fit_range = np.max(fitness) - np.min(fitness)
                # If converged, break to outer loop -> triggers restart
                if fit_std < 1e-9 or fit_range < 1e-9:
                    break

            # 3. Parameter Generation
            # Dynamic p-best rate (jSO strategy): decreases from 0.25 to 0.11 implies shift to exploitation
            p_val = 0.25 - (0.25 - 0.11) * progress
            p_val = max(0.05, p_val) # Safety floor
            
            r_indices = np.random.randint(0, history_size, pop_size)
            mu_sf = memory_sf[r_indices]
            mu_scr = memory_scr[r_indices]
            
            # Generate CR (Normal dist)
            cr = np.random.normal(mu_scr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F (Cauchy dist)
            f = np.zeros(pop_size)
            for i in range(pop_size):
                while True:
                    val = mu_sf[i] + 0.1 * np.random.standard_cauchy()
                    if val > 0:
                        if val > 1: val = 1.0
                        f[i] = val
                        break

            # 4. Mutation & Crossover
            # Sorting for p-best selection
            sorted_indices = np.argsort(fitness)
            
            # Prepare Union for r2
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
                
            trials = np.zeros_like(pop)
            
            # Vectorized-ish loop for clarity and specific index exclusion
            # current-to-pbest/1 strategy
            num_pbest = max(2, int(p_val * pop_size))
            pbest_pool = sorted_indices[:num_pbest]
            
            for i in range(pop_size):
                pbest_idx = np.random.choice(pbest_pool)
                x_pbest = pop[pbest_idx]
                
                # r1 != i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # r2 != i, r2 != r1
                r2 = i
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, len(union_pop))
                x_r2 = union_pop[r2]
                
                mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (x_r1 - x_r2)
                
                # Crossover
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) <= cr[i]
                mask[j_rand] = True
                
                trial = np.where(mask, mutant, pop[i])
                trials[i] = apply_bounds_reflect(trial, min_b, max_b)

            # 5. Evaluation and Selection
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            success_sf = []
            success_scr = []
            diff_f = []
            
            for i in range(pop_size):
                if check_time(): return best_global_val
                
                t_val = func(trials[i])
                
                if t_val <= fitness[i]:
                    if t_val < fitness[i]:
                        archive.append(pop[i].copy())
                        success_sf.append(f[i])
                        success_scr.append(cr[i])
                        diff_f.append(fitness[i] - t_val)
                    
                    new_pop[i] = trials[i]
                    new_fitness[i] = t_val
                    
                    if t_val < best_global_val:
                        best_global_val = t_val
                        best_global_vec = trials[i].copy()
            
            pop = new_pop
            fitness = new_fitness
            
            # Manage Archive
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
                
            # 6. Memory Update (Weighted Lehmer Mean)
            if len(success_sf) > 0:
                s_sf = np.array(success_sf)
                s_scr = np.array(success_scr)
                w_diff = np.array(diff_f)
                
                # Normalize weights
                if np.sum(w_diff) > 0:
                    w = w_diff / np.sum(w_diff)
                    mean_sf = np.sum(w * (s_sf ** 2)) / np.sum(w * s_sf)
                    mean_scr = np.sum(w * s_scr)
                else:
                    mean_sf = np.mean(s_sf)
                    mean_scr = np.mean(s_scr)
                
                # Conservative update
                memory_sf[mem_k] = 0.5 * memory_sf[mem_k] + 0.5 * mean_sf
                memory_scr[mem_k] = 0.5 * memory_scr[mem_k] + 0.5 * mean_scr
                mem_k = (mem_k + 1) % history_size

        # End of inner loop (Restart Triggered)
        restart_count += 1
        
    return best_global_val
