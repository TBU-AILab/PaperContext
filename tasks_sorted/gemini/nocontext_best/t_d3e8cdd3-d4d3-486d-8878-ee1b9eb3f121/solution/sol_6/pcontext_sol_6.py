#Here is a fully operational, self-contained implementation.
#
#This algorithm uses **jSO (a specialized variant of L-SHADE)** combined with a **Restart Strategy** and **MTS-LS1 (Multiple Trajectory Search - Local Search 1)**.
#
#### Key Improvements over the previous submission:
#1.  **Restart Strategy**: The previous algorithm likely got stuck in a local optimum (value ~5.08). This version detects stagnation (low population variance or stagnant fitness) and performs a "soft restart" (keeping the best solution, resetting the rest with increased exploration pressure).
#2.  **MTS-LS1 Local Search**: Instead of simple coordinate descent, this uses MTS-LS1, a robust local search method often used in winning CEC (Congress on Evolutionary Computation) algorithms. It uses dynamic step sizes per dimension, allowing it to navigate narrow valleys efficiently.
#3.  **jSO Adaptation**: It implements the specific "current-to-pbest-w/1" mutation strategy and weighted memory updates from the jSO algorithm (a highly ranked improvement on L-SHADE), which balances exploration and exploitation better than standard DE.
#4.  **Time-Aware Scheduling**: The algorithm partitions the remaining time dynamically, reserving a specific slice for final local search polishing while allowing the evolutionary cycle to restart multiple times if the function is easy or deceptive.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using jSO (L-SHADE variant) with Restart capabilities 
    and MTS-LS1 Local Search polishing.
    """
    
    # --- Configuration ---
    start_time = time.time()
    
    # Helper for bounds
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Check time helper
    def check_time():
        return (time.time() - start_time)
        
    def is_time_left(safety_margin=0.0):
        return check_time() < (max_time - safety_margin)

    # Global Best Tracking
    g_best_val = float('inf')
    g_best_pos = np.random.uniform(lb, ub) # placeholder

    # --- 1. MTS-LS1 Local Search Function ---
    # A robust local search that maintains step sizes for each dimension
    def mts_ls1(best_pos, best_val, available_time_budget):
        ls_start = time.time()
        
        # Initialize search range (SR)
        sr = (ub - lb) * 0.4
        current_pos = best_pos.copy()
        current_val = best_val
        
        # Search directions and improvement flags
        # We perform a streamlined version of MTS-LS1
        dim_indices = np.arange(dim)
        
        # Don't spend more than budget
        while (time.time() - ls_start) < available_time_budget:
            improved = False
            np.random.shuffle(dim_indices) # Randomize order of dimensions
            
            for i in dim_indices:
                if (time.time() - ls_start) >= available_time_budget:
                    break
                
                original_val_i = current_pos[i]
                
                # Try negative direction
                current_pos[i] = np.clip(original_val_i - sr[i], lb[i], ub[i])
                val = func(current_pos)
                
                if val < current_val:
                    current_val = val
                    best_val = val # update reference
                    improved = True
                else:
                    # Try positive direction
                    current_pos[i] = np.clip(original_val_i + 0.5 * sr[i], lb[i], ub[i])
                    val = func(current_pos)
                    
                    if val < current_val:
                        current_val = val
                        best_val = val
                        improved = True
                    else:
                        # Restore
                        current_pos[i] = original_val_i
            
            if not improved:
                sr /= 2.0 # Reduce search radius if no improvement found in full pass
                if np.max(sr) < 1e-15: # Converged
                    break
            else:
                # If improved, copy back to best_pos
                best_pos[:] = current_pos[:]
                
        return current_pos, current_val

    # --- 2. Main Optimization Loop (Restarts) ---
    # We treat the problem as a sequence of "Epochs". 
    # If an epoch converges, we restart.
    
    # Initial Parameters for jSO/L-SHADE
    restart_count = 0
    
    while is_time_left(safety_margin=0.05 * max_time):
        
        # --- Initialization per Epoch ---
        # Population Size (Linear Reduction strategy vars)
        pop_size_init = int(25 * dim * np.sqrt(restart_count + 1)) # Increase pop slightly on restarts
        pop_size_init = min(pop_size_init, 500) # Cap max init size
        pop_size_end = 4
        pop_size = pop_size_init
        
        # Archives and Memory
        archive = []
        H = 5 # Memory size
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Population
        population = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, float('inf'))
        
        # If we have a global best from previous restart, inject it
        if restart_count > 0:
            population[0] = g_best_pos.copy()
            
        # Evaluate Initial Population
        for i in range(pop_size):
            if not is_time_left(): break
            fitness[i] = func(population[i])
            if fitness[i] < g_best_val:
                g_best_val = fitness[i]
                g_best_pos = population[i].copy()
                
        # Epoch Loop
        max_generations_est = 1000 # dynamic estimate
        gen = 0
        
        # Stagnation counter
        stagnation_counter = 0
        last_best_fit_in_epoch = g_best_val
        
        while is_time_left():
            # Time & Progress Calculation
            elapsed = check_time()
            # Calculate progress relative to "Generations" or "Time"
            # Since max_time is the hard constraint, we drive reduction by Time.
            # However, for restarts, we map current restart progress [0, 1]
            
            # Linear Population Size Reduction (LPSR)
            # We assume this "Epoch" aims to last until max_time if it doesn't converge.
            # So progress is (now - epoch_start) / (max_time - epoch_start) roughly.
            # To simplify, we calculate reduction based on current population vs target.
            
            p_remain = (max_time - elapsed) / max_time
            # Non-linear progress curve for more aggressive early exploration
            progress = 1.0 - p_remain 
            
            # Calculate Target Population Size
            pop_target = int(round((pop_size_end - pop_size_init) * progress + pop_size_init))
            pop_target = max(pop_size_end, pop_target)
            
            # Reduction
            if pop_size > pop_target:
                n_reduce = pop_size - pop_target
                sort_idx = np.argsort(fitness)
                population = population[sort_idx]
                fitness = fitness[sort_idx]
                
                # Remove worst
                population = population[:-n_reduce]
                fitness = fitness[:-n_reduce]
                pop_size = pop_target
                
                # Resize Archive
                if len(archive) > pop_size:
                    # Resize archive randomly
                    del_count = len(archive) - pop_size
                    # Simple list reduction
                    for _ in range(del_count):
                        archive.pop(np.random.randint(len(archive)))

            # --- jSO Parameter Generation ---
            # Sort for p-best selection
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # p-best value for mutation (decreases linearly)
            # from 0.25 to 0.05 roughly
            p_val = 0.25 - (0.20 * progress)
            p_val = max(0.05, p_val)
            
            # Generate F and CR
            idx_r = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[idx_r]
            m_f = mem_f[idx_r]
            
            # Cauchy F
            f = np.random.standard_cauchy(pop_size) * 0.1 + m_f
            f = np.clip(f, 0, 1)
            f[f <= 0] = 0.5 # Regenerate/fix (simple fix)
            
            # Normal CR
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            # jSO logic: CR sorts towards median for some gens, but random is robust enough here
            
            # --- Mutation: current-to-pbest-w/1 ---
            # v = x + Fw(x_pbest - x) + F(x_r1 - x_r2)
            # Note: "w" logic in jSO is F_w = F if F < 0.7 else 0.7. Simple F is used here for speed.
            
            # Select pbest indices
            num_pbest = max(1, int(p_val * pop_size))
            pbest_indices = np.random.randint(0, num_pbest, pop_size)
            # Since pop is sorted, indices 0..num_pbest are the top
            x_pbest = population[pbest_indices]
            
            # Select r1, r2
            # r1 != i
            r1_indices = np.random.randint(0, pop_size, pop_size)
            same_mask = (r1_indices == np.arange(pop_size))
            r1_indices[same_mask] = (r1_indices[same_mask] + 1) % pop_size
            x_r1 = population[r1_indices]
            
            # r2 != r1, r2 != i, from Pop U Archive
            if len(archive) > 0:
                archive_np = np.array(archive)
                pool = np.vstack((population, archive_np))
            else:
                pool = population
                
            pool_size = len(pool)
            r2_indices = np.random.randint(0, pool_size, pop_size)
            # Collision handling skipped for vectorization speed (minor impact)
            x_r2 = pool[r2_indices]
            
            # Compute Mutant
            # Broadcasting F
            F_col = f[:, np.newaxis]
            
            # Specific jSO weight factor Fw for the first difference vector
            # Fw = F if F < 0.7 else 0.7 (approx)
            Fw_col = np.where(F_col < 0.7, F_col, 0.7)
            
            mutants = population + Fw_col * (x_pbest - population) + F_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_j = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            
            CR_col = cr[:, np.newaxis]
            mask = (rand_j <= CR_col) | j_rand_mask
            trial_vecs = np.where(mask, mutants, population)
            
            # Bound Constraint (Clip)
            trial_vecs = np.clip(trial_vecs, lb, ub)
            
            # --- Selection ---
            # Evaluate trials
            new_fitness = np.zeros(pop_size)
            success_mask = np.zeros(pop_size, dtype=bool)
            diff_f = np.zeros(pop_size)
            
            # We must loop for func calls, but we can check time
            for i in range(pop_size):
                if not is_time_left(): break
                
                t_val = func(trial_vecs[i])
                
                if t_val < fitness[i]:
                    new_fitness[i] = t_val
                    success_mask[i] = True
                    diff_f[i] = fitness[i] - t_val
                    
                    # Add replaced to archive
                    archive.append(population[i].copy())
                    
                    # Update Pop
                    population[i] = trial_vecs[i]
                    fitness[i] = t_val
                    
                    if t_val < g_best_val:
                        g_best_val = t_val
                        g_best_pos = trial_vecs[i].copy()
                        stagnation_counter = 0 # Reset stagnation
                else:
                    new_fitness[i] = fitness[i]
            
            # Archive maintenance
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
                
            # --- Memory Update (Weighted Lehmer Mean) ---
            if np.any(success_mask):
                succ_f = f[success_mask]
                succ_cr = cr[success_mask]
                w = diff_f[success_mask]
                w_sum = np.sum(w)
                
                if w_sum > 0:
                    w = w / w_sum
                    
                    # Mean WL for F
                    mean_f_l = np.sum(w * (succ_f**2)) / np.sum(w * succ_f)
                    
                    # Mean WL for CR
                    # jSO uses special logic for CR (mean of squares? no, usually weighted arithmetic)
                    mean_cr_l = np.sum(w * succ_cr)
                    
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f_l
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr_l
                    
                    k_mem = (k_mem + 1) % H
            
            # --- Convergence/Restart Detection ---
            # 1. Population Collapse
            pop_std = np.mean(np.std(population, axis=0))
            if pop_std < 1e-12:
                break # Break inner loop -> Trigger restart
                
            # 2. Fitness Stagnation
            if abs(g_best_val - last_best_fit_in_epoch) < 1e-12:
                stagnation_counter += 1
            else:
                last_best_fit_in_epoch = g_best_val
            
            # Trigger restart if stagnant for too long (relative to pop size)
            if stagnation_counter > 20 + (dim * 2):
                break
                
            gen += 1
            
        restart_count += 1
        
        # Before restarting, check if we need to do a quick polish?
        # We leave the heavy polishing for the very end of max_time.

    # --- 3. Final Polishing Phase ---
    # Use remaining time for MTS-LS1 on the global best
    remaining = (max_time - check_time())
    if remaining > 0.01:
        # Perform Local Search
        g_best_pos, g_best_val = mts_ls1(g_best_pos, g_best_val, remaining)
        
    return g_best_val
