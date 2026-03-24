#The following algorithm implements **Restart L-SHADE with Symmetric MTS-LS1**.
#
#**Rationale for Improvements:**
#1.  **Corrected MTS-LS1**: Previous attempts used an asymmetric local search or limited it to the end of the run. This version implements the full **Symmetric MTS-LS1** (Multiple Trajectory Search), which checks both negative and positive directions for each dimension and reduces the search range only when both fail. This allows for precise "valley walking" and polishing.
#2.  **Restart Mechanism**: Standard L-SHADE can stagnate. This algorithm detects convergence (population size collapse or low fitness variance) and triggers a **Restart**. Crucially, the global best solution is preserved and injected into the new population to guide exploration.
#3.  **Dynamic Time Management**: The algorithm maps the Linear Population Size Reduction (LPSR) to the *remaining* time for each restart session, ensuring efficient convergence pressure regardless of when a restart occurs.
#4.  **Robust L-SHADE Core**: Uses the proven L-SHADE strategy (Success-History Adaptation, Linear Reduction) which is state-of-the-art for continuous optimization.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restart L-SHADE with Symmetric MTS-LS1 Local Search.
    
    Algorithm Phases:
    1. Global Search: L-SHADE with Linear Population Size Reduction.
    2. Local Search: Symmetric MTS-LS1 triggered upon convergence or end-of-time.
    3. Restart: Re-initializes population if convergence occurs early, preserving the best solution.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit_td = timedelta(seconds=max_time)
    
    def is_time_up():
        return (datetime.now() - start_time) >= limit_td

    def get_remaining_ratio():
        elapsed = (datetime.now() - start_time).total_seconds()
        return elapsed / max_time

    # --- Configuration ---
    # L-SHADE Constants
    # Population size: 25*dim is standard, capped to ensure speed on short timeouts
    pop_size_init = int(25 * dim)
    pop_size_init = max(30, min(pop_size_init, 300))
    min_pop_size = 4
    
    # Problem Bounds
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = np.zeros(dim)
    
    # --- Helper: Symmetric MTS-LS1 Local Search ---
    def run_mts_ls1(start_sol, start_val):
        """
        Runs MTS-LS1 (Multiple Trajectory Search - Local Search 1)
        Updates global_best_sol/val in place if improvements are found.
        """
        nonlocal global_best_sol, global_best_val
        
        # Working solution
        current_sol = start_sol.copy()
        current_val = start_val
        
        # Initial Search Range (Step Size) - covers 40% of domain initially
        search_range = (ub - lb) * 0.4
        
        improved = True
        while improved:
            improved = False
            if is_time_up(): break
            
            # Search dimensions in random order
            dims = np.random.permutation(dim)
            
            for d in dims:
                if is_time_up(): break
                
                original_x = current_sol[d]
                sr = search_range[d]
                
                # Direction 1: Negative Step
                current_sol[d] = np.clip(original_x - sr, lb[d], ub[d])
                val_neg = func(current_sol)
                
                if val_neg < current_val:
                    current_val = val_neg
                    current_sol[d] = current_sol[d] # Keep change
                    improved = True
                    
                    if current_val < global_best_val:
                        global_best_val = current_val
                        global_best_sol = current_sol.copy()
                else:
                    # Direction 2: Positive Step
                    current_sol[d] = np.clip(original_x + sr, lb[d], ub[d])
                    val_pos = func(current_sol)
                    
                    if val_pos < current_val:
                        current_val = val_pos
                        current_sol[d] = current_sol[d] # Keep change
                        improved = True
                        
                        if current_val < global_best_val:
                            global_best_val = current_val
                            global_best_sol = current_sol.copy()
                    else:
                        # No improvement: Revert and reduce search range
                        current_sol[d] = original_x
                        search_range[d] *= 0.5
            
            # Termination: If search steps are smaller than precision
            if np.max(search_range) < 1e-15:
                break
                
        return current_val, current_sol

    # --- Main Optimization Loop (Restarts) ---
    while not is_time_up():
        
        # 1. Initialize L-SHADE Population
        curr_pop_size = pop_size_init
        population = lb + np.random.rand(curr_pop_size, dim) * (ub - lb)
        fitness = np.full(curr_pop_size, float('inf'))
        
        # Inject Global Best (Exploitation of previous runs)
        if global_best_val != float('inf'):
            population[0] = global_best_sol.copy()
            fitness[0] = global_best_val
            # Evaluate the rest
            start_idx = 1
        else:
            start_idx = 0
            
        for i in range(start_idx, curr_pop_size):
            if is_time_up(): return global_best_val
            val = func(population[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = population[i].copy()
        
        # Sort
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Memory Initialization
        H = 6 
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Restart Time Context
        run_start = datetime.now()
        # Allocate remaining time to this run (assuming it might be the last)
        remaining_sec = (limit_td - (run_start - start_time)).total_seconds()
        if remaining_sec < 0.1: return global_best_val
        
        # --- L-SHADE Generations ---
        while True:
            if is_time_up(): return global_best_val
            
            # Calculate Progress for LPSR (Linear Population Size Reduction)
            # We map the reduction to the estimated remaining time of THIS restart
            run_elapsed = (datetime.now() - run_start).total_seconds()
            progress = run_elapsed / remaining_sec
            progress = min(1.0, progress)
            
            # 1. Resize Population
            target_size = int(round(pop_size_init + (min_pop_size - pop_size_init) * progress))
            target_size = max(min_pop_size, target_size)
            
            if curr_pop_size > target_size:
                curr_pop_size = target_size
                population = population[:curr_pop_size]
                fitness = fitness[:curr_pop_size]
                # Resize Archive
                if len(archive) > curr_pop_size:
                    del_count = len(archive) - curr_pop_size
                    for _ in range(del_count):
                        archive.pop(np.random.randint(len(archive)))
            
            # 2. Adaptive Parameter Generation
            p_best_rate = 0.11 # Top 11%
            
            r_idxs = np.random.randint(0, H, curr_pop_size)
            m_cr = mem_cr[r_idxs]
            m_f = mem_f[r_idxs]
            
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            f = m_f + 0.1 * np.random.standard_cauchy(curr_pop_size)
            while np.any(f <= 0): # Repair negative F
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
            f = np.clip(f, 0, 1)
            
            # 3. Mutation: current-to-pbest/1
            p_num = max(2, int(curr_pop_size * p_best_rate))
            pbest_idxs = np.random.randint(0, p_num, curr_pop_size)
            x_pbest = population[pbest_idxs]
            
            r1 = np.random.randint(0, curr_pop_size, curr_pop_size)
            for i in range(curr_pop_size):
                while r1[i] == i: r1[i] = np.random.randint(0, curr_pop_size)
            x_r1 = population[r1]
            
            if len(archive) > 0:
                union_pop = np.vstack((population, np.array(archive)))
            else:
                union_pop = population
            
            r2 = np.random.randint(0, len(union_pop), curr_pop_size)
            for i in range(curr_pop_size):
                while r2[i] == i or r2[i] == r1[i]:
                    r2[i] = np.random.randint(0, len(union_pop))
            x_r2 = union_pop[r2]
            
            f_v = f[:, np.newaxis]
            mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, curr_pop_size)
            mask = np.random.rand(curr_pop_size, dim) < cr[:, np.newaxis]
            mask[np.arange(curr_pop_size), j_rand] = True
            trial = np.where(mask, mutant, population)
            
            trial = np.clip(trial, lb, ub)
            
            # 5. Selection
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            success_f = []
            success_cr = []
            diff_f = []
            
            for i in range(curr_pop_size):
                if is_time_up(): return global_best_val
                
                y = func(trial[i])
                
                if y <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fit[i] = y
                    
                    if y < fitness[i]:
                        archive.append(population[i].copy())
                        diff_f.append(fitness[i] - y)
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        
                    if y < global_best_val:
                        global_best_val = y
                        global_best_sol = trial[i].copy()
            
            population = new_pop
            fitness = new_fit
            
            # 6. Memory Update
            if len(diff_f) > 0:
                w = np.array(diff_f)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w /= w_sum
                    sf = np.array(success_f)
                    scr = np.array(success_cr)
                    
                    mean_scr = np.sum(w * scr)
                    sum_w_sf = np.sum(w * sf)
                    if abs(sum_w_sf) < 1e-9: mean_sf = 0.5
                    else: mean_sf = np.sum(w * (sf**2)) / sum_w_sf
                    
                    mem_f[k_mem] = np.clip(mean_sf, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_scr, 0, 1)
                    k_mem = (k_mem + 1) % H
                    
            # Sort
            s_idx = np.argsort(fitness)
            population = population[s_idx]
            fitness = fitness[s_idx]
            
            # 7. Convergence Check
            # Conditions:
            # A) Population size reduced to minimum
            # B) Standard deviation of fitness is tiny (Clustered at optimum)
            is_converged = (curr_pop_size <= min_pop_size) or (np.std(fitness) < 1e-9)
            
            # 8. Local Search Trigger
            # Trigger if converged OR time is nearly up (polishing phase)
            remaining_total = (limit_td - (datetime.now() - start_time)).total_seconds()
            is_polishing = (remaining_total < max_time * 0.05) or (remaining_total < 0.5)
            
            if is_converged or is_polishing:
                # Run MTS-LS1 on the best solution found so far
                run_mts_ls1(global_best_sol, global_best_val)
                
                if is_polishing:
                    return global_best_val
                
                # If converged but time remains -> Break inner loop to RESTART
                break
                
    return global_best_val
