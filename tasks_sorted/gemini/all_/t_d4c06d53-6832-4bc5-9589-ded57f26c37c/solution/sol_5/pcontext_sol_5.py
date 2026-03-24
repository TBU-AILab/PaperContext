#The following algorithm implements **Restarting L-SHADE with Coordinate Descent Polishing**.
#
#### Rationale for Improvement
#The previous best result (16.14) was achieved by **Restarting L-SHADE** with a simple Gaussian polishing. To further minimize the output, this version refines three critical components:
#1.  **Coordinate Descent Polishing (MTS-LS1)**: Instead of a random Gaussian walk (which can be inefficient in high dimensions), the polishing phase now uses a deterministic **Coordinate Descent** strategy (inspired by MTS-LS1). This systematically refines each dimension, draining local basins of attraction much more effectively than random sampling.
#2.  **Adaptive Population Sizing**: The population size is slightly increased (`10 * dim`, capped at 100) compared to the previous version to prevent premature convergence in the early phases, while the restart mechanism ensures we don't waste time in stagnant basins.
#3.  **Robust Restart Trigger**: The restart condition monitors both population variance (convergence) and fitness improvement (stagnation). Upon triggering, the algorithm performs a focused local search on the global best solution to ensure no precision is left on the table before re-initializing the population to explore new areas.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting L-SHADE with Coordinate Descent Polishing.
    
    Mechanism:
    1. Global Search: L-SHADE (History-based parameter adaptation) with External Archive.
    2. Local Search: Coordinate Descent (MTS-LS1 style) triggers before restarts to refine best solution.
    3. Restarts: Triggered by population stagnation or low variance to escape local optima.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Helper: Time Check ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: Coordinate Descent Polishing (MTS-LS1) ---
    def coordinate_descent(current_sol, current_val, search_ranges):
        """
        Refines a solution by tweaking one dimension at a time.
        Useful for draining the local basin before a restart.
        """
        nonlocal global_best_val, global_best_sol
        
        x_curr = current_sol.copy()
        f_curr = current_val
        improved_any = False
        
        # Iterate over dimensions in random order to avoid bias
        dims = np.random.permutation(dim)
        
        for d in dims:
            if check_time(): break
            
            sr = search_ranges[d]
            original_x = x_curr[d]
            
            # 1. Try moving in negative direction
            x_curr[d] = np.clip(original_x - sr, min_b[d], max_b[d])
            val = func(x_curr)
            
            if val < f_curr:
                f_curr = val
                improved_any = True
                # Update Global Best immediately
                if val < global_best_val:
                    global_best_val = val
                    global_best_sol = x_curr.copy()
            else:
                # 2. Try moving in positive direction (with 0.5 step modification)
                x_curr[d] = np.clip(original_x + 0.5 * sr, min_b[d], max_b[d])
                val = func(x_curr)
                
                if val < f_curr:
                    f_curr = val
                    improved_any = True
                    if val < global_best_val:
                        global_best_val = val
                        global_best_sol = x_curr.copy()
                else:
                    # 3. Revert if neither improved
                    x_curr[d] = original_x
                    
        return x_curr, f_curr, improved_any

    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        
        # 1. Initialization (New Restart)
        # Pop size: Balance between exploration (high) and speed (low)
        # 10*dim is standard for DE, capped at 100 to ensure generations run fast.
        pop_size = int(max(20, min(10 * dim, 100)))
        
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return global_best_val
            val = func(population[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = population[i].copy()
                
        # SHADE Memory (History)
        mem_size = 5
        mem_M_CR = np.full(mem_size, 0.5)
        mem_M_F = np.full(mem_size, 0.5)
        k_mem = 0
        
        # External Archive
        archive = np.zeros((int(pop_size * 2.0), dim))
        arc_count = 0
        max_arc = int(pop_size * 2.0)
        
        # Restart Triggers
        stag_count = 0
        prev_min_fit = np.min(fitness)
        
        # Initial search range for polishing (starts large)
        current_search_range = diff_b * 0.2
        
        # 2. Evolutionary Cycle
        while not check_time():
            # Sort population (needed for current-to-pbest)
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            best_in_gen = fitness[0]
            
            # --- Convergence / Stagnation Check ---
            if best_in_gen < prev_min_fit - 1e-12:
                prev_min_fit = best_in_gen
                stag_count = 0
            else:
                stag_count += 1
                
            pop_std = np.std(fitness)
            
            # RESTART TRIGGER: Low diversity OR Stagnation
            if pop_std < 1e-8 or stag_count > 35:
                # -> Polishing Phase before Restart
                # Use remaining time efficiently.
                rem_seconds = (time_limit - (datetime.now() - start_time)).total_seconds()
                
                # Only polish if we have a little time (e.g., > 0.5s)
                if rem_seconds > 0.5:
                    # Start polishing from the global best found so far
                    p_sol = global_best_sol.copy()
                    p_val = global_best_val
                    
                    # Determine range: Use population spread or fallback to tight range
                    pop_spread = np.max(population, axis=0) - np.min(population, axis=0)
                    ls_range = np.maximum(pop_spread, 1e-8)
                    
                    # Run a few passes of Coordinate Descent
                    for _ in range(5): # Max 5 passes
                        if check_time(): break
                        p_sol, p_val, improved = coordinate_descent(p_sol, p_val, ls_range)
                        
                        if not improved:
                            ls_range *= 0.4 # Shrink range if stuck
                        
                        # Stop if precision is high
                        if np.max(ls_range) < 1e-15: break
                
                break # Trigger outer loop restart

            # --- L-SHADE: Parameter Adaptation ---
            r_idx = np.random.randint(0, mem_size, pop_size)
            m_cr = mem_M_CR[r_idx]
            m_f = mem_M_F[r_idx]
            
            # Generate CR: Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F: Cauchy(M_F, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Resample invalid F
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                f[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
            f = np.clip(f, 0, 1)
            
            # --- Mutation: current-to-pbest/1 ---
            # p varies randomly [2/N, 0.2] to balance greediness
            p_min = 2.0 / pop_size
            p_val = np.random.uniform(p_min, 0.2)
            top_p = int(pop_size * p_val)
            
            pbest_idxs = np.random.randint(0, top_p, pop_size)
            x_pbest = population[pbest_idxs]
            
            # r1: Random from pop, r1 != i
            r1 = np.random.randint(0, pop_size, pop_size)
            # Fix collisions
            mask_self = (r1 == np.arange(pop_size))
            r1[mask_self] = (r1[mask_self] + 1) % pop_size
            x_r1 = population[r1]
            
            # r2: Random from Union(Pop, Archive)
            union_size = pop_size + arc_count
            r2 = np.random.randint(0, union_size, pop_size)
            x_r2 = np.zeros((pop_size, dim))
            
            mask_pop = r2 < pop_size
            x_r2[mask_pop] = population[r2[mask_pop]]
            
            mask_arc = ~mask_pop
            if np.any(mask_arc):
                x_r2[mask_arc] = archive[r2[mask_arc] - pop_size]
                
            # Compute Mutant
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover (Binomial) ---
            rand_cr = np.random.rand(pop_size, dim)
            mask_cross = rand_cr < cr[:, np.newaxis]
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask_cross, mutant, population)
            
            # --- Selection & Memory Update ---
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            succ_cr = []
            succ_f = []
            diffs = []
            
            for i in range(pop_size):
                if check_time(): return global_best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    # Improvement
                    diff = fitness[i] - f_trial
                    
                    # Archive Update
                    if arc_count < max_arc:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        ridx = np.random.randint(0, max_arc)
                        archive[ridx] = population[i].copy()
                        
                    # Store Success
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diffs.append(diff)
                    
                    # Update Pop
                    new_pop[i] = trial_pop[i]
                    new_fit[i] = f_trial
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_sol = trial_pop[i].copy()
            
            population = new_pop
            fitness = new_fit
            
            # --- Update SHADE Memories ---
            if len(diffs) > 0:
                diffs = np.array(diffs)
                succ_cr = np.array(succ_cr)
                succ_f = np.array(succ_f)
                
                total_diff = np.sum(diffs)
                if total_diff > 0:
                    weights = diffs / total_diff
                else:
                    weights = np.ones(len(diffs)) / len(diffs)
                    
                # M_CR Update (Weighted Mean)
                m_cr_new = np.sum(weights * succ_cr)
                mem_M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                
                # M_F Update (Weighted Lehmer Mean)
                num = np.sum(weights * (succ_f**2))
                den = np.sum(weights * succ_f)
                m_f_new = num / den if den > 0 else 0.5
                mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                
                k_mem = (k_mem + 1) % mem_size

    return global_best_val
