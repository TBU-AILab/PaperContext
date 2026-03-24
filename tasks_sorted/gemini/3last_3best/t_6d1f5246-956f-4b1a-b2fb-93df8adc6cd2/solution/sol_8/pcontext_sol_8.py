#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 29.28926929139743
#2. output value is: 9.510001392407787
#3. output value is: 41.458591949705784
#
#The following Python code implements a **Restarting Memetic SHADE with Opposition-Based Initialization and Accelerated Local Search**.
#
#**Key Improvements:**
#1.  **Accelerated Local Search**: Replaces the simple coordinate descent with an accelerated pattern search. If a step in a certain direction yields improvement, the algorithm aggressively expands the step size (doubling it) to traverse valleys or slopes quickly. This significantly improves polishing performance on correlated landscapes compared to fixed-step methods.
#2.  **Opposition-Based Initialization (OBL)**: Generates initial solutions and their opposites (symmetric with respect to bounds), evaluating both to start the evolution in the most promising basins of attraction. This was a key factor in the best performing algorithm previously.
#3.  **SHADE Adaptation**: Utilizes success-history adaptation for crossover ($CR$) and mutation ($F$) rates, ensuring the algorithm tunes itself to the specific function topology.
#4.  **Restart Mechanism**: Monitors convergence and stagnation. If the population converges (low variance) or stagnates (no global improvement), it triggers the local search to refine the best solution and then restarts the population to explore new areas, maximizing time usage.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Memetic SHADE with OBL and Accelerated Local Search.
    
    Components:
    - OBL: Opposition-Based Learning for robust initialization.
    - SHADE: Adaptive DE for global search.
    - Accelerated Local Search: Pattern search with step expansion for fast polishing.
    - Restart: Time-aware restarts to escape local optima.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Configuration & Pre-processing
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size: Adaptive based on dimension.
    # Clipped to ensure performance on limited time budget.
    NP = int(np.clip(15 * dim, 30, 80))
    
    # Global Best Tracker
    global_best_val = float('inf')
    
    # -------------------------------------------------------------------------
    # Helper: Check Time
    # -------------------------------------------------------------------------
    def check_time():
        return datetime.now() - start_time >= time_limit

    # -------------------------------------------------------------------------
    # Local Search: Accelerated Pattern Search
    # -------------------------------------------------------------------------
    def accelerated_local_search(x_start, f_start):
        """
        Performs coordinate descent with step size acceleration.
        If a direction is successful, the step size increases to move faster.
        """
        x = x_start.copy()
        f = f_start
        
        # Initial step sizes (2% of domain)
        step_sizes = diff_b * 0.02
        
        # Iteration limit to conserve time for evolutionary phase
        max_ls_iter = 15
        
        for _ in range(max_ls_iter):
            if check_time(): break
            
            improved_in_pass = False
            
            # Random dimension order
            dims = np.random.permutation(dim)
            
            for i in dims:
                if check_time(): break
                
                original_xi = x[i]
                current_step = step_sizes[i]
                
                # 1. Try Positive Direction
                x[i] = np.clip(original_xi + current_step, min_b[i], max_b[i])
                f_new = func(x)
                
                if f_new < f:
                    f = f_new
                    improved_in_pass = True
                    # Accelerate: Keep moving in this direction with growing steps
                    while True:
                        if check_time(): break
                        current_step *= 2.0
                        prev_xi = x[i]
                        x[i] = np.clip(original_xi + current_step, min_b[i], max_b[i])
                        
                        # Stop if hit bound
                        if np.abs(x[i] - prev_xi) < 1e-12: break
                        
                        f_acc = func(x)
                        if f_acc < f:
                            f = f_acc
                        else:
                            # Revert to last good point
                            x[i] = prev_xi
                            break
                    # Move to next dimension after exploiting this one
                    continue 
                else:
                    # Revert positive probe
                    x[i] = original_xi
                
                # 2. Try Negative Direction
                current_step = step_sizes[i] # Reset step
                x[i] = np.clip(original_xi - current_step, min_b[i], max_b[i])
                f_new = func(x)
                
                if f_new < f:
                    f = f_new
                    improved_in_pass = True
                    # Accelerate
                    while True:
                        if check_time(): break
                        current_step *= 2.0
                        prev_xi = x[i]
                        x[i] = np.clip(original_xi - current_step, min_b[i], max_b[i])
                        
                        if np.abs(x[i] - prev_xi) < 1e-12: break
                        
                        f_acc = func(x)
                        if f_acc < f:
                            f = f_acc
                        else:
                            x[i] = prev_xi
                            break
                else:
                    # Revert negative probe
                    x[i] = original_xi
            
            # If no improvement in full pass, reduce step size
            if not improved_in_pass:
                step_sizes *= 0.5
                if np.max(step_sizes) < 1e-9:
                    break
                    
        return x, f

    # -------------------------------------------------------------------------
    # Main Restart Loop
    # -------------------------------------------------------------------------
    while not check_time():
        
        # --- 1. Initialization with OBL ---
        pop = min_b + np.random.rand(NP, dim) * diff_b
        pop_opp = min_b + max_b - pop # Opposite population
        
        fit = np.zeros(NP)
        fit_opp = np.zeros(NP)
        
        # Evaluate Random Pop
        for i in range(NP):
            if check_time(): return global_best_val
            fit[i] = func(pop[i])
            if fit[i] < global_best_val: global_best_val = fit[i]
            
        # Evaluate Opposite Pop
        for i in range(NP):
            if check_time(): return global_best_val
            fit_opp[i] = func(pop_opp[i])
            if fit_opp[i] < global_best_val: global_best_val = fit_opp[i]
            
        # Select best NP individuals from Union(Pop, Pop_Opp)
        mask = fit_opp < fit
        pop = np.where(mask[:, None], pop_opp, pop)
        fitness = np.where(mask, fit_opp, fit)
        
        # --- 2. SHADE Configuration ---
        H = 6 # Memory size
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Stagnation monitoring
        no_improv_count = 0
        last_best_val = np.min(fitness)
        
        # --- 3. Evolutionary Loop ---
        while not check_time():
            
            # Sort population (for pbest selection)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            curr_best = fitness[0]
            
            # Stagnation Check
            if np.abs(curr_best - last_best_val) < 1e-10:
                no_improv_count += 1
            else:
                no_improv_count = 0
                last_best_val = curr_best
            
            # Convergence/Stagnation Trigger -> Polish & Restart
            # If fitness spread is tiny OR stagnation for >25 gens
            spread = fitness[-1] - fitness[0]
            if spread < 1e-8 or no_improv_count > 25:
                # Run Local Search on the best solution
                ls_x, ls_f = accelerated_local_search(pop[0], fitness[0])
                if ls_f < global_best_val:
                    global_best_val = ls_f
                break # Break inner loop to trigger restart
            
            # --- Parameter Generation ---
            r_idx = np.random.randint(0, H, NP)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.random.standard_cauchy(NP)
            f = np.where(f > 1.0, 1.0, f)
            f = np.where(f <= 0.0, 0.5, f) # Conservative reset
            
            # --- Mutation: current-to-pbest/1 ---
            # Random p in [2/NP, 0.2]
            p_vals = np.random.uniform(2/NP, 0.2, NP)
            top_indices = np.maximum((NP * p_vals).astype(int), 1)
            
            # Efficiently select pbest indices
            pbest_idxs = np.array([np.random.randint(0, t) for t in top_indices])
            x_pbest = pop[pbest_idxs]
            
            # r1
            r1_idxs = np.random.randint(0, NP, NP)
            x_r1 = pop[r1_idxs]
            
            # r2 from Union(Pop, Archive)
            if len(archive) > 0:
                arc_arr = np.array(archive)
                pop_all = np.vstack((pop, arc_arr))
            else:
                pop_all = pop
                
            r2_idxs = np.random.randint(0, len(pop_all), NP)
            x_r2 = pop_all[r2_idxs]
            
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_u = np.random.rand(NP, dim)
            cross_mask = rand_u < cr[:, None]
            j_rand = np.random.randint(0, dim, NP)
            cross_mask[np.arange(NP), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # --- Bound Handling (Bounce Back / Midpoint) ---
            # Helps preserve evolutionary momentum near bounds
            low_viol = trial < min_b
            if np.any(low_viol):
                trial[low_viol] = (min_b + pop)[low_viol] * 0.5
            
            high_viol = trial > max_b
            if np.any(high_viol):
                trial[high_viol] = (max_b + pop)[high_viol] * 0.5
                
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            succ_diff = []
            succ_f = []
            succ_cr = []
            
            for i in range(NP):
                if check_time(): return global_best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    diff = fitness[i] - f_trial
                    
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    
                    # Update population
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    succ_diff.append(diff)
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        no_improv_count = 0 # Reset stagnation
            
            # Manage Archive Size
            while len(archive) > NP:
                archive.pop(np.random.randint(0, len(archive)))
                
            # --- Update Memory ---
            if len(succ_diff) > 0:
                diffs = np.array(succ_diff)
                total_diff = np.sum(diffs)
                
                if total_diff > 0:
                    w = diffs / total_diff
                    sf = np.array(succ_f)
                    scr = np.array(succ_cr)
                    
                    # Weighted Lehmer Mean for F
                    mean_f = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-15)
                    # Weighted Arithmetic Mean for CR
                    mean_cr = np.sum(w * scr)
                    
                    M_F[k_mem] = np.clip(mean_f, 0, 1)
                    M_CR[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H

    return global_best_val
