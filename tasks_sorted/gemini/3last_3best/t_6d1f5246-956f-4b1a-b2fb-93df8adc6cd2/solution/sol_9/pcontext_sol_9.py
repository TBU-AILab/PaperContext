#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 6.192530844357421
#2. output value is: 29.28926929139743
#3. output value is: 9.510001392407787
#
#The following Python code implements a **Restarting Memetic SHADE with Soft-Restarts and Accelerated Pattern Search**.
#
#**Key Improvements:**
#1.  **Soft-Restart Mechanism**: Unlike the previous best algorithm which performed a full random restart, this version **preserves the global best individual** when restarting. The rest of the population is re-initialized using Opposition-Based Learning (OBL). This ensures the algorithm retains the best basin of attraction found so far while forcefully introducing diversity to escape local optima.
#2.  **Accelerated Pattern Search**: The local search strategy (Coordinate Descent) is refined. It aggressively **doubles the step size** in successful directions to quickly traverse valleys ("Line Search" approximation). It is triggered when the population stagnates or converges, providing a high-precision polish.
#3.  **Adaptive Population & SHADE**: Implements the SHADE logic with success-history adaptation for crossover and mutation rates. The population size is set adaptively based on dimension but clipped to a safe range to ensure sufficient generations within the time limit.
#4.  **Robust Bound Handling**: Uses a "bounce-back" midpoint strategy (average of parent and bound) rather than simple clipping. This preserves the evolutionary momentum and prevents genes from sticking to the edges of the search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Memetic SHADE with Soft-Restarts 
    and Accelerated Pattern Search.
    
    Strategy:
    1. Evolution: SHADE (Success-History Adaptive Differential Evolution) for global exploration.
    2. Initialization: Opposition-Based Learning (OBL) to maximize initial coverage.
    3. Local Search: Accelerated Coordinate Descent for fast exploitation of valleys.
    4. Restart: Soft restart preserves the elite solution while resetting the population 
       to escape stagnation.
    """
    start_time = datetime.now()
    # Safety buffer to ensure we return before timeout
    time_limit = timedelta(seconds=max_time) - timedelta(seconds=0.05)

    # -------------------------------------------------------------------------
    # Helper: Check Time
    # -------------------------------------------------------------------------
    def check_time():
        return datetime.now() - start_time >= time_limit

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size
    # Adaptive: Enough to cover dimensions, but small enough to be fast.
    NP_init = int(np.clip(20 * dim, 40, 100))
    
    # Global Best Tracker
    global_best_val = float('inf')
    global_best_x = None

    # -------------------------------------------------------------------------
    # Local Search: Accelerated Pattern Search
    # -------------------------------------------------------------------------
    def accelerated_pattern_search(x_center, f_center):
        """
        Performs coordinate descent. If a step improves the solution, 
        the step size is doubled (accelerated) to move quickly down slopes.
        """
        x_curr = x_center.copy()
        f_curr = f_center
        
        # Initial step size: 5% of domain (aggressive start)
        step_sizes = diff_b * 0.05
        
        # Budget for local search
        max_ls_iter = 20
        
        for _ in range(max_ls_iter):
            if check_time(): break
            
            improved_in_pass = False
            # Randomize dimension order to avoid bias
            dims_order = np.random.permutation(dim)
            
            for i in dims_order:
                if check_time(): break
                
                orig_x = x_curr[i]
                step = step_sizes[i]
                
                # Directions to probe
                directions = [1, -1]
                
                for d in directions:
                    # 1. Probe
                    x_curr[i] = np.clip(orig_x + d * step, min_b[i], max_b[i])
                    f_new = func(x_curr)
                    
                    if f_new < f_curr:
                        f_curr = f_new
                        improved_in_pass = True
                        
                        # 2. Accelerate: Keep moving in this direction
                        # Double step size until improvement stops or bounds hit
                        while True:
                            if check_time(): break
                            
                            step *= 2.0
                            prev_x_val = x_curr[i]
                            
                            # Move further
                            x_curr[i] = np.clip(orig_x + d * step, min_b[i], max_b[i])
                            
                            # Check if hit bound
                            if np.abs(x_curr[i] - prev_x_val) < 1e-15:
                                break
                                
                            f_acc = func(x_curr)
                            if f_acc < f_curr:
                                f_curr = f_acc
                            else:
                                # Revert to last good point
                                x_curr[i] = prev_x_val
                                break
                        
                        # Move to next dimension after finding improvement
                        break 
                    else:
                        # Revert probe if no success
                        x_curr[i] = orig_x
            
            # If no improvement in a full pass over all dimensions, shrink steps
            if not improved_in_pass:
                step_sizes *= 0.5
                # Terminate if precision limit reached
                if np.max(step_sizes) < 1e-9:
                    break
                    
        return x_curr, f_curr

    # -------------------------------------------------------------------------
    # Main Restart Loop
    # -------------------------------------------------------------------------
    while not check_time():
        
        # --- Initialization ---
        NP = NP_init
        
        # 1. Base Population
        pop = min_b + np.random.rand(NP, dim) * diff_b
        
        # 2. Soft Restart Logic
        # If we have a global best, inject it into the new population
        # to preserve the best basin of attraction.
        if global_best_x is not None:
            pop[0] = global_best_x.copy()
            
        # 3. Opposition-Based Learning (OBL)
        # Generate opposite population: x_opp = min + max - x
        pop_opp = min_b + max_b - pop
        
        # Evaluate both sets
        fit = np.zeros(NP)
        fit_opp = np.zeros(NP)
        
        for i in range(NP):
            if check_time(): return global_best_val
            
            fit[i] = func(pop[i])
            if fit[i] < global_best_val:
                global_best_val = fit[i]
                global_best_x = pop[i].copy()
                
            fit_opp[i] = func(pop_opp[i])
            if fit_opp[i] < global_best_val:
                global_best_val = fit_opp[i]
                global_best_x = pop_opp[i].copy()
                
        # Select best NP from Union(Pop, Pop_Opp)
        mask_opp = fit_opp < fit
        pop = np.where(mask_opp[:, None], pop_opp, pop)
        fitness = np.where(mask_opp, fit_opp, fit)
        
        # --- SHADE Setup ---
        H = 6 # Memory size
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Stagnation tracking
        no_improv_count = 0
        last_best_run = np.min(fitness)
        
        # --- Evolutionary Loop ---
        while not check_time():
            
            # Sort population (for p-best selection)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Stagnation Detection
            curr_best = fitness[0]
            if curr_best < last_best_run:
                last_best_run = curr_best
                no_improv_count = 0
            else:
                no_improv_count += 1
            
            # Restart Trigger:
            # 1. Population converged (variance/spread is tiny)
            # 2. No improvement for significant generations (stagnation)
            spread = fitness[-1] - fitness[0]
            if spread < 1e-9 or no_improv_count > 30:
                # Polish the champion before restarting
                pol_x, pol_f = accelerated_pattern_search(pop[0], fitness[0])
                if pol_f < global_best_val:
                    global_best_val = pol_f
                    global_best_x = pol_x.copy()
                break # Break to outer loop (Restart)
            
            # Generate Parameters (SHADE)
            r_idx = np.random.randint(0, H, NP)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1)
            f_params = mu_f + 0.1 * np.random.standard_cauchy(NP)
            f_params = np.where(f_params > 1.0, 1.0, f_params)
            f_params = np.where(f_params <= 0.0, 0.5, f_params)
            
            # Mutation: current-to-pbest/1
            # Randomize p in [0.05, 0.2] for dynamic greediness
            p_val = np.random.uniform(0.05, 0.2)
            top_p = int(max(2, NP * p_val))
            
            pbest_idxs = np.random.randint(0, top_p, NP)
            x_pbest = pop[pbest_idxs]
            
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
            
            # Compute Mutant
            f_col = f_params[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            rand_u = np.random.rand(NP, dim)
            mask_cr = rand_u < cr[:, None]
            j_rand = np.random.randint(0, dim, NP)
            mask_cr[np.arange(NP), j_rand] = True
            
            trial = np.where(mask_cr, mutant, pop)
            
            # Bound Handling: Midpoint Back-projection
            # If x outside, set x = (x_old + bound) / 2
            min_mat = np.tile(min_b, (NP, 1))
            max_mat = np.tile(max_b, (NP, 1))
            
            low_viol = trial < min_mat
            if np.any(low_viol):
                trial[low_viol] = (pop[low_viol] + min_mat[low_viol]) * 0.5
                
            high_viol = trial > max_mat
            if np.any(high_viol):
                trial[high_viol] = (pop[high_viol] + max_mat[high_viol]) * 0.5
                
            trial = np.clip(trial, min_b, max_b)
            
            # Selection & Update
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
                    
                    # Update Pop
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    succ_diff.append(diff)
                    succ_f.append(f_params[i])
                    succ_cr.append(cr[i])
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_x = trial[i].copy()
                        no_improv_count = 0 # Reset stagnation
            
            # Maintain Archive Size
            while len(archive) > NP:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update SHADE Memory
            if len(succ_diff) > 0:
                diffs = np.array(succ_diff)
                total_diff = np.sum(diffs)
                
                if total_diff > 0:
                    weights = diffs / total_diff
                    sf = np.array(succ_f)
                    scr = np.array(succ_cr)
                    
                    # Weighted Lehmer Mean for F
                    mean_f = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-15)
                    # Weighted Mean for CR
                    mean_cr = np.sum(weights * scr)
                    
                    M_F[k_mem] = np.clip(mean_f, 0, 1)
                    M_CR[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H

    return global_best_val
