#The following Python code implements a **Restarting Memetic SHADE (Success-History Adaptive Differential Evolution) with Opposition-Based Initialization**. 
#
#This algorithm is designed to robustly find the minimum value within a limited time frame by combining several advanced strategies:
#1.  **Opposition-Based Initialization**: Generates an initial population and its opposite within the bounds, selecting the best individuals to jump-start the search.
#2.  **SHADE Adaptation**: Uses a historical memory of success parameters (Memory $H=6$) to adapt crossover ($CR$) and mutation ($F$) rates for the specific landscape.
#3.  **External Archive**: Preserves diversity by storing high-quality solutions that were recently replaced, preventing premature convergence.
#4.  **Local Search Polish**: When the population converges, a Coordinate Descent (Pattern Search) phase refines the best solution to high precision.
#5.  **Restart Mechanism**: Automatically restarts the population if convergence or stagnation is detected, ensuring the available time is fully utilized to explore different basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Memetic SHADE with Opposition-Based Initialization.
    
    Key Components:
    - OBL (Opposition-Based Learning) for initialization.
    - SHADE (Success-History Adaptive DE) for evolutionary search.
    - Coordinate Descent for local polishing.
    - Time-aware restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Configuration & Pre-processing
    # -------------------------------------------------------------------------
    # Bounds processing for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size
    # Adaptive based on dimension. We clamp to [40, 100] to balance 
    # exploration capability with generation speed within time limits.
    NP = int(np.clip(20 * dim, 40, 100))
    
    # SHADE Memory Size
    H = 6
    
    # Global Best Tracker
    global_best_val = float('inf')
    
    # -------------------------------------------------------------------------
    # Helper: Time Check
    # -------------------------------------------------------------------------
    def check_time():
        return datetime.now() - start_time >= time_limit

    # -------------------------------------------------------------------------
    # Helper: Local Search (Coordinate Descent)
    # -------------------------------------------------------------------------
    def local_search(x_input, f_input):
        """
        Refines a solution using coordinate descent (pattern search)
        with shrinking step sizes.
        """
        x_curr = x_input.copy()
        f_curr = f_input
        
        # Initial step size: 1% of the domain
        step_sizes = diff_b * 0.01
        
        # Limit iterations to avoid spending too much time in LS
        max_ls_iter = 20
        
        for _ in range(max_ls_iter):
            if check_time(): break
            
            # Stop if step size is too small (precision limit)
            if np.max(step_sizes) < 1e-9:
                break
                
            improved = False
            # Random shuffle of dimensions to avoid bias
            dims_perm = np.random.permutation(dim)
            
            for j in dims_perm:
                if check_time(): break
                
                original_val = x_curr[j]
                step = step_sizes[j]
                
                # 1. Try positive step
                x_curr[j] = np.clip(original_val + step, min_b[j], max_b[j])
                val = func(x_curr)
                if val < f_curr:
                    f_curr = val
                    improved = True
                    continue
                
                # 2. Try negative step
                x_curr[j] = np.clip(original_val - step, min_b[j], max_b[j])
                val = func(x_curr)
                if val < f_curr:
                    f_curr = val
                    improved = True
                    continue
                
                # 3. No improvement: Revert
                x_curr[j] = original_val
            
            if not improved:
                # Shrink step size
                step_sizes *= 0.5
                
        return x_curr, f_curr

    # -------------------------------------------------------------------------
    # Main Restart Loop
    # -------------------------------------------------------------------------
    while not check_time():
        
        # --- 1. Initialization with OBL ---
        # Generate random population
        pop = min_b + np.random.rand(NP, dim) * diff_b
        
        # Generate opposite population
        pop_opp = min_b + max_b - pop
        
        # Evaluate both to select the best starting set
        fit_pop = np.zeros(NP)
        fit_opp = np.zeros(NP)
        
        # Evaluate Random Pop
        for i in range(NP):
            if check_time(): return global_best_val
            val = func(pop[i])
            fit_pop[i] = val
            if val < global_best_val: global_best_val = val
            
        # Evaluate Opposite Pop
        for i in range(NP):
            if check_time(): return global_best_val
            val = func(pop_opp[i])
            fit_opp[i] = val
            if val < global_best_val: global_best_val = val
            
        # Select best N individuals from Union(Pop, Pop_Opp)
        mask_opp = fit_opp < fit_pop
        pop = np.where(mask_opp[:, None], pop_opp, pop)
        fitness = np.where(mask_opp, fit_opp, fit_pop)
        
        # --- 2. SHADE Setup ---
        # Memory for F and CR
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Archive
        archive = []
        
        # --- 3. Evolutionary Loop ---
        while not check_time():
            
            # Sort population (required for current-to-pbest)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Check Convergence
            # If fitness spread is negligible, we are converged.
            if np.abs(fitness[-1] - fitness[0]) < 1e-8:
                # Perform Local Search on best, then restart
                ls_vec, ls_val = local_search(pop[0], fitness[0])
                if ls_val < global_best_val:
                    global_best_val = ls_val
                break 
            
            # Generate Parameters using Memory
            r_idxs = np.random.randint(0, H, NP)
            mu_cr = mem_cr[r_idxs]
            mu_f = mem_f[r_idxs]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.random.standard_cauchy(NP)
            # Handle F bounds
            f = np.where(f > 1.0, 1.0, f)
            f = np.where(f <= 0.0, 0.5, f) # Conservative reset if negative
            
            # Mutation: DE/current-to-pbest/1
            # p is random in [2/NP, 0.2]
            p_val = np.random.uniform(2/NP, 0.2)
            top_p = int(max(2, NP * p_val))
            
            pbest_idxs = np.random.randint(0, top_p, NP)
            x_pbest = pop[pbest_idxs]
            
            r1_idxs = np.random.randint(0, NP, NP)
            x_r1 = pop[r1_idxs]
            
            # x_r2 from Union(Pop, Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((pop, archive_np))
            else:
                union_pop = pop
                
            r2_idxs = np.random.randint(0, len(union_pop), NP)
            x_r2 = union_pop[r2_idxs]
            
            # Compute Mutant
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            rand_vals = np.random.rand(NP, dim)
            mask = rand_vals < cr[:, None]
            # Ensure at least one parameter is mutated
            j_rand = np.random.randint(0, dim, NP)
            mask[np.arange(NP), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # Bound Handling: Midpoint Back-projection
            # If out of bounds, set value to (bound + parent) / 2
            # Vectorized implementation
            low_viol = trial < min_b
            if np.any(low_viol):
                # Broadcasting (min_b + pop) works correctly
                trial[low_viol] = (min_b + pop)[low_viol] * 0.5
                
            high_viol = trial > max_b
            if np.any(high_viol):
                trial[high_viol] = (max_b + pop)[high_viol] * 0.5
                
            trial = np.clip(trial, min_b, max_b)
            
            # Selection and Updates
            succ_diff = []
            succ_f = []
            succ_cr = []
            
            for i in range(NP):
                if check_time(): return global_best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    # Successful Update
                    diff = fitness[i] - f_trial
                    
                    # Add old parent to archive
                    archive.append(pop[i].copy())
                    
                    # Update Pop
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    # Store success params
                    succ_diff.append(diff)
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        
            # Maintain Archive Size <= NP
            while len(archive) > NP:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update SHADE History Memory (Weighted Lehmer Mean)
            if len(succ_diff) > 0:
                diff_arr = np.array(succ_diff)
                total_diff = np.sum(diff_arr)
                
                if total_diff > 0:
                    weights = diff_arr / total_diff
                    sf = np.array(succ_f)
                    scr = np.array(succ_cr)
                    
                    mean_f = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-15)
                    mean_cr = np.sum(weights * scr)
                    
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H
                    
    return global_best_val
