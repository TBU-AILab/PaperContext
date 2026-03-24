#The following algorithm implements **L-SHADE-pAdapt** (Linear Population Size Reduction - Success-History Adaptive Differential Evolution with p-best Adaptation).
#
#**Key Improvements over previous best attempts:**
#1.  **L-SHADE with Optimized Population Sizing**: It combines the successful Linear Population Size Reduction (LPSR) strategy with a more robust initial population size derived from the best performing specific-SHADE implementation. This ensures the algorithm doesn't waste time on an excessively large initial population but still benefits from the "funneling" effect of LPSR (shrinking from ~50-80 down to 4 individuals) to maximize exploitation in the final moments.
#2.  **Adaptive Greedy Control (p-adaptation)**: The parameter `p` (controlling the greediness of the mutation strategy `current-to-pbest`) is not fixed (usually 0.11 or 0.05). Instead, it adapts linearly from `0.2` (high exploration) to `0.05` (high exploitation) over the course of the optimization. This prevents premature convergence in the early stages and accelerates convergence in the late stages.
#3.  **Fully Vectorized & Robust**: The implementation uses strict vectorization for parameter generation, mutation, and crossover to minimize Python interpreter overhead. It includes robust handling for boundary constraints and parameter repairs (Cauchy/Normal distribution constraints) without using slow loops.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-pAdapt (Linear Population Reduction SHADE with p-adaptation).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Initial Population Size: 
    # Scaled by dimension but capped to ensure responsiveness within max_time.
    # 12*dim proved more effective than 18*dim in restricted time scenarios.
    pop_size_init = int(12 * dim)
    pop_size_init = max(30, min(pop_size_init, 80))
    
    # Minimum Population Size (for LPSR)
    pop_size_min = 4
    
    # Current Population Size
    pop_size = pop_size_init
    
    # Archive size relative to population (A = pop_size * arc_rate)
    arc_rate = 2.6
    
    # Memory Size for Adaptive Parameters (H)
    H = 6
    
    # p-best Adaptation Parameters (Exploration -> Exploitation)
    p_max = 0.2
    p_min = 0.05
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize Memory (M_CR, M_F) to 0.5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer
    
    # External Archive (store inferior solutions)
    archive = []
    
    best_val = float('inf')
    
    # -------------------------------------------------------------------------
    # Initial Evaluation
    # -------------------------------------------------------------------------
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------
    while True:
        # Check overall time
        elapsed_obj = datetime.now() - start_time
        if elapsed_obj >= time_limit:
            return best_val
        
        # Calculate Progress (0.0 -> 1.0)
        progress = elapsed_obj.total_seconds() / max_time
        if progress > 1.0: progress = 1.0
        
        # ---------------------------------------------------------------------
        # 1. Linear Population Size Reduction (LPSR)
        # ---------------------------------------------------------------------
        target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            # Sort population by fitness to keep best individuals
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices[:target_size]]
            fitness = fitness[sort_indices[:target_size]]
            pop_size = target_size
            
            # Reduce Archive to maintain ratio
            curr_arc_cap = int(pop_size * arc_rate)
            if len(archive) > curr_arc_cap:
                # Randomly keep 'curr_arc_cap' elements
                keep_idxs = np.random.choice(len(archive), curr_arc_cap, replace=False)
                archive = [archive[k] for k in keep_idxs]
                
        # ---------------------------------------------------------------------
        # 2. Adaptation of p (for current-to-pbest mutation)
        # ---------------------------------------------------------------------
        p_curr = p_max + (p_min - p_max) * progress
        
        # ---------------------------------------------------------------------
        # 3. Parameter Generation (Vectorized)
        # ---------------------------------------------------------------------
        # Pick random memory slots
        r_idxs = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idxs]
        mu_f = mem_f[r_idxs]
        
        # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Constrain F
        f[f > 1] = 1.0
        
        # Regenerate F <= 0
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg):
                break
            n_neg = np.sum(mask_neg)
            f[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(n_neg)
            f[f > 1] = 1.0
            
        # ---------------------------------------------------------------------
        # 4. Mutation: current-to-pbest/1 (Vectorized)
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        # ---------------------------------------------------------------------
        # Identify p-best
        sorted_idx = np.argsort(fitness)
        n_pbest = max(2, int(p_curr * pop_size))
        pbest_pool = sorted_idx[:n_pbest]
        
        # Select pbest for each individual
        pbest_choices = np.random.choice(pbest_pool, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1 (distinct from i)
        # Shift random indices to ensure r1 != i
        shift_r1 = np.random.randint(1, pop_size, pop_size)
        r1_idxs = (np.arange(pop_size) + shift_r1) % pop_size
        x_r1 = pop[r1_idxs]
        
        # Select r2 (distinct from i, r1) from Union(Pop, Archive)
        if len(archive) > 0:
            arr_archive = np.array(archive)
            pop_all = np.vstack((pop, arr_archive))
        else:
            pop_all = pop
        
        n_all = len(pop_all)
        r2_idxs = np.random.randint(0, n_all, pop_size)
        
        # Fix collisions: r2 != i and r2 != r1
        curr_idxs = np.arange(pop_size)
        bad_r2 = (r2_idxs == curr_idxs) | (r2_idxs == r1_idxs)
        
        while np.any(bad_r2):
            n_bad = np.sum(bad_r2)
            r2_idxs[bad_r2] = np.random.randint(0, n_all, n_bad)
            bad_r2 = (r2_idxs == curr_idxs) | (r2_idxs == r1_idxs)
            
        x_r2 = pop_all[r2_idxs]
        
        # Compute Mutant Vectors
        f_v = f[:, None]
        mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
        
        # ---------------------------------------------------------------------
        # 5. Crossover (Binomial)
        # ---------------------------------------------------------------------
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        
        # Ensure at least one parameter comes from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trials = np.where(cross_mask, mutant, pop)
        
        # ---------------------------------------------------------------------
        # 6. Bound Constraints
        # ---------------------------------------------------------------------
        trials = np.clip(trials, min_b, max_b)
        
        # ---------------------------------------------------------------------
        # 7. Evaluation & Selection
        # ---------------------------------------------------------------------
        succ_f = []
        succ_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            f_tri = func(trials[i])
            
            # Greedy Selection
            if f_tri <= fitness[i]:
                # If strictly better, record for memory update
                if f_tri < fitness[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fitness.append(fitness[i] - f_tri)
                    # Add parent to archive
                    archive.append(pop[i].copy())
                
                pop[i] = trials[i]
                fitness[i] = f_tri
                
                if f_tri < best_val:
                    best_val = f_tri
        
        # ---------------------------------------------------------------------
        # 8. Archive Maintenance & Memory Update
        # ---------------------------------------------------------------------
        # Enforce archive size limit
        curr_arc_cap = int(pop_size * arc_rate)
        while len(archive) > curr_arc_cap:
            # Remove random element
            rm_idx = np.random.randint(0, len(archive))
            archive.pop(rm_idx)
            
        # Update Memory if successful individuals found
        if len(succ_f) > 0:
            arr_sf = np.array(succ_f)
            arr_scr = np.array(succ_cr)
            arr_diff = np.array(diff_fitness)
            
            # Weights based on fitness improvement
            total_diff = np.sum(arr_diff)
            if total_diff > 0:
                weights = arr_diff / total_diff
                
                # Weighted Lehmer Mean for F
                denom = np.sum(weights * arr_sf)
                if denom == 0:
                    mean_f = 0.5
                else:
                    mean_f = np.sum(weights * (arr_sf**2)) / denom
                
                # Weighted Mean for CR
                mean_cr = np.sum(weights * arr_scr)
                
                # Update memory
                mem_f[k_mem] = mean_f
                mem_cr[k_mem] = mean_cr
                k_mem = (k_mem + 1) % H

    return best_val
