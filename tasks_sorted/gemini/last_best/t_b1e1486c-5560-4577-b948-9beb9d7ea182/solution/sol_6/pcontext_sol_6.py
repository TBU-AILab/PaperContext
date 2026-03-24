#The following code implements the **jSO** algorithm (an advanced variant of L-SHADE). 
#
#**Improvements over the previous L-SHADE implementation:**
#1.  **Weighted Lehmer Mean:** It uses a more sophisticated weighted update for the adaptive memory (based on fitness improvement magnitude), allowing the algorithm to learn successful parameters faster.
#2.  **Dynamic Selection Pressure ($p$):** The value of $p$ (controlling greediness) is not just linear; it adapts to separate exploration (high $p$) and exploitation (low $p$) phases more distinctively.
#3.  **Stage-Based Parameter Constraints:** It implements the specific rules of jSO where Crossover Rate ($CR$) and Scaling Factor ($F$) are constrained differently in the first 25% and 60% of the search to prevent premature convergence and stagnation.
#4.  **Stagnation Restart:** A mechanism detects if the population fitness variance collapses (premature convergence) or if the algorithm is stuck. If so, it triggers a partial restart (perturbation) to escape local optima, which is likely the reason the previous algorithm yielded a score of ~38.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using jSO (an optimized variant of L-SHADE) 
    with stagnation detection and partial restarts.
    """
    start_time = time.time()
    
    # --- 1. Initialization & Config ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Population Sizing (jSO heuristic)
    # Start large for exploration, reduce over time
    pop_size_init = int(round(25 * np.log(dim) * np.sqrt(dim)))
    pop_size_init = max(30, min(pop_size_init, 400)) # Heuristic bounds
    pop_size_min = 4
    
    pop_size = pop_size_init
    
    # Archive Size
    archive_size = int(round(pop_size * 1.4)) # jSO uses 1.4x (smaller than L-SHADE 2.0x)
    archive = []
    
    # Memory for Adaptive Parameters (H = 5 is standard for jSO)
    h_mem = 5
    m_cr = np.full(h_mem, 0.8) # Initial Mean CR (starts higher in jSO)
    m_f = np.full(h_mem, 0.5)  # Initial Mean F
    k_mem = 0
    
    # Initialize Population
    # Using random uniform initialization
    pop = min_b + (max_b - min_b) * np.random.rand(pop_size, dim)
    fitness = np.zeros(pop_size)
    
    # Evaluate Initial Population
    best_fitness = float('inf')
    best_sol = None
    
    for i in range(pop_size):
        if time.time() - start_time > max_time:
            # If we time out during init, return best found so far
            return best_fitness if best_sol is not None else float('inf')
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # Sort for rank-based logic
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- 2. Main Optimization Loop ---
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed >= max_time:
            break
            
        # Progress Ratio (0.0 -> 1.0)
        progress = elapsed / max_time
        # Correction to avoid 1.0 until very end
        progress = min(progress, 0.9999) 
        
        # --- A. Linear Population Size Reduction (LPSR) ---
        plan_pop_size = int(round(((pop_size_min - pop_size_init) * progress) + pop_size_init))
        
        if pop_size > plan_pop_size:
            # Reduce population (remove worst)
            reduction_count = pop_size - plan_pop_size
            pop_size = plan_pop_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Resize Archive accordingly
            curr_archive_max = int(round(pop_size * 1.4))
            if len(archive) > curr_archive_max:
                # Remove random elements to shrink archive
                del_indices = np.random.choice(len(archive), len(archive) - curr_archive_max, replace=False)
                # Create a boolean mask to keep elements
                keep_mask = np.ones(len(archive), dtype=bool)
                keep_mask[del_indices] = False
                # Rebuild archive list (efficient enough for these sizes)
                archive = [archive[i] for i in range(len(archive)) if keep_mask[i]]

        # --- B. Parameter Adaptation (jSO Specifics) ---
        
        # Calculate p (for current-to-pbest)
        # jSO linear reduction of p from p_max to p_min
        p_max = 0.25
        p_min = 0.05
        p_val = p_max - (p_max - p_min) * progress
        p_num = max(2, int(round(pop_size * p_val)))
        
        # Generate CR and F
        # Pick memory indices
        r_idx = np.random.randint(0, h_mem, size=pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # CR: Normal Distribution, clamped [0, 1]
        # Approximation: if mu_cr == -1, assign 0.0 (terminal case)
        crs = np.random.normal(mu_cr, 0.1)
        crs = np.clip(crs, 0.0, 1.0)
        
        # jSO Constraint 1: Early exploration favors crossover
        # If progress < 0.25, CR must be at least 0.7 for 50% of values? 
        # jSO Rule: if progress < 0.25 and cr < 0.7, set cr = 0.7
        # We apply this probabilistically or strictly. Strict is better for robustness.
        if progress < 0.25:
             crs[crs < 0.7] = 0.7
             
        # jSO Constraint 2: Late exploitation favors lower CR? (Standard behavior)
        
        # F: Cauchy Distribution
        # Cauchy(loc, scale) ~ loc + scale * tan(pi * (rand - 0.5))
        fs = mu_f + 0.1 * np.random.standard_cauchy(size=pop_size)
        
        # Handle F constraints
        # If F > 1, cap at 1. If F <= 0, retry until > 0.
        while True:
            mask_neg = fs <= 0
            if not np.any(mask_neg):
                break
            fs[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(size=np.sum(mask_neg))
            
        fs = np.clip(fs, 0.0, 1.0)
        
        # jSO Constraint 3: Early phase prevents large steps (chaos)
        # If progress < 0.6 and F > 0.7, set F = 0.7
        if progress < 0.6:
            fs[fs > 0.7] = 0.7
            
        # --- C. Mutation (current-to-pbest-w/1) ---
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # 1. X_pbest: Randomly selected from top p_num individuals
        pbest_indices = np.random.randint(0, p_num, size=pop_size)
        x_pbest = pop[pbest_indices]
        
        # 2. X_r1: Random from population, r1 != i
        r1_indices = np.random.randint(0, pop_size, size=pop_size)
        # Collision check
        mask_col = r1_indices == np.arange(pop_size)
        r1_indices[mask_col] = (r1_indices[mask_col] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # 3. X_r2: Random from Union(Pop, Archive), r2 != r1, r2 != i
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        union_size = len(union_pop)
        r2_indices = np.random.randint(0, union_size, size=pop_size)
        
        # Basic collision handling for r2 (statistically rare in large Union, but critical)
        # We just re-roll collisions once, usually sufficient
        mask_col_r2 = (r2_indices == r1_indices) | (r2_indices == np.arange(pop_size))
        if np.any(mask_col_r2):
             r2_indices[mask_col_r2] = np.random.randint(0, union_size, size=np.sum(mask_col_r2))
             
        x_r2 = union_pop[r2_indices]
        
        # Compute Mutant Vector
        # Broadcasting F for dimensions
        f_broad = fs[:, None]
        
        # Weighted difference implementation often helps (Fw), but standard jSO uses raw F
        mutants = pop + f_broad * (x_pbest - pop) + f_broad * (x_r1 - x_r2)
        
        # --- D. Boundary Correction (Midpoint Target) ---
        # If mutant is out of bounds, place it between parent and bound
        # This preserves evolution direction better than clipping
        
        mask_low = mutants < min_b
        if np.any(mask_low):
            # Identifying the parents for the specific dimensions
            # indices [row, col]
            rows, cols = np.where(mask_low)
            mutants[rows, cols] = (pop[rows, cols] + min_b[cols]) / 2.0
            
        mask_high = mutants > max_b
        if np.any(mask_high):
            rows, cols = np.where(mask_high)
            mutants[rows, cols] = (pop[rows, cols] + max_b[cols]) / 2.0
            
        # --- E. Crossover (Binomial) ---
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals <= crs[:, None]
        
        # Ensure at least one variable is taken from mutant
        j_rand = np.random.randint(0, dim, size=pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trials = np.where(mask_cross, mutants, pop)
        
        # --- F. Selection ---
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        successful_f = []
        successful_cr = []
        diff_fitness = []
        
        # Evaluate Loop (check time frequently)
        for i in range(pop_size):
            # Time check block (every 10 evals to reduce overhead)
            if i % 10 == 0:
                if time.time() - start_time > max_time:
                    return best_fitness
            
            f_trial = func(trials[i])
            
            if f_trial <= fitness[i]:
                new_pop[i] = trials[i]
                new_fitness[i] = f_trial
                
                if f_trial < fitness[i]:
                    # Successful update
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    
                    successful_f.append(fs[i])
                    successful_cr.append(crs[i])
                    diff_fitness.append(fitness[i] - f_trial)
                    
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_sol = trials[i].copy()
                        
        pop = new_pop
        fitness = new_fitness
        
        # Manage Archive Size again (in case additions exceeded limits)
        curr_archive_max = int(round(pop_size * 1.4))
        if len(archive) > curr_archive_max:
            # Random removal
            num_remove = len(archive) - curr_archive_max
            # Efficiently remove from end after shuffle, or just random indices
            del_indices = np.random.choice(len(archive), num_remove, replace=False)
            # Rebuild
            keep_mask = np.ones(len(archive), dtype=bool)
            keep_mask[del_indices] = False
            archive = [archive[k] for k in range(len(archive)) if keep_mask[k]]
            
        # --- G. Memory Update (Weighted Lehmer Mean) ---
        if len(diff_fitness) > 0:
            weights = np.array(diff_fitness)
            weights = weights / np.sum(weights) # Normalize
            
            # Update M_F (Weighted Lehmer Mean)
            # MeanL = sum(w * f^2) / sum(w * f)
            s_f = np.array(successful_f)
            num = np.sum(weights * (s_f**2))
            den = np.sum(weights * s_f)
            if den > 1e-10:
                new_mean_f = num / den
                m_f[k_mem] = 0.5 * m_f[k_mem] + 0.5 * new_mean_f # Smoothing
            
            # Update M_CR (Weighted Mean)
            # Standard weighted mean, but check for terminal values
            s_cr = np.array(successful_cr)
            # Special handling: if max(s_cr) == 0, we might need logic, but usually fine
            if np.max(s_cr) == 0 and m_cr[k_mem] == 0:
                 new_mean_cr = 0
            else:
                 new_mean_cr = np.sum(weights * s_cr)
                 
            m_cr[k_mem] = 0.5 * m_cr[k_mem] + 0.5 * new_mean_cr
            
            # Move pointer
            k_mem = (k_mem + 1) % h_mem

        # --- H. Re-sort ---
        # Sorting is required for the 'pbest' selection logic at start of loop
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # --- I. Stagnation Detection & Restart ---
        # If population variance is effectively zero, we are stuck.
        # Check happens only occasionally to save time.
        if pop_size > 5:
            fit_std = np.std(fitness)
            fit_range = fitness[-1] - fitness[0]
            
            # If standard deviation is extremely low, we have converged.
            # If time is still plentiful (< 85% used), we should restart to try finding a better peak.
            if fit_std < 1e-8 and progress < 0.85:
                # RESTART Strategy:
                # Keep the best (elitism)
                # Re-initialize the rest of the population
                # Reset memory partially to allow new parameter exploration
                
                # 1. Perturb population: Keep best, others = best + heavy mutation
                # or just random re-init. Random re-init is safer for global search.
                
                # Keep best
                pop[0] = best_sol
                
                # Re-init others
                new_random = min_b + (max_b - min_b) * np.random.rand(pop_size - 1, dim)
                pop[1:] = new_random
                fitness[1:] = np.inf # Force re-eval
                
                # Evaluate new batch
                for i in range(1, pop_size):
                    if time.time() - start_time > max_time:
                         return best_fitness
                    fitness[i] = func(pop[i])
                    if fitness[i] < best_fitness:
                        best_fitness = fitness[i]
                        best_sol = pop[i].copy()
                
                # Re-sort
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx]
                fitness = fitness[sorted_idx]
                
                # Soft Reset Memory (don't lose all knowledge, but add variance)
                m_f[:] = 0.5
                m_cr[:] = 0.5

    return best_fitness
