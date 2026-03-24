#The output value is: 11.381635825618886
#
#The algorithm code is:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting SHADE (Success-History Adaptive Differential Evolution).
    SHADE adapts control parameters (F and CR) using a historical memory of successful values
    and utilizes an external archive to maintain diversity.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Population size: SHADE typically uses ~18*D, but we clamp it to [30, 100]
    # to ensure we get enough generations within the time limit for expensive functions.
    NP = int(np.clip(10 * dim, 30, 100))
    
    # Memory size for SHADE history (H)
    H = 10 
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')

    # -------------------------------------------------------------------------
    # Main Restart Loop
    # -------------------------------------------------------------------------
    # The algorithm restarts if the population converges or stagnates, 
    # allowing exploration of different basins of attraction.
    while True:
        # Check time before starting a new restart
        if datetime.now() - start_time >= time_limit:
            return global_best_val
            
        # Initialize Population (Uniform Random)
        pop = min_b + np.random.rand(NP, dim) * diff_b
        fitness = np.full(NP, float('inf'))
        
        # Evaluate Initial Population
        for i in range(NP):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
        
        # Initialize SHADE Memory (History)
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive
        # Stores parent vectors that were successfully replaced to preserve diversity
        archive = []
        
        # ---------------------------------------------------------------------
        # Evolutionary Loop
        # ---------------------------------------------------------------------
        while True:
            # Time Check
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            # Sort population by fitness (needed for current-to-pbest selection)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Convergence Check: Restart if population fitness range is negligible
            if np.abs(fitness[-1] - fitness[0]) < 1e-8:
                break
            
            # 1. Parameter Generation
            # -----------------------
            # Select random index from memory for each individual
            r_idxs = np.random.randint(0, H, NP)
            mu_cr = mem_cr[r_idxs]
            mu_f = mem_f[r_idxs]
            
            # Generate CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            # numpy doesn't implement Cauchy(loc, scale), so we use standard_cauchy
            f = mu_f + 0.1 * np.random.standard_cauchy(NP)
            
            # Handle F bounds
            # If F > 1, cap at 1.0. If F <= 0, we treat it as 0.5 (simplification of retry strategy)
            f = np.where(f > 1, 1.0, f)
            f = np.where(f <= 0, 0.5, f)
            
            # 2. Mutation: DE/current-to-pbest/1
            # ----------------------------------
            # Equation: V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            
            # Select p-best vectors (p is random in [2/NP, 0.2])
            p_val = np.random.uniform(2/NP, 0.2)
            top_p_cnt = int(max(2, NP * p_val))
            
            pbest_idxs = np.random.randint(0, top_p_cnt, NP)
            x_pbest = pop[pbest_idxs]
            
            # Select r1 (random from pop)
            r1_idxs = np.random.randint(0, NP, NP)
            x_r1 = pop[r1_idxs]
            
            # Select r2 (random from Union(Population, Archive))
            if len(archive) > 0:
                archive_np = np.array(archive)
                pop_archive = np.vstack((pop, archive_np))
            else:
                pop_archive = pop
            
            size_pa = len(pop_archive)
            r2_idxs = np.random.randint(0, size_pa, NP)
            x_r2 = pop_archive[r2_idxs]
            
            # Compute Mutant Vectors
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            # -----------------------
            rand_vals = np.random.rand(NP, dim)
            mask = rand_vals < cr[:, None]
            
            # Guarantee at least one parameter changes per individual
            j_rand = np.random.randint(0, dim, NP)
            mask[np.arange(NP), j_rand] = True
            
            trial_pop = np.where(mask, mutant, pop)
            
            # 4. Bound Handling (Midpoint Correction)
            # ---------------------------------------
            # Instead of clipping, set violated params to (bound + parent) / 2
            # This preserves evolutionary direction better than hard clipping.
            
            min_b_broad = np.tile(min_b, (NP, 1))
            max_b_broad = np.tile(max_b, (NP, 1))
            
            lower_mask = trial_pop < min_b
            if np.any(lower_mask):
                trial_pop[lower_mask] = (min_b_broad[lower_mask] + pop[lower_mask]) * 0.5
                
            upper_mask = trial_pop > max_b
            if np.any(upper_mask):
                trial_pop[upper_mask] = (max_b_broad[upper_mask] + pop[upper_mask]) * 0.5
                
            # Final clip to ensure bounds are respected strictly
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 5. Selection and Updates
            # ------------------------
            new_fitness = np.zeros(NP)
            succ_mask = np.zeros(NP, dtype=bool)
            diff_f = np.zeros(NP)
            
            # Evaluate trial vectors
            for i in range(NP):
                if datetime.now() - start_time >= time_limit:
                    return global_best_val
                
                f_trial = func(trial_pop[i])
                new_fitness[i] = f_trial
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    succ_mask[i] = True
                    diff_f[i] = fitness[i] - f_trial
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
            
            # Update Population, Archive, and History
            success_indices = np.where(succ_mask)[0]
            
            if len(success_indices) > 0:
                # Update Archive: Add replaced parents
                for idx in success_indices:
                    archive.append(pop[idx].copy())
                
                # Maintain Archive Size <= NP (remove random if full)
                while len(archive) > NP:
                    rm_idx = np.random.randint(0, len(archive))
                    archive.pop(rm_idx)
                
                # Update Population
                pop[success_indices] = trial_pop[success_indices]
                fitness[success_indices] = new_fitness[success_indices]
                
                # Update SHADE Memory
                # We use Weighted Lehmer Mean for F and Weighted Arithmetic Mean for CR
                # weights are proportional to fitness improvement
                succ_diffs = diff_f[success_indices]
                total_diff = np.sum(succ_diffs)
                
                if total_diff > 0:
                    weights = succ_diffs / total_diff
                    
                    s_f = f[success_indices]
                    s_cr = cr[success_indices]
                    
                    # Mean wL (Lehmer) for F: sum(w*f^2)/sum(w*f)
                    mean_f_lehmer = np.sum(weights * (s_f ** 2)) / np.sum(weights * s_f)
                    
                    # Mean wA (Arithmetic) for CR: sum(w*cr)
                    mean_cr_wa = np.sum(weights * s_cr)
                    
                    # Update Memory at pointer k
                    mem_f[k_mem] = np.clip(mean_f_lehmer, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr_wa, 0, 1)
                    
                    k_mem = (k_mem + 1) % H
    
    return global_best_val
