#The previous analysis indicates that the **SHADE-based algorithm (Algorithm 1)** performed significantly better (Score: 1.14) than the jDE variants (Scores: 32.32, 37.86). This suggests that the **Success-History Adaptation** and **Archive** mechanisms are highly effective for this problem class.
#
#However, to reach the global minimum (score near 0.0), we can improve upon the SHADE implementation by:
#1.  **Refining the Restart Strategy with Local Search**: The previous restart was a "hard" reset. Before discarding the converged population, we should exploit the best solution found so far using a deterministic **Coordinate Descent (Pattern Search)**. This helps to "polish" the solution, draining the local basin of attraction to its absolute minimum.
#2.  **Optimizing Population Size**: A population of `15 * dim` (used in Code 1) can be computationally expensive for high dimensions. We will adjust this to `12 * dim` to allow for more generations within the time limit while maintaining diversity.
#3.  **Tighter Convergence Criteria**: We switch from checking standard deviation to checking the value range (`max - min` fitness), which is a more direct measure of convergence for restart purposes.
#
#Here is the improved algorithm (**SHADE with Coordinate Descent Restart**).
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    SHADE (Success-History based Adaptive Differential Evolution) with 
    Archive, Local Search Polish, and Restart Strategy.
    """
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Linear with dimension, but capped to avoid extreme slowness
    # 12 * dim is a balanced choice between diversity and speed
    pop_size = int(max(30, 12 * dim))
    
    # Archive size matches population size
    archive_size = pop_size
    
    # History Memory size
    H = 6
    
    # Pre-process bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Memory for Adaptive Parameters (M_F, M_CR)
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive
    archive = np.zeros((archive_size, dim))
    arc_count = 0
    
    # Track Global Best
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Time Check
        if datetime.now() - start_time >= time_limit:
            return best_val
            
        # 1. Parameter Generation
        # -----------------------
        r_idx = np.random.randint(0, H, pop_size)
        mu_f = mem_f[r_idx]
        mu_cr = mem_cr[r_idx]
        
        # Generate F (Cauchy)
        # If F <= 0, regenerate. If F > 1, clip to 1.
        F = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        while True:
            mask_neg = F <= 0
            if not np.any(mask_neg):
                break
            F[mask_neg] = mu_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
        F = np.minimum(F, 1.0)
        
        # Generate CR (Normal)
        CR = np.random.normal(mu_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # 2. Mutation (DE/current-to-pbest/1)
        # -----------------------------------
        # Sort by fitness
        sorted_indices = np.argsort(fitness)
        
        # Select p-best (top 11%)
        p_share = 0.11
        num_top = max(2, int(p_share * pop_size))
        top_indices = sorted_indices[:num_top]
        
        pbest_choices = np.random.choice(top_indices, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1 (distinct from current)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Handle collisions simply by resampling potentially colliding indices once
        # (Perfect collision handling is expensive, this is sufficient)
        cols = (r1_indices == np.arange(pop_size))
        if np.any(cols):
            r1_indices[cols] = np.random.randint(0, pop_size, np.sum(cols))
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Population U Archive)
        if arc_count > 0:
            pool = np.vstack((pop, archive[:arc_count]))
        else:
            pool = pop
        
        r2_indices = np.random.randint(0, pool.shape[0], pop_size)
        x_r2 = pool[r2_indices]
        
        # Compute mutant
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        # -----------------------
        rand_cross = np.random.rand(pop_size, dim)
        mask = rand_cross < CR[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutant, pop)
        
        # 4. Bound Handling
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 5. Evaluation & Selection
        # -------------------------
        success_f = []
        success_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Archive update
                if arc_count < archive_size:
                    archive[arc_count] = pop[i].copy()
                    arc_count += 1
                else:
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = pop[i].copy()
                
                # Memory update preparation
                imp = fitness[i] - f_trial
                if imp > 0:
                    success_f.append(F[i])
                    success_cr.append(CR[i])
                    diff_fitness.append(imp)
                
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()
                    
        # 6. Memory Update
        # ----------------
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            w = np.array(diff_fitness)
            if np.sum(w) == 0:
                w = np.ones_like(w)
            w = w / np.sum(w)
            
            # Weighted Lehmer Mean for F
            mean_f = np.sum(w * (s_f**2)) / np.sum(w * s_f)
            
            # Weighted Mean for CR
            mean_cr = np.sum(w * s_cr)
            
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            k_mem = (k_mem + 1) % H
            
        # 7. Convergence Check & Restart Strategy
        # ---------------------------------------
        # Check if population has collapsed (loss of diversity)
        # Using fitness range is faster and more robust than std dev
        fit_range = np.max(fitness) - np.min(fitness)
        
        # Threshold for restart: when population is essentially flat
        if fit_range < 1e-8 and not np.isinf(fit_range):
            
            # A. Local Search Polish (Coordinate Descent)
            # Before restarting, squeeze the best solution
            # Scales relative to domain size
            polish_scales = [1e-2, 1e-4, 1e-6] 
            
            current_polish = best_vec.copy()
            current_val = best_val
            
            for scale in polish_scales:
                step_size = scale * diff_b
                improved = True
                while improved:
                    improved = False
                    for d in range(dim):
                        if datetime.now() - start_time >= time_limit:
                            return best_val
                        
                        # Try moving in positive direction
                        old_val = current_polish[d]
                        current_polish[d] = np.clip(old_val + step_size[d], min_b[d], max_b[d])
                        val = func(current_polish)
                        
                        if val < current_val:
                            current_val = val
                            best_val = val
                            best_vec = current_polish.copy()
                            improved = True
                            continue
                        
                        # Try moving in negative direction
                        current_polish[d] = np.clip(old_val - step_size[d], min_b[d], max_b[d])
                        val = func(current_polish)
                        
                        if val < current_val:
                            current_val = val
                            best_val = val
                            best_vec = current_polish.copy()
                            improved = True
                            continue
                        
                        # Revert if no improvement
                        current_polish[d] = old_val
            
            # B. Hard Restart
            # Re-initialize population to explore new basins
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_vec.copy() # Keep best
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Reset Archive and Memory
            archive = np.zeros((archive_size, dim))
            arc_count = 0
            mem_f = np.full(H, 0.5)
            mem_cr = np.full(H, 0.5)
            
    return best_val
