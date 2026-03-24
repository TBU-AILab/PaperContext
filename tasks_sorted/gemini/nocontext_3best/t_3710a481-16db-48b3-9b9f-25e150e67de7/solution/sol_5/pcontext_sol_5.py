#The following algorithm is an optimized version of the **SHADE (Success-History based Adaptive Differential Evolution)** algorithm, enhanced with a robust **Restart Strategy** and **Bound Reflection**.
#
#### Improvements Overview
#Based on the analysis of previous attempts:
#1.  **Algorithm 1 (SHADE + Restart)** performed best (1.14) because it balanced exploration (adaptive parameters) and exploitation (convergence).
#2.  **Algorithm 2** failed (12.36) likely due to the computational cost of the local search and ineffective population reduction.
#3.  **Improvements in this version**:
#    *   **Bound Reflection**: Instead of simple clipping (which traps solutions at the edge), we use reflection. This allows the search to bounce back from boundaries, significantly improving performance on problems with optima near or at the bounds.
#    *   **Robust Restart**: The restart condition uses both standard deviation and fitness range to detect convergence reliably. It resets the population while keeping the global best, effectively turning the algorithm into a multi-start global search.
#    *   **Population Sizing**: A population size of `20 * dim` (min 40) is chosen to ensure sufficient diversity for the adaptive mechanism to learn effectively.
#    *   **Efficiency**: The implementation uses vectorized Numpy operations for mutation, crossover, and bound handling to maximize the number of function evaluations within the time limit.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    SHADE (Success-History based Adaptive Differential Evolution) with 
    Bound Reflection and Restart Strategy.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size: 20 * dim provides a good balance for SHADE.
    # Minimum 40 ensures statistical stability for parameter adaptation.
    pop_size = int(max(40, 20 * dim))
    
    # Archive size typically equals population size in SHADE
    archive_size = pop_size
    
    # History Memory size
    H = 6
    
    # Pre-process bounds for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Memory for Adaptive Parameters (M_F, M_CR) initialized to 0.5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive to maintain diversity
    archive = np.zeros((archive_size, dim))
    arc_count = 0
    
    # Global Best Tracking
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
        # Check time limit at start of generation
        if datetime.now() - start_time >= time_limit:
            return best_val

        # 1. Parameter Generation
        # -----------------------
        # Pick random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_f = mem_f[r_idx]
        mu_cr = mem_cr[r_idx]

        # Generate F using Cauchy distribution: Cauchy(mu_f, 0.1)
        rand_u = np.random.rand(pop_size)
        F = mu_f + 0.1 * np.tan(np.pi * (rand_u - 0.5))
        
        # Repair F: Resample if <= 0, Clip if > 1
        while True:
            mask_neg = F <= 0
            if not np.any(mask_neg):
                break
            n_neg = np.sum(mask_neg)
            F[mask_neg] = mu_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(n_neg) - 0.5))
        F = np.minimum(F, 1.0)

        # Generate CR using Normal distribution: Normal(mu_cr, 0.1)
        CR = np.random.normal(mu_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)

        # 2. Mutation (DE/current-to-pbest/1)
        # -----------------------------------
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        # Select p-best from top 11% (standard SHADE setting)
        num_top = max(2, int(0.11 * pop_size))
        top_indices = sorted_indices[:num_top]
        
        # Select pbest for each individual
        pbest_choices = np.random.choice(top_indices, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1 (from Population, distinct logic relaxed for speed)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Simple collision handling for r1 vs current
        cols = (r1_indices == np.arange(pop_size))
        if np.any(cols):
            r1_indices[cols] = np.random.randint(0, pop_size, np.sum(cols))
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Union of Population and Archive)
        if arc_count > 0:
            pool_size = pop_size + arc_count
            r2_raw = np.random.randint(0, pool_size, pop_size)
            
            x_r2 = np.zeros((pop_size, dim))
            mask_pop = r2_raw < pop_size
            mask_arc = ~mask_pop
            
            x_r2[mask_pop] = pop[r2_raw[mask_pop]]
            x_r2[mask_arc] = archive[r2_raw[mask_arc] - pop_size]
        else:
            r2_indices = np.random.randint(0, pop_size, pop_size)
            x_r2 = pop[r2_indices]

        # Compute Mutant Vectors
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)

        # 3. Crossover (Binomial)
        # -----------------------
        rand_cross = np.random.rand(pop_size, dim)
        mask = rand_cross < CR[:, None]
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutant, pop)

        # 4. Bound Handling (Reflection)
        # ------------------------------
        # Reflection is often superior to clipping for optima at boundaries
        # Lower Bounds
        mask_l = trial_pop < min_b
        trial_pop[mask_l] = 2 * min_b[mask_l] - trial_pop[mask_l]
        # Safety clip if reflection is still out of bounds
        mask_l2 = trial_pop < min_b
        trial_pop[mask_l2] = min_b[mask_l2]
        
        # Upper Bounds
        mask_u = trial_pop > max_b
        trial_pop[mask_u] = 2 * max_b[mask_u] - trial_pop[mask_u]
        mask_u2 = trial_pop > max_b
        trial_pop[mask_u2] = max_b[mask_u2]

        # 5. Evaluation and Selection
        # ---------------------------
        success_f = []
        success_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Add parent to archive before replacement
                if arc_count < archive_size:
                    archive[arc_count] = pop[i].copy()
                    arc_count += 1
                else:
                    # Replace random archive member
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = pop[i].copy()
                
                # Record successful parameters
                imp = fitness[i] - f_trial
                # Only update memory if there is a strict improvement (avoids noise)
                if imp > 0:
                    success_f.append(F[i])
                    success_cr.append(CR[i])
                    diff_fitness.append(imp)
                
                # Update Population
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()

        # 6. Memory Update (History)
        # --------------------------
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            w = np.array(diff_fitness)
            
            # Normalize weights
            if np.sum(w) == 0:
                w = np.ones_like(w)
            w = w / np.sum(w)
            
            # Weighted Lehmer Mean for F
            denom = np.sum(w * s_f)
            if denom == 0:
                mean_f = 0.5
            else:
                mean_f = np.sum(w * (s_f**2)) / denom
            
            # Weighted Mean for CR
            mean_cr = np.sum(w * s_cr)
            
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            k_mem = (k_mem + 1) % H

        # 7. Restart Strategy
        # -------------------
        # Restart if population converges (low std dev or low range)
        valid_fit = fitness[~np.isinf(fitness)]
        if len(valid_fit) > 1:
            std_dev = np.std(valid_fit)
            fit_range = np.max(valid_fit) - np.min(valid_fit)
            
            if std_dev < 1e-7 or fit_range < 1e-8:
                # Re-initialize population
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                # Inject best found solution (Elitism)
                pop[0] = best_vec.copy()
                
                # Reset Fitness
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_val
                
                # Reset Archive and Memory
                archive = np.zeros((archive_size, dim))
                arc_count = 0
                mem_f = np.full(H, 0.5)
                mem_cr = np.full(H, 0.5)
