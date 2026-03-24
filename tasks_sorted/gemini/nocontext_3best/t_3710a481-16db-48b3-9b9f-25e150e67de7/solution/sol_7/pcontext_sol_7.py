#The best performing algorithm in the previous attempts was **SHADE with Restart** (Score: 1.14), which significantly outperformed Standard DE and jDE. The SHADE algorithm's success is attributed to its history-based parameter adaptation and the use of an archive to maintain diversity. The second attempt using Coordinate Descent Restart failed (Score: 12.36), likely because the local search was too computationally expensive and the reduced population size limited exploration.
#
#To improve upon the best result and approach the global minimum (0.0), I propose an **Enhanced SHADE with Midpoint-Target Bound Handling and Restart**.
#
#**Key Improvements:**
#1.  **Midpoint-Target Bound Handling**: Instead of simple clipping (which causes population bunching at the edges) or random re-initialization, we use a "midpoint bounce-back" strategy. If a trial vector violates a bound, the value is set to the midpoint between the parent value and the bound. This preserves the search direction while keeping the solution valid and improving diversity near the boundaries.
#2.  **Expanded Archive Size**: We increase the archive size to `2.0 * pop_size` (up from `1.0`). This stores more historical diversity, preventing the `current-to-pbest` mutation from leading to premature convergence too quickly.
#3.  **Refined Restart Trigger**: We use the fitness range (`max - min`) as the convergence criterion instead of standard deviation. This is a cheaper and more direct measure of population flatness. When the population converges to a point (range < 1e-8), we perform a hard restart, keeping only the global best solution to anchor the next search phase.
#4.  **Robust Parameters**: We maintain a robust population size of `max(40, 15 * dim)` to ensure adequate coverage for high-dimensional problems, balancing exploration and exploitation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Enhanced SHADE (Success-History based Adaptive Differential Evolution)
    Features:
    - History-based parameter adaptation (F, CR)
    - External Archive (size 2*NP) for diversity preservation
    - Midpoint-Target Bound Handling (superior to clipping)
    - Restart Strategy based on fitness range convergence
    """
    
    # --- Initialization & Configuration ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Population size: Linear scaling with dimension, minimum 40
    pop_size = int(max(40, 15 * dim))
    
    # Archive size: Stores inferior solutions to guide mutation
    archive_size = int(2.0 * pop_size)
    
    # History Memory size for adaptive parameters
    H = 6
    
    # Parse bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Adaptive Parameter Memory (Lehmer Mean / Arithmetic Mean)
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive
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
        # Time Check
        if datetime.now() - start_time >= time_limit:
            return best_val
            
        # 1. Parameter Generation
        # -----------------------
        # Select random memory slot for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_f = mem_f[r_idx]
        mu_cr = mem_cr[r_idx]
        
        # Generate F using Cauchy distribution
        # F_i = Cauchy(mu_f, 0.1)
        F = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Handle F constraints (F > 0, F <= 1)
        # If F <= 0, regenerate until positive
        while True:
            mask_neg = F <= 0
            if not np.any(mask_neg):
                break
            F[mask_neg] = mu_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
        
        # If F > 1, clip to 1
        F = np.minimum(F, 1.0)
        
        # Generate CR using Normal distribution
        # CR_i = Normal(mu_cr, 0.1)
        CR = np.random.normal(mu_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # 2. Mutation Strategy: current-to-pbest/1
        # ----------------------------------------
        # Sort population by fitness
        sorted_indices = np.argsort(fitness)
        
        # Select p-best individuals (top 11%)
        p_share = 0.11
        num_top = max(2, int(p_share * pop_size))
        top_indices = sorted_indices[:num_top]
        
        # Randomly choose one pbest for each individual
        pbest_indices = np.random.choice(top_indices, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (random from population, distinct from current i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Fix collision with self (i)
        cols_self = (r1_indices == np.arange(pop_size))
        if np.any(cols_self):
            r1_indices[cols_self] = np.random.randint(0, pop_size, np.sum(cols_self))
        x_r1 = pop[r1_indices]
        
        # Select r2 (random from Population UNION Archive, distinct from r1)
        if arc_count > 0:
            pool = np.vstack((pop, archive[:arc_count]))
        else:
            pool = pop
        
        r2_indices = np.random.randint(0, pool.shape[0], pop_size)
        # Fix collision with r1
        cols_r1 = (r2_indices == r1_indices)
        if np.any(cols_r1):
            r2_indices[cols_r1] = np.random.randint(0, pool.shape[0], np.sum(cols_r1))
        x_r2 = pool[r2_indices]
        
        # Compute Mutant Vector
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        # -----------------------
        rand_cross = np.random.rand(pop_size, dim)
        mask = rand_cross < CR[:, None]
        
        # Ensure at least one dimension is inherited from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutant, pop)
        
        # 4. Bound Handling (Midpoint Target)
        # -----------------------------------
        # Instead of hard clipping, place violating points between parent and bound.
        # This preserves diversity better than stacking at the edge.
        mask_lower = trial_pop < min_b
        trial_pop[mask_lower] = (pop[mask_lower] + min_b[mask_lower]) / 2.0
        
        mask_upper = trial_pop > max_b
        trial_pop[mask_upper] = (pop[mask_upper] + max_b[mask_upper]) / 2.0
        
        # 5. Evaluation and Selection
        # ---------------------------
        success_f = []
        success_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                # Add parent to archive before replacement
                if arc_count < archive_size:
                    archive[arc_count] = pop[i].copy()
                    arc_count += 1
                else:
                    # Replace random member
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = pop[i].copy()
                
                # Store successful parameters
                imp = fitness[i] - f_trial
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
                    
        # 6. Update History Memory
        # ------------------------
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            w = np.array(diff_fitness)
            
            # Normalize weights
            if np.sum(w) == 0:
                w = np.ones_like(w)
            w = w / np.sum(w)
            
            # Weighted Lehmer Mean for F
            mean_f = np.sum(w * (s_f**2)) / np.sum(w * s_f)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            # Weighted Arithmetic Mean for CR
            mean_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 7. Convergence Check & Restart
        # ------------------------------
        # Check fitness range to detect collapsed population
        fit_range = np.max(fitness) - np.min(fitness)
        
        # If population has converged to a point (and not just all inf)
        if fit_range < 1e-8 and not np.isinf(fit_range):
            # Perform Hard Restart
            # Re-initialize entire population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Keep the single best solution found so far
            # This acts as a "seed" for the next search phase
            pop[0] = best_vec.copy()
            
            # Reset fitness array
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Reset Archive and Memory to adapt to the new phase/basin
            archive = np.zeros((archive_size, dim))
            arc_count = 0
            mem_f = np.full(H, 0.5)
            mem_cr = np.full(H, 0.5)

    return best_val
