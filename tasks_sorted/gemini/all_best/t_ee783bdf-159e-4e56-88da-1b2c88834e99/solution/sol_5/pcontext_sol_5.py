#The output value of the best generated algorithm is: 9.107165697037004.
#The previous best algorithm used **SHADE** (Success-History based Adaptive Differential Evolution) with a restart mechanism. While it achieved the best result so far (9.10), it likely got stuck in a local optimum or suffered from boundary sticking.
#
#To improve upon this, I propose an optimized **SHADE** implementation with **Midpoint-Target Bound Handling** and a refined **Restart Mechanism**.
#
#**Key Improvements:**
#1.  **Midpoint-Target Bound Handling**: Instead of simple clipping (which sets values to the bound) or random bounce-back, this method sets out-of-bound variables to the midpoint between the parent value and the bound (`(parent + bound) / 2`). This preserves the search direction and prevents the population from collapsing onto the boundaries, a common issue in bounded optimization.
#2.  **Refined Restart**: The previous restart mechanism might have been too aggressive or not aggressive enough. This version monitors population diversity (standard deviation). If the population stagnates, it performs a soft restart: it keeps the single best solution found so far, re-initializes the rest of the population randomly, and resets the adaptive history to learn the new landscape.
#3.  **Dynamic p-best Selection**: The `current-to-pbest` mutation strategy now uses a randomized `p` (within a range) each generation. This balances greedy exploitation (small `p`) and broader exploration (large `p`) dynamically.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using SHADE with Midpoint Bound Handling and Restart Mechanism.
    """
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Adapted to dimension, clamped for efficiency
    # Heuristic: ~18*dim, but kept within [30, 150] to balance exploration/speed
    pop_size = int(np.clip(18 * dim, 30, 150))
    
    # SHADE memory parameters
    H = 6 # Memory size
    mem_cr = np.full(H, 0.5) # Memory for Crossover Rate
    mem_f = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0
    
    # External archive for diversity maintenance
    archive = []
    
    # Bounds preparation
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initial Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_idx = -1
    best_fit = float('inf')
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_fit
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_idx = i
            
    if best_idx == -1: return float('inf')

    # --- Main Optimization Loop ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Stagnation Check & Restart
        # Calculate statistics to detect convergence
        fit_std = np.std(fitness)
        fit_range = np.max(fitness) - np.min(fitness)
        
        # Restart if population converged (stagnation)
        # Thresholds: std < 1e-5 or range < 1e-6 implies zero diversity
        if fit_std < 1e-5 or fit_range < 1e-6:
            # Preserve Global Best
            saved_best_vec = pop[best_idx].copy()
            saved_best_val = best_fit
            
            # Reset Population (Randomize)
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Elitism: Place best individual at index 0
            pop[0] = saved_best_vec
            
            # Reset Fitness (mark as inf to force re-eval/update logic, except best)
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = saved_best_val
            best_idx = 0
            
            # Reset Memory & Archive (new basin might need different params)
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            archive = []
            
            # Re-evaluate new population (skip index 0, which is best)
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit: return best_fit
                val = func(pop[i])
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_idx = i
            
            # Continue to next iteration immediately
            continue

        # 2. Parameter Generation (SHADE Adaptive Logic)
        r_idx = np.random.randint(0, H, pop_size)
        
        # CR: Normal(mem_cr, 0.1)
        cr = np.random.normal(mem_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F: Cauchy(mem_f, 0.1)
        f = mem_f[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F: if F <= 0, regenerate. if F > 1, clip to 1.
        retry_limit = 10
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            retry_limit -= 1
            if retry_limit < 0:
                f[mask_neg] = 0.5 # Fallback
                break
            count = np.sum(mask_neg)
            f[mask_neg] = mem_f[r_idx][mask_neg] + 0.1 * np.random.standard_cauchy(count)
        
        f = np.clip(f, 0, 1)
        
        # 3. Mutation: current-to-pbest/1
        # Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        
        # Dynamic p-value: Randomly choose p between 2/N and 0.2
        # This varies selection pressure: smaller p = more greedy/exploitation
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        num_pbest = int(max(2, p_val * pop_size))
        pbest_candidates = sorted_indices[:num_pbest]
        
        # Select p-best indices
        pbest_idx = np.random.choice(pbest_candidates, pop_size)
        x_pbest = pop[pbest_idx]
        
        # Select r1 distinct from i
        r1_idx = np.random.randint(0, pop_size, pop_size)
        hits = r1_idx == np.arange(pop_size)
        r1_idx[hits] = (r1_idx[hits] + 1) % pop_size
        x_r1 = pop[r1_idx]
        
        # Select r2 from Union(Population, Archive)
        if len(archive) > 0:
            arc_arr = np.array(archive)
            union_pop = np.vstack((pop, arc_arr))
        else:
            union_pop = pop
            
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idx]
        
        # Compute Mutant Vectors
        f_col = f.reshape(-1, 1)
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover: Binomial
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Handling: Midpoint Target
        # Instead of clipping, place out-of-bound solutions halfway between parent and bound.
        # This avoids stacking on the edges and preserves search direction.
        min_b_broad = np.tile(min_b, (pop_size, 1))
        max_b_broad = np.tile(max_b, (pop_size, 1))
        
        mask_l = trial < min_b_broad
        if np.any(mask_l):
            trial[mask_l] = (pop[mask_l] + min_b_broad[mask_l]) / 2.0
            
        mask_h = trial > max_b_broad
        if np.any(mask_h):
            trial[mask_h] = (pop[mask_h] + max_b_broad[mask_h]) / 2.0
            
        # 6. Evaluation & Selection
        success_cr = []
        success_f = []
        diff_fitness = []
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fit
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Improvement or neutral move
                if f_trial < fitness[i]:
                    archive.append(pop[i].copy())
                    success_cr.append(cr[i])
                    success_f.append(f[i])
                    diff_fitness.append(fitness[i] - f_trial)
                
                new_pop[i] = trial[i]
                new_fitness[i] = f_trial
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_idx = i
        
        pop = new_pop
        fitness = new_fitness
        
        # 7. Update Memory & Archive
        # Limit Archive Size
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # Update Historical Memory (Weighted Lehmer Mean)
        if len(diff_fitness) > 0:
            w = np.array(diff_fitness)
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
                
                s_cr = np.array(success_cr)
                s_f = np.array(success_f)
                
                mean_cr = np.sum(w * s_cr)
                
                # Lehmer Mean for F
                mean_f_num = np.sum(w * (s_f ** 2))
                mean_f_den = np.sum(w * s_f)
                if mean_f_den > 0:
                    mean_f = mean_f_num / mean_f_den
                else:
                    mean_f = 0.5
                    
                mem_cr[k_mem] = mean_cr
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                
                k_mem = (k_mem + 1) % H

    return best_fit
