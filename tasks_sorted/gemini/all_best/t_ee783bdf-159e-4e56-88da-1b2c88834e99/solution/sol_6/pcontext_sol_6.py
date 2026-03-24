#The output value of the best generated algorithm is: 9.107165697037004.
#The previous best result was achieved using **SHADE** with a restart mechanism and midpoint bound handling. Other attempts with Linear Population Size Reduction (L-SHADE) or complex local searches performed worse, likely due to loss of diversity or inefficient budget usage.
#
#To improve upon the 9.10 result, I propose an **Enhanced SHADE with Stagnation-Triggered Local Search and Soft Restart**.
#
#**Key Improvements:**
#1.  **Refined Bound Handling**: We stick to the **Midpoint Target** strategy (`(old + bound) / 2`) as it outperformed clipping in previous iterations by preserving search direction near boundaries.
#2.  **Stagnation Detection**: Instead of just monitoring population variance (standard deviation), we also track "Generations without Improvement".
#3.  **Local Search Polish**: Before restarting, we apply a lightweight **Coordinate Descent** on the best individual. This ensures we have fully exploited the current basin of attraction before abandoning it.
#4.  **Soft Restart**: Upon stagnation, we keep the improved best solution, reset the SHADE memory (to learn new landscape features), and re-initialize the rest of the population to find new basins.
#5.  **Dynamic P-value**: The `p` parameter in `current-to-pbest` selection is randomized per generation to alternate between greedy exploitation and broader exploration.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Enhanced SHADE algorithm with Midpoint Bound Handling, 
    Coordinate Descent Local Search on stagnation, 
    and Soft Restart mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper: Bound Handling (Midpoint Target) ---
    def apply_bound_handling(trial, old, min_b, max_b):
        """
        If a variable violates a bound, set it to the midpoint 
        between the previous valid value and the bound.
        """
        # Lower bound violation
        mask_l = trial < min_b
        if np.any(mask_l):
            trial[mask_l] = (old[mask_l] + min_b[mask_l]) / 2.0
            
        # Upper bound violation
        mask_h = trial > max_b
        if np.any(mask_h):
            trial[mask_h] = (old[mask_h] + max_b[mask_h]) / 2.0
        return trial

    # --- Helper: Local Search (Coordinate Descent) ---
    def local_search(current_best, current_fit, bounds_np, time_limit, start_time, budget):
        """
        Performs a quick coordinate descent on the best solution 
        to refine it before a restart.
        """
        x = current_best.copy()
        fit = current_fit
        d_len = len(x)
        min_b = bounds_np[:, 0]
        max_b = bounds_np[:, 1]
        diff_b = max_b - min_b
        
        # Initial step size: 1% of domain
        step_sizes = diff_b * 0.01
        
        evals = 0
        improved = True
        
        while evals < budget and improved:
            if (datetime.now() - start_time) >= time_limit:
                break
            
            improved = False
            for d in range(d_len):
                if evals >= budget: break
                
                old_val = x[d]
                
                # Try Forward
                x[d] = np.clip(old_val + step_sizes[d], min_b[d], max_b[d])
                f_new = func(x)
                evals += 1
                if f_new < fit:
                    fit = f_new
                    improved = True
                    continue
                
                # Try Backward
                x[d] = np.clip(old_val - step_sizes[d], min_b[d], max_b[d])
                f_new = func(x)
                evals += 1
                if f_new < fit:
                    fit = f_new
                    improved = True
                    continue
                
                # Revert if no improvement
                x[d] = old_val
            
            # If no improvement in a full sweep, shrink step sizes
            if not improved:
                step_sizes *= 0.5
                # Continue only if steps are still significant (> 1e-6 relative to bounds)
                if np.any(step_sizes > (diff_b * 1e-6)):
                    improved = True
                else:
                    improved = False
                    
        return x, fit

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Population Size: Adaptive based on dimension, clamped for efficiency
    pop_size = int(np.clip(20 * dim, 40, 120))
    
    # SHADE Memory Parameters
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Archive setup
    archive = []
    
    # Initial Population
    pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
    fitness = np.full(pop_size, float('inf'))
    
    best_idx = -1
    best_fit = float('inf')
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_fit
        val = func(pop[i])
        fitness[i] = val
        if val < best_fit:
            best_fit = val
            best_idx = i
            
    if best_idx == -1: return float('inf')
    
    # Stagnation counter
    gens_no_improve = 0
    
    # --- Main Loop ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Stagnation Check & Restart
        # Calculate population diversity
        fit_std = np.std(fitness)
        stagnant = False
        
        # Conditions for restart:
        # a) Population has converged (low variance)
        # b) Global best hasn't improved for many generations
        if fit_std < 1e-6 or gens_no_improve > 50:
            stagnant = True
            
        if stagnant:
            # Phase A: Local Polish
            # Spend a small budget to refine the best solution
            ls_budget = max(10, 5 * dim)
            bx, bf = local_search(pop[best_idx], best_fit, bounds_np, time_limit, start_time, ls_budget)
            
            # Update best if improved
            if bf < best_fit:
                best_fit = bf
                pop[best_idx] = bx
                fitness[best_idx] = bf
            else:
                bx = pop[best_idx].copy() # Ensure we have the current best
            
            # Phase B: Soft Restart
            # Place best at index 0
            pop[0] = bx
            fitness[0] = best_fit
            best_idx = 0
            
            # Re-initialize the rest of the population
            pop[1:] = min_b + np.random.rand(pop_size - 1, dim) * (max_b - min_b)
            
            # Reset memory and archive to adapt to new exploration
            fitness[1:] = float('inf')
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            archive = []
            gens_no_improve = 0
            
            # Evaluate new individuals
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit: return best_fit
                val = func(pop[i])
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_idx = i
            
            continue # Skip normal DE step this iteration

        # 2. Parameter Generation (SHADE)
        r_idx = np.random.randint(0, H, pop_size)
        
        # CR: Normal distribution
        cr = np.random.normal(mem_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F: Cauchy distribution
        f = mem_f[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
        # Repair F (if <= 0 regenerate, if > 1 clip)
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            f[mask_neg] = mem_f[r_idx][mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
        f = np.clip(f, 0, 1)
        
        # 3. Mutation (current-to-pbest/1)
        # Dynamic p: Randomly select greediness between 2/N and 0.2
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        top_cnt = int(max(2, p_val * pop_size))
        
        # Get p-best indices
        sorted_idx = np.argsort(fitness)
        pbest_indices = sorted_idx[:top_cnt]
        
        # Select pbest for each individual
        p_choices = np.random.choice(pbest_indices, pop_size)
        x_pbest = pop[p_choices]
        
        # Select r1 (distinct from i)
        r1_idx = np.random.randint(0, pop_size, pop_size)
        hits = r1_idx == np.arange(pop_size)
        r1_idx[hits] = (r1_idx[hits] + 1) % pop_size
        x_r1 = pop[r1_idx]
        
        # Select r2 (distinct from i, r1) from Union(Pop, Archive)
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
            
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idx]
        
        # Calculate Mutant Vector
        f_col = f.reshape(-1, 1)
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, pop)
        
        # 5. Bound Handling (Midpoint Target)
        min_b_all = np.tile(min_b, (pop_size, 1))
        max_b_all = np.tile(max_b, (pop_size, 1))
        trial = apply_bound_handling(trial, pop, min_b_all, max_b_all)
        
        # 6. Evaluation and Selection
        succ_cr = []
        succ_f = []
        diff_fit = []
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        global_improved = False
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fit
            
            f_tri = func(trial[i])
            
            if f_tri <= fitness[i]:
                # Improvement
                if f_tri < fitness[i]:
                    archive.append(pop[i].copy())
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diff_fit.append(fitness[i] - f_tri)
                
                new_pop[i] = trial[i]
                new_fitness[i] = f_tri
                
                if f_tri < best_fit:
                    best_fit = f_tri
                    best_idx = i
                    global_improved = True
        
        pop = new_pop
        fitness = new_fitness
        
        # Update stagnation counter
        if global_improved:
            gens_no_improve = 0
        else:
            gens_no_improve += 1
        
        # 7. Update Archive (Limit size)
        target_arc = int(pop_size * 2.0)
        while len(archive) > target_arc:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 8. Update Memory (Weighted Lehmer Mean)
        if len(diff_fit) > 0:
            diff_fit = np.array(diff_fit)
            succ_cr = np.array(succ_cr)
            succ_f = np.array(succ_f)
            
            w = diff_fit / np.sum(diff_fit)
            
            m_cr = np.sum(w * succ_cr)
            
            num = np.sum(w * (succ_f**2))
            den = np.sum(w * succ_f)
            m_f = num / den if den > 0 else 0.5
            
            mem_cr[k_mem] = m_cr
            mem_f[k_mem] = np.clip(m_f, 0, 1)
            k_mem = (k_mem + 1) % H

    return best_fit
