#To improve upon the best result (4.818), I propose an **Enhanced SHADE with Stagnation-Triggered Local Search and Soft Restart**.
#
#**Improvements:**
#1.  **Refined Bound Handling**: We stick to the **Midpoint Target** strategy (`(old + bound) / 2`) as it outperformed clipping in previous iterations by preserving search direction near boundaries.
#2.  **Robust Stagnation Detection**: Combines variance checking with a "generations without improvement" counter.
#3.  **End-Game Polish**: Dedicated time checks to switch to a pure Local Search (Coordinate Descent) when the time budget is nearly exhausted (< 5%).
#4.  **Soft Restart with Local Exploitation**: When restarting, we don't just randomize. We keep the best solution and generate a small cluster of solutions around it ("Near Best") to exploit the current basin, while randomizing the rest for global exploration.
#5.  **Dynamic p-value**: The `p` parameter in `current-to-pbest` selection is randomized per generation to alternate between greedy exploitation and broader exploration.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Enhanced SHADE algorithm with Midpoint Bound Handling, 
    Coordinate Descent Local Search on stagnation, 
    and Soft Restart mechanism with local exploitation.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # --- Parameters ---
    # Population Size: Adaptive based on dimension, clamped for efficiency
    pop_size = int(np.clip(15 * dim, 30, 100))
    
    # SHADE Memory Parameters
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Archive setup
    archive = []
    
    # --- Helper: Bound Handling (Midpoint Target) ---
    def apply_bound_handling(trial, old, min_b_all, max_b_all):
        """
        If a variable violates a bound, set it to the midpoint 
        between the previous valid value and the bound.
        """
        # Lower bound violation
        mask_l = trial < min_b_all
        if np.any(mask_l):
            trial[mask_l] = (old[mask_l] + min_b_all[mask_l]) * 0.5
            
        # Upper bound violation
        mask_h = trial > max_b_all
        if np.any(mask_h):
            trial[mask_h] = (old[mask_h] + max_b_all[mask_h]) * 0.5
        return trial

    # --- Helper: Local Search (Coordinate Descent) ---
    def local_search(current_best, current_fit, budget_ratio):
        """
        Performs coordinate descent to refine the solution.
        Budget depends on remaining time.
        """
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        remaining = max_time - elapsed
        
        # If practically no time left, skip
        if remaining < 1e-4: return current_best, current_fit
        
        # Allocate budget
        budget_sec = min(remaining, max_time * budget_ratio)
        ls_end = now + timedelta(seconds=budget_sec)
        
        x = current_best.copy()
        fit = current_fit
        
        # Initial step size: 0.5% of domain
        step_sizes = (max_b - min_b) * 0.005
        
        improved_global = True
        while improved_global and datetime.now() < ls_end:
            improved_global = False
            
            for d in range(dim):
                if datetime.now() >= ls_end: break
                
                old_val = x[d]
                
                # Try Forward
                x[d] = np.clip(old_val + step_sizes[d], min_b[d], max_b[d])
                f_new = func(x)
                if f_new < fit:
                    fit = f_new
                    improved_global = True
                    continue
                
                # Try Backward
                x[d] = np.clip(old_val - step_sizes[d], min_b[d], max_b[d])
                f_new = func(x)
                if f_new < fit:
                    fit = f_new
                    improved_global = True
                    continue
                
                # Revert
                x[d] = old_val
            
            if not improved_global:
                # Reduce step size
                step_sizes *= 0.5
                # Continue if steps are still meaningful
                if np.any(step_sizes > 1e-9):
                    improved_global = True
                    
        return x, fit

    # --- Initial Population ---
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
    
    # Pre-allocate tiled bounds for speed
    min_b_all = np.tile(min_b, (pop_size, 1))
    max_b_all = np.tile(max_b, (pop_size, 1))
    
    # --- Main Loop ---
    while (datetime.now() - start_time) < time_limit:
        
        # --- End-Game Polish ---
        # If less than 5% of time or 0.1s remains, switch to pure local search
        elapsed_sec = (datetime.now() - start_time).total_seconds()
        remaining_sec = max_time - elapsed_sec
        if remaining_sec < 0.1 or (remaining_sec / max_time) < 0.05:
            sol, fit = local_search(pop[best_idx], best_fit, 1.0) # Use all remaining time
            if fit < best_fit:
                best_fit = fit
            return best_fit

        # --- Stagnation Check & Restart ---
        fit_std = np.std(fitness)
        stagnant = False
        
        # Trigger if low variance OR no improvement for ~35 generations
        if fit_std < 1e-6 or gens_no_improve > 35:
            stagnant = True
            
        if stagnant:
            # 1. Local Polish before restart (5% budget)
            bx, bf = local_search(pop[best_idx], best_fit, 0.05)
            
            if bf < best_fit:
                best_fit = bf
                pop[best_idx] = bx
                fitness[best_idx] = bf
            else:
                bx = pop[best_idx].copy()
            
            # 2. Soft Restart
            # Keep Best at index 0
            pop[0] = bx
            fitness[0] = best_fit
            best_idx = 0
            
            # Generate 30% of population around the best (Exploitation)
            n_near = int(0.3 * pop_size)
            for k in range(1, 1 + n_near):
                # Gaussian cloud with sigma = 5% of domain
                noise = np.random.normal(0, 0.05, dim) * (max_b - min_b)
                candidate = np.clip(pop[0] + noise, min_b, max_b)
                pop[k] = candidate
                
            # Generate rest randomly (Exploration)
            n_rand = pop_size - 1 - n_near
            pop[1+n_near:] = min_b + np.random.rand(n_rand, dim) * (max_b - min_b)
            
            # Reset Memory & Archive
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
            
            continue # Restart cycle complete

        # --- SHADE Parameter Generation ---
        r_idx = np.random.randint(0, H, pop_size)
        
        # CR: Normal distribution
        cr = np.random.normal(mem_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F: Cauchy distribution
        f = mem_f[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            f[mask_neg] = mem_f[r_idx][mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
        f = np.clip(f, 0, 1)
        
        # --- Mutation (current-to-pbest/1) ---
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        top_cnt = int(max(2, p_val * pop_size))
        
        sorted_idx = np.argsort(fitness)
        pbest_pool = sorted_idx[:top_cnt]
        
        pbest_idx = np.random.choice(pbest_pool, pop_size)
        x_pbest = pop[pbest_idx]
        
        r1_idx = np.random.randint(0, pop_size, pop_size)
        hits = r1_idx == np.arange(pop_size)
        r1_idx[hits] = (r1_idx[hits] + 1) % pop_size
        x_r1 = pop[r1_idx]
        
        # r2 from Union(Pop, Archive)
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
            
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idx]
        
        f_col = f.reshape(-1, 1)
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # --- Crossover (Binomial) ---
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, pop)
        
        # --- Bound Handling (Midpoint Target) ---
        trial = apply_bound_handling(trial, pop, min_b_all, max_b_all)
        
        # --- Evaluation and Selection ---
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
        
        if global_improved:
            gens_no_improve = 0
        else:
            gens_no_improve += 1
        
        # --- Archive Maintenance ---
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # --- Memory Update (Weighted Lehmer Mean) ---
        if len(diff_fit) > 0:
            diff_np = np.array(diff_fit)
            w = diff_np / np.sum(diff_np)
            
            m_cr = np.sum(w * np.array(succ_cr))
            
            sf = np.array(succ_f)
            num = np.sum(w * (sf**2))
            den = np.sum(w * sf)
            m_f = num / den if den > 0 else 0.5
            
            mem_cr[k_mem] = m_cr
            mem_f[k_mem] = np.clip(m_f, 0, 1)
            k_mem = (k_mem + 1) % H

    return best_fit
