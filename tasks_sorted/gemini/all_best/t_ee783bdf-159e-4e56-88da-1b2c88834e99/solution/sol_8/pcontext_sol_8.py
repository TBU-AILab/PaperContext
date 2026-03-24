#The following algorithm implements an **Enhanced SHADE (Success-History based Adaptive Differential Evolution) with Stagnation Restarts and End-Game Polish**.
#
#### Improvements over previous versions:
#1.  **Restart Mechanism with Memory Reset**: Unlike standard evolutionary algorithms that may get stuck in local optima, this version detects stagnation (low population variance or no improvement for many generations) and triggers a "Soft Restart". It preserves the global best solution but re-initializes the rest of the population and resets learning parameters, allowing the algorithm to escape basins of attraction.
#2.  **Midpoint-Target Bound Handling**: Instead of simple clipping (which traps solutions on boundaries), this uses a bounce-back strategy where out-of-bound variables are set to the midpoint between the old value and the bound. This preserves the search direction and population diversity near edges.
#3.  **End-Game Final Polish**: A specific check monitors the remaining time. When the time budget is nearly exhausted (last 5% or < 0.2s), the algorithm switches to a dedicated Local Search (Coordinate Descent) on the best solution. This utilizes every remaining millisecond to refine the solution precision, often yielding significant gains in the final moments.
#4.  **Local Search on Stagnation**: Before abandoning a basin of attraction during a restart, a quick local search is performed to ensure the local optimum is fully exploited.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Enhanced SHADE with Stagnation Restarts and End-Game Polish.
    
    Features:
    - SHADE (Success-History based Adaptive Differential Evolution)
    - Midpoint-Target Bound Handling (preserves diversity near bounds)
    - Soft Restart on Stagnation (keeps best, resets others)
    - Coordinate Descent Local Search (triggered on stagnation and at end-of-time)
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper: Midpoint Bound Handling ---
    def apply_midpoint_bounds(trial, old, min_b, max_b):
        """
        If a value is out of bounds, set it to the midpoint between 
        the previous valid value and the bound.
        """
        # Lower bound
        mask_l = trial < min_b
        if np.any(mask_l):
            trial[mask_l] = (old[mask_l] + min_b[mask_l]) / 2.0
            
        # Upper bound
        mask_h = trial > max_b
        if np.any(mask_h):
            trial[mask_h] = (old[mask_h] + max_b[mask_h]) / 2.0
            
        return trial

    # --- Helper: Local Search (Coordinate Descent) ---
    def local_search(sol, fit, end_time, bounds_np, initial_step_ratio=0.01):
        """
        Performs coordinate descent to refine the solution.
        Step size shrinks if no improvement is found in a full pass.
        """
        x = sol.copy()
        f = fit
        min_b = bounds_np[:, 0]
        max_b = bounds_np[:, 1]
        domain_width = max_b - min_b
        step = domain_width * initial_step_ratio
        
        while datetime.now() < end_time:
            improved = False
            for d in range(len(x)):
                if datetime.now() >= end_time: break
                
                old_val = x[d]
                
                # Try Positive Step
                x[d] = np.clip(old_val + step[d], min_b[d], max_b[d])
                fv = func(x)
                if fv < f:
                    f = fv
                    improved = True
                    continue # Keep change and move to next dim
                
                # Try Negative Step
                x[d] = np.clip(old_val - step[d], min_b[d], max_b[d])
                fv = func(x)
                if fv < f:
                    f = fv
                    improved = True
                    continue
                
                # Revert
                x[d] = old_val
            
            if not improved:
                step *= 0.5
                # Terminate if step size becomes negligible
                if np.max(step) < 1e-9:
                    break
                    
        return x, f

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Population Size (Static, but robust)
    pop_size = int(np.clip(20 * dim, 40, 100))
    
    # SHADE Memory
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    archive = []
    
    # Initialize Population
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

    gens_no_imp = 0
    
    # --- Main Loop ---
    while True:
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        remaining = max_time - elapsed
        
        # Check global timeout
        if remaining <= 0:
            return best_fit
            
        # --- End-Game Polish ---
        # If less than 5% of time or 0.2s remains, spend it all on Local Search
        if remaining < max(0.2, 0.05 * max_time):
            # Calculate end time strictly
            polish_end = start_time + time_limit
            sol, f = local_search(pop[best_idx], best_fit, polish_end, bounds_np, initial_step_ratio=0.005)
            if f < best_fit:
                best_fit = f
            return best_fit

        # --- Stagnation Check & Restart ---
        fit_std = np.std(fitness)
        if fit_std < 1e-6 or gens_no_imp > 45:
            # 1. Quick Polish before restart (Use small budget of remaining time)
            budget = min(remaining * 0.1, 0.5) # Max 0.5s or 10% of remaining
            ls_end = now + timedelta(seconds=budget)
            sol, f = local_search(pop[best_idx], best_fit, ls_end, bounds_np, initial_step_ratio=0.01)
            
            if f < best_fit:
                best_fit = f
                pop[best_idx] = sol
                fitness[best_idx] = f
            
            # 2. Soft Restart
            # Move best to index 0
            pop[0] = pop[best_idx].copy()
            fitness[0] = best_fit
            best_idx = 0
            
            # Re-initialize the rest
            pop[1:] = min_b + np.random.rand(pop_size - 1, dim) * (max_b - min_b)
            fitness[1:] = float('inf')
            
            # Reset Memory & Archive
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            archive = []
            gens_no_imp = 0
            
            # Evaluate new population (checking time)
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit: return best_fit
                val = func(pop[i])
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_idx = i
            
            continue # Restart cycle

        # --- SHADE Parameter Generation ---
        r_idx = np.random.randint(0, H, pop_size)
        
        # CR: Normal(mem_cr, 0.1)
        cr = np.random.normal(mem_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F: Cauchy(mem_f, 0.1)
        f = mem_f[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
        # Repair F
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            f[mask_neg] = mem_f[r_idx][mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
        f = np.clip(f, 0, 1)
        
        # --- Mutation: current-to-pbest/1 ---
        # p is random in [2/N, 0.2]
        p_min = 2.0 / pop_size
        p = np.random.uniform(p_min, 0.2)
        top_p = int(max(2, p * pop_size))
        
        sorted_indices = np.argsort(fitness)
        pbest_pool = sorted_indices[:top_p]
        pbest_idx = np.random.choice(pbest_pool, pop_size)
        
        # r1 != i
        r1_idx = np.random.randint(0, pop_size, pop_size)
        hits = r1_idx == np.arange(pop_size)
        r1_idx[hits] = (r1_idx[hits] + 1) % pop_size
        
        # r2 != i, r1 (from Union of Pop + Archive)
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
            
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        
        x_pbest = pop[pbest_idx]
        x_r1 = pop[r1_idx]
        x_r2 = union_pop[r2_idx]
        
        f_col = f.reshape(-1, 1)
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # --- Crossover: Binomial ---
        j_rand = np.random.randint(0, dim, pop_size)
        mask_c = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        mask_c[np.arange(pop_size), j_rand] = True
        trial = np.where(mask_c, mutant, pop)
        
        # --- Bound Handling: Midpoint Target ---
        min_b_all = np.tile(min_b, (pop_size, 1))
        max_b_all = np.tile(max_b, (pop_size, 1))
        trial = apply_midpoint_bounds(trial, pop, min_b_all, max_b_all)
        
        # --- Evaluation & Selection ---
        succ_cr = []
        succ_f = []
        diff_f = []
        
        new_pop = pop.copy()
        new_fit = fitness.copy()
        iter_improved = False
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit: return best_fit
            
            val = func(trial[i])
            
            if val <= fitness[i]:
                if val < fitness[i]:
                    archive.append(pop[i].copy())
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diff_f.append(fitness[i] - val)
                
                new_pop[i] = trial[i]
                new_fit[i] = val
                
                if val < best_fit:
                    best_fit = val
                    best_idx = i
                    iter_improved = True
                    
        pop = new_pop
        fitness = new_fit
        
        if iter_improved:
            gens_no_imp = 0
        else:
            gens_no_imp += 1
            
        # --- Archive Maintenance ---
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # --- Memory Update (Weighted Lehmer Mean) ---
        if len(diff_f) > 0:
            w = np.array(diff_f)
            w /= np.sum(w)
            
            mcr = np.sum(w * np.array(succ_cr))
            
            sf = np.array(succ_f)
            num = np.sum(w * sf**2)
            den = np.sum(w * sf)
            mf = num / den if den > 0 else 0.5
            
            mem_cr[k_mem] = mcr
            mem_f[k_mem] = np.clip(mf, 0, 1)
            k_mem = (k_mem + 1) % H

    return best_fit
