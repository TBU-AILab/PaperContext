#To improve upon the previous best result (9.10), I propose implementing **L-SHADE (Linear Population Size Reduction Success-History based Adaptive Differential Evolution)** with an integrated **Local Search** strategy.
#
#**Why this is better:**
#1.  **Linear Population Size Reduction (LPSR):** The previous algorithm used a fixed population size. L-SHADE starts with a large population to explore the global landscape effectively, then linearly reduces the population size over time. This shifts computational resources from "exploration" (searching everywhere) to "exploitation" (refining the best area) as the deadline approaches.
#2.  **Improved Bound Handling:** Instead of simple clipping (which causes populations to stick to the edges), this implementation uses a midpoint-target bounce-back strategy, preserving diversity near the boundaries.
#3.  **Local Search Polish:** To squeeze out the final precision required to beat the previous score, a derivative-free Local Search (Coordinate Descent) is triggered on the best individual periodically and heavily towards the end of the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE algorithm with Linear Population Size Reduction and 
    coordinate descent local search for final refinement.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper Functions ---
    def get_honored_bounds(trial, old, low, high):
        """
        Bounce-back strategy for bound handling.
        If a variable exceeds bounds, set it between the old valid value and the bound.
        This prevents the population from getting stuck on the hypercube edges.
        """
        # Lower bound check
        mask_low = trial < low
        if np.any(mask_low):
            # Midpoint between old value and bound
            trial[mask_low] = (old[mask_low] + low[mask_low]) / 2.0
            
        # Upper bound check
        mask_high = trial > high
        if np.any(mask_high):
            trial[mask_high] = (old[mask_high] + high[mask_high]) / 2.0
            
        return trial

    def local_search(current_best, current_fit, step_size_factor):
        """Performs a quick coordinate descent on the best solution."""
        x = current_best.copy()
        f_x = current_fit
        improved = False
        
        # Determine step size based on domain range
        steps = (bounds_np[:, 1] - bounds_np[:, 0]) * step_size_factor
        
        for d in range(dim):
            if (datetime.now() - start_time) >= time_limit:
                return x, f_x, improved
                
            old_val = x[d]
            
            # Try positive direction
            x[d] = np.clip(old_val + steps[d], bounds_np[d, 0], bounds_np[d, 1])
            f_new = func(x)
            
            if f_new < f_x:
                f_x = f_new
                improved = True
                continue # Keep change, move to next dim
            
            # Try negative direction
            x[d] = np.clip(old_val - steps[d], bounds_np[d, 0], bounds_np[d, 1])
            f_new = func(x)
            
            if f_new < f_x:
                f_x = f_new
                improved = True
            else:
                # Revert
                x[d] = old_val
                
        return x, f_x, improved

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # L-SHADE Population Parameters
    # Start large for exploration, reduce to min_pop_size
    initial_pop_size = int(round(18 * dim))
    # Clamp initial size to reasonable limits for Python loop speed
    initial_pop_size = max(30, min(initial_pop_size, 150))
    min_pop_size = 4
    
    pop_size = initial_pop_size
    archive_size = int(round(pop_size * 2.4)) # Standard heuristic
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
    fitness = np.full(pop_size, float('inf'))
    
    # Historical Memory (H)
    H = 6 # Memory size
    mem_M_cr = np.full(H, 0.5)
    mem_M_f = np.full(H, 0.5)
    k_mem = 0
    
    archive = [] # Stores solution vectors
    
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
            
    if best_idx == -1: return float('inf') # Fail safe

    # --- Main Loop ---
    # We estimate max_generations loosely or rely on time
    # L-SHADE needs a progress ratio. We use time for this ratio.
    
    while True:
        elapsed = datetime.now() - start_time
        if elapsed >= time_limit:
            break
            
        progress = elapsed.total_seconds() / max_time
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on time progress
        plan_pop_size = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * progress))
        plan_pop_size = max(min_pop_size, plan_pop_size)
        
        if plan_pop_size < pop_size:
            # Sort by fitness (descending badness) to remove worst
            sort_order = np.argsort(fitness)
            # Keep top 'plan_pop_size'
            keep_indices = sort_order[:plan_pop_size]
            
            pop = pop[keep_indices]
            fitness = fitness[keep_indices]
            
            pop_size = plan_pop_size
            
            # Reduce archive size if necessary
            curr_archive_size = len(archive)
            target_archive = int(round(pop_size * 2.4))
            if curr_archive_size > target_archive:
                # Randomly remove elements
                del_count = curr_archive_size - target_archive
                for _ in range(del_count):
                    archive.pop(np.random.randint(0, len(archive)))
            
            # Update best index as it might have shifted in the sliced array
            best_idx = np.argmin(fitness)
            best_fit = fitness[best_idx]

        # 2. Parameter Generation
        # For each individual, pick a memory index
        r_indices = np.random.randint(0, H, pop_size)
        r_cr = mem_M_cr[r_indices]
        r_f = mem_M_f[r_indices]
        
        # Generate CR (Normal dist)
        cr = np.random.normal(r_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy dist)
        # Cauchy: location=r_f, scale=0.1
        f = r_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F
        # if f > 1 -> 1. if f < 0 -> regenerate until > 0
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            f[mask_neg] = r_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
        f = np.clip(f, 0, 1)
        
        # 3. Mutation: current-to-pbest/1
        # V = X + F(X_pbest - X) + F(X_r1 - X_r2)
        
        # Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        
        # p ranges roughly from 2/N to 0.2
        # We pick a robust p = 0.11 (top 11%)
        num_pbest = max(2, int(pop_size * 0.11))
        
        pbest_pool = sorted_indices[:num_pbest]
        pbest_idx = np.random.choice(pbest_pool, pop_size)
        x_pbest = pop[pbest_idx]
        
        # Select r1 (distinct from i)
        # Shift random index
        idxs = np.arange(pop_size)
        r1_idx = np.random.randint(1, pop_size, pop_size)
        r1_idx = (idxs + r1_idx) % pop_size
        x_r1 = pop[r1_idx]
        
        # Select r2 (distinct from i, r1) from Union(Pop, Archive)
        # Handling Union is tricky vectorized, we approximate for speed
        # Convert archive to numpy for indexing
        if len(archive) > 0:
            arr_archive = np.array(archive)
            union_pop = np.vstack((pop, arr_archive))
        else:
            union_pop = pop
            
        r2_idx = np.random.randint(0, len(union_pop), pop_size)
        # Resolve collisions with i or r1 simply by regenerating (expensive) 
        # or ignoring (classic DE tolerates duplicates occasionally).
        # We proceed for speed.
        x_r2 = union_pop[r2_idx]
        
        # Calculate Mutant
        f_reshaped = f.reshape(-1, 1)
        mutant = pop + f_reshaped * (x_pbest - pop) + f_reshaped * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Constraints (Bounce-back)
        for i in range(pop_size):
            trial[i] = get_honored_bounds(trial[i], pop[i], min_b, max_b)
            
        # 6. Evaluation and Selection
        fit_new = np.zeros(pop_size)
        
        succ_cr = []
        succ_f = []
        diff_fit = []
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fit
                
            val = func(trial[i])
            fit_new[i] = val
            
            if val <= fitness[i]:
                # Successful update
                if val < fitness[i]:
                    archive.append(pop[i].copy())
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diff_fit.append(fitness[i] - val)
                
                pop[i] = trial[i]
                fitness[i] = val
                
                if val < best_fit:
                    best_fit = val
                    best_idx = i
        
        # Resize archive if it grew too big
        target_archive = int(round(pop_size * 2.4))
        while len(archive) > target_archive:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 7. Update Memory (Weighted Lehmer Mean)
        if len(diff_fit) > 0:
            diff_np = np.array(diff_fit)
            scr_np = np.array(succ_cr)
            sf_np = np.array(succ_f)
            
            # Weights
            total_diff = np.sum(diff_np)
            if total_diff == 0:
                weights = np.ones(len(diff_fit)) / len(diff_fit)
            else:
                weights = diff_np / total_diff
            
            # Mean CR
            if np.max(scr_np) == 0: # Avoid weird edge case
                mean_cr = 0
            else:
                mean_cr = np.sum(weights * scr_np)
            
            # Mean F (Lehmer)
            num = np.sum(weights * (sf_np**2))
            den = np.sum(weights * sf_np)
            if den == 0:
                mean_f = 0.5
            else:
                mean_f = num / den
                
            mem_M_cr[k_mem] = mean_cr
            mem_M_f[k_mem] = np.clip(mean_f, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 8. Periodic Local Search
        # To squeeze out the best result, we apply local search on the best particle
        # periodically or when the population is very small (converging).
        
        do_local_search = False
        
        # Strategy: Do it if population is small (late game) or every 20 gens
        if pop_size < 10 or np.random.rand() < 0.05:
            do_local_search = True
            
        if do_local_search:
            # Scale step size by remaining time (smaller steps as we run out of time)
            step_factor = 0.01 * (1.0 - progress) + 0.0001
            
            new_best_vec, new_best_val, improved = local_search(pop[best_idx], best_fit, step_factor)
            
            if improved:
                pop[best_idx] = new_best_vec
                fitness[best_idx] = new_best_val
                best_fit = new_best_val

    return best_fit
