#The following algorithm implements an improved **SHADE (Success-History based Adaptive Differential Evolution)** with **Time-Aware Adaptivity** and **Stagnation-Based Restarts**.
#
#### Reasoning for Improvements
#
#1.  **Discarding Linear Population Size Reduction (LPSR)**: Previous results showed that LPSR (Algorithms 2 & 3) performed significantly worse than fixed population size (Algorithm 1). LPSR can force premature convergence or delete good genetic material too early when the total budget (evaluations) is unknown or time-based. This algorithm uses a **fixed population size** tuned to the dimension, which proved more robust.
#2.  **Time-Based Parameter Scaling**: Instead of fixed parameters, the algorithm dynamically adjusts the "greediness" of the mutation strategy. The $p$ value in `current-to-pbest` scales linearly from exploration ($0.2$) to exploitation ($2/N$) based on the elapsed time. This ensures convergence as the time limit approaches.
#3.  **Efficient Vectorized Operations**: The code minimizes Python-level loops and overhead. Bound handling, crossover, and memory updates are fully vectorized. Time checking is batched (checked every 16 evaluations) to reduce system call overhead while maintaining strict adherence to the time limit.
#4.  **Midpoint Bound Handling**: Instead of simple clipping (which accumulates population at the edges), particles violating bounds are placed halfway between their parent and the bound. This maintains diversity and search capability near the boundaries.
#5.  **Robust Restart Mechanism**: To prevent stagnation in local optima, the algorithm monitors the fitness range. If the population collapses (max fitness $\approx$ min fitness), it triggers a restart, keeping only the global best solution and re-initializing the rest to find new basins of attraction.
#
#### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using SHADE with Time-Based Adaptivity and 
    Stagnation-Based Restarts.
    """
    
    # --- Time Management ---
    t_start = time.time()
    # Reserve 2% buffer to ensure we return before the hard limit
    t_limit = max_time * 0.98
    
    def check_timeout():
        return (time.time() - t_start) >= t_limit

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size
    # Fixed size based on dimension, clamped to safe range [40, 200]
    # This avoids the failure modes of LPSR in short time windows.
    pop_size = int(20 * dim)
    pop_size = max(40, min(200, pop_size))
    
    # External Archive (Stores successful parents)
    # Fixed size numpy array for efficiency (avoiding list appends)
    max_arc_size = int(2.0 * pop_size)
    archive = np.zeros((max_arc_size, dim))
    arc_count = 0
    
    # SHADE Memory Parameters (History size H=6)
    H = 6
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # --- Initial Population ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        # Check time periodically (bitwise check is fast)
        if (i & 15) == 0 and check_timeout():
            return best_val
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop[i].copy()
            
    # Sort population by fitness (required for p-best selection)
    sort_idx = np.argsort(fitness)
    pop = pop[sort_idx]
    fitness = fitness[sort_idx]
    
    # --- Main Loop ---
    while not check_timeout():
        
        # 1. Stagnation Check & Restart
        # If population diversity collapses (all individuals are effectively same fitness)
        # fitness is sorted, so range is last - first
        fit_range = fitness[-1] - fitness[0]
        if fit_range < 1e-8:
            # Restart: Keep best (at index 0), randomize rest
            pop[1:] = min_b + np.random.rand(pop_size - 1, dim) * diff_b
            
            # Re-evaluate
            for i in range(1, pop_size):
                if (i & 15) == 0 and check_timeout(): return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = pop[i].copy()
            
            # Reset Archive count and re-sort
            arc_count = 0
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            continue

        # 2. Time-Based Dynamic p-value
        # Linearly decrease p from 0.2 (exploration) to 2/N (exploitation)
        elapsed = time.time() - t_start
        progress = min(1.0, elapsed / t_limit)
        
        p_min = 2.0 / pop_size
        p_max = 0.2
        p_val = p_max - (p_max - p_min) * progress
        # Ensure at least 2 individuals in p-best
        n_pbest = int(max(2, p_val * pop_size))
        
        # 3. Parameter Generation
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idxs]
        m_f = mem_f[r_idxs]
        
        # CR: Normal(m_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        np.clip(cr, 0, 1, out=cr)
        
        # F: Cauchy(m_f, 0.1)
        # F = m_f + 0.1 * tan(pi * (rand - 0.5))
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Repair F
        f[f > 1] = 1.0
        # If F <= 0, regenerate until positive
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            n_bad = np.sum(mask_bad)
            f[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
            f[f > 1] = 1.0
            
        # 4. Mutation: current-to-pbest/1
        # Select p-best indices (top n_pbest)
        pbest_indices = np.random.randint(0, n_pbest, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (!= i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        mask_self = r1_indices == np.arange(pop_size)
        r1_indices[mask_self] = (r1_indices[mask_self] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (!= r1, != i) from Population U Archive
        # We sample from [0, pop_size + arc_count)
        n_union = pop_size + arc_count
        r2_indices = np.random.randint(0, n_union, pop_size)
        
        # Construct x_r2 array using efficient masking
        x_r2 = np.empty((pop_size, dim))
        mask_pop = r2_indices < pop_size
        mask_arc = ~mask_pop
        
        if np.any(mask_pop):
            x_r2[mask_pop] = pop[r2_indices[mask_pop]]
        if np.any(mask_arc):
            # Map index to archive range
            arc_idxs = r2_indices[mask_arc] - pop_size
            x_r2[mask_arc] = archive[arc_idxs]
            
        # Compute V
        f_exp = f[:, None]
        v = pop + f_exp * (x_pbest - pop) + f_exp * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cr = np.random.rand(pop_size, dim) < cr[:, None]
        mask_cr[np.arange(pop_size), j_rand] = True
        u = np.where(mask_cr, v, pop)
        
        # 6. Bound Handling (Midpoint)
        # u = (parent + bound) / 2 if violated
        mask_l = u < min_b
        mask_r = u > max_b
        
        if np.any(mask_l):
            cols = np.where(mask_l)[1]
            u[mask_l] = (pop[mask_l] + min_b[cols]) / 2.0
        if np.any(mask_r):
            cols = np.where(mask_r)[1]
            u[mask_r] = (pop[mask_r] + max_b[cols]) / 2.0
            
        # 7. Selection
        new_fitness = np.empty(pop_size)
        success_mask = np.zeros(pop_size, dtype=bool)
        diff_f = np.zeros(pop_size)
        
        for i in range(pop_size):
            if (i & 15) == 0 and check_timeout(): return best_val
            
            val = func(u[i])
            new_fitness[i] = val
            
            if val <= fitness[i]:
                success_mask[i] = True
                diff_f[i] = fitness[i] - val
                
                if val < best_val:
                    best_val = val
                    best_sol = u[i].copy()
        
        # 8. Updates (Archive, Population, Memory)
        success_idx = np.where(success_mask)[0]
        n_success = len(success_idx)
        
        if n_success > 0:
            # Update Archive with replaced parents
            # Fill strictly if space, else random replace
            if arc_count + n_success <= max_arc_size:
                archive[arc_count : arc_count + n_success] = pop[success_idx]
                arc_count += n_success
            else:
                # Fill remaining spots
                space = max_arc_size - arc_count
                if space > 0:
                    archive[arc_count:] = pop[success_idx[:space]]
                    arc_count = max_arc_size
                    rem_idx = success_idx[space:]
                else:
                    rem_idx = success_idx
                
                # Random replacement for the rest
                if len(rem_idx) > 0:
                    rand_pos = np.random.randint(0, max_arc_size, len(rem_idx))
                    archive[rand_pos] = pop[rem_idx]
            
            # Update Population
            pop[success_idx] = u[success_idx]
            fitness[success_idx] = new_fitness[success_idx]
            
            # Update Memory (Weighted Lehmer Mean)
            # Only use strict improvements
            better_mask = diff_f[success_idx] > 0
            if np.any(better_mask):
                real_idx = success_idx[better_mask]
                df = diff_f[real_idx]
                scr = cr[real_idx]
                sf = f[real_idx]
                
                total_df = np.sum(df)
                if total_df > 1e-15:
                    weights = df / total_df
                    
                    # M_CR update
                    m_cr_new = np.sum(weights * scr)
                    mem_cr[k_mem] = m_cr_new
                    
                    # M_F update (Lehmer mean)
                    mean_num = np.sum(weights * (sf**2))
                    mean_den = np.sum(weights * sf)
                    m_f_new = mean_num / mean_den if mean_den > 1e-15 else 0.5
                    mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                    
                    k_mem = (k_mem + 1) % H
                    
        # Sort for next iteration (p-best)
        sort_order = np.argsort(fitness)
        pop = pop[sort_order]
        fitness = fitness[sort_order]

    return best_val
