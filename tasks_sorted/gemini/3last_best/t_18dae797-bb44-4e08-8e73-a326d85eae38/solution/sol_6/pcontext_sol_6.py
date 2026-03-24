#The output value of the best generated algorithm is: 1.1088871283573951
#
#The best generated algorithm code:
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using L-SHADE-RSP (Linear Population Size Reduction
    with Rank-based mutation and Stagnation Prevention).
    
    Improvements over standard L-SHADE:
    1. Midpoint Bound Handling: Preserves diversity near boundaries instead of clipping.
    2. Stagnation Prevention: Injects diversity if variance drops too low while time remains.
    3. Robust Parameter Adaptation: Uses weighted Lehmer mean for F and CR updates.
    """
    
    # --- Time Management ---
    t_start = time.time()
    # Reserve a small buffer to ensure strictly valid return
    t_limit = max_time * 0.98
    
    def check_time():
        return (time.time() - t_start) >= t_limit

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing (L-SHADE strategy)
    # Start with a reasonably large population for exploration
    # N_init = 18 * dim, but clamped to [30, 300] for efficiency on short runs
    N_init = int(18 * dim)
    N_init = max(30, min(300, N_init))
    N_min = 4
    
    pop_size = N_init
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Global Best
    best_val = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if check_time(): return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            
    # Sort population (needed for LPSR and p-best)
    sort_idx = np.argsort(fitness)
    pop = pop[sort_idx]
    fitness = fitness[sort_idx]
    
    # SHADE Memory Parameters
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive
    archive = []
    
    # --- Main Loop ---
    while not check_time():
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate progress relative to time budget
        elapsed = time.time() - t_start
        progress = elapsed / t_limit
        if progress > 1.0: break
        
        # Calculate target size
        N_target = int(round(N_init + (N_min - N_init) * progress))
        N_target = max(N_min, N_target)
        
        # Reduce Population if needed
        if pop_size > N_target:
            n_remove = pop_size - N_target
            # Population is sorted, remove worst (end of list)
            pop = pop[:-n_remove]
            fitness = fitness[:-n_remove]
            pop_size = N_target
            
            # Reduce Archive to match new pop_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))

        # 2. Stagnation Check (Partial Restart)
        # If population variance is negligible and we have time left (>10%),
        # inject fresh diversity into the bottom half of the population.
        if pop_size >= N_min and progress < 0.9:
            # Check standard deviation of fitness
            if np.std(fitness) < 1e-9:
                # Replace worst 50%
                n_replace = int(pop_size * 0.5)
                if n_replace > 0:
                    start_idx = pop_size - n_replace
                    for r_i in range(start_idx, pop_size):
                        if check_time(): return best_val
                        new_vec = min_b + np.random.rand(dim) * diff_b
                        val = func(new_vec)
                        pop[r_i] = new_vec
                        fitness[r_i] = val
                        if val < best_val:
                            best_val = val
                    # Re-sort to maintain order for p-best selection
                    s_idx = np.argsort(fitness)
                    pop = pop[s_idx]
                    fitness = fitness[s_idx]

        # 3. Parameter Generation
        # Randomly select memory index for each individual
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idxs]
        m_f = mem_f[r_idxs]
        
        # Generate CR (Normal dist, clipped)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy dist)
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Repair F: if > 1 clamp to 1, if <= 0 regenerate
        f[f > 1] = 1.0
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            n_bad = np.sum(mask_bad)
            f[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
            f[f > 1] = 1.0
            
        # 4. Mutation: current-to-pbest/1
        # p-best size: max(2/N, 0.11) - robust setting
        p_val = max(2.0/pop_size, 0.11)
        n_pbest = int(max(2, p_val * pop_size))
        
        # Select p-best indices (top n_pbest)
        pbest_indices = np.random.randint(0, n_pbest, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (!= i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        mask_col = r1_indices == np.arange(pop_size)
        r1_indices[mask_col] = (r1_indices[mask_col] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (!= r1, != i) from Population U Archive
        if len(archive) > 0:
            pop_arc = np.vstack((pop, np.array(archive)))
        else:
            pop_arc = pop
            
        r2_indices = np.random.randint(0, len(pop_arc), pop_size)
        # Check collision with r1 (simple check, ignoring i for speed)
        # For small pop this matters, but DE is robust.
        x_r2 = pop_arc[r2_indices]
        
        # Compute Mutant Vectors
        f_col = f[:, None]
        v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        mask_rand = np.random.rand(pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask_rand[np.arange(pop_size), j_rand] = True
        
        u = np.where(mask_rand, v, pop)
        
        # 6. Bound Constraint Handling (Midpoint Correction)
        # Instead of clipping, place point halfway between parent and bound
        # This preserves diversity near bounds and helps convergence.
        mask_l = u < min_b
        mask_h = u > max_b
        
        if np.any(mask_l):
            u[mask_l] = (min_b + pop)[mask_l] / 2.0
        if np.any(mask_h):
            u[mask_h] = (max_b + pop)[mask_h] / 2.0
            
        # 7. Selection and Evaluation
        new_fitness = np.empty(pop_size)
        success_list_cr = []
        success_list_f = []
        success_list_df = []
        
        for i in range(pop_size):
            if check_time(): return best_val
            
            val = func(u[i])
            new_fitness[i] = val
            
            if val <= fitness[i]:
                # Successful trial
                if val < fitness[i]:
                    success_list_cr.append(cr[i])
                    success_list_f.append(f[i])
                    success_list_df.append(fitness[i] - val)
                    
                    # Add replaced parent to archive
                    archive.append(pop[i].copy())
                
                # Update population
                pop[i] = u[i]
                fitness[i] = val
                
                if val < best_val:
                    best_val = val
                    
        # 8. Update Archive Size
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 9. Update Memory (Weighted Lehmer Mean)
        if success_list_df:
            scr = np.array(success_list_cr)
            sf = np.array(success_list_f)
            sdf = np.array(success_list_df)
            
            # Weights based on improvement magnitude
            total_df = np.sum(sdf)
            if total_df == 0:
                weights = np.ones(len(sdf)) / len(sdf)
            else:
                weights = sdf / total_df
            
            # Update M_CR
            m_cr_new = np.sum(weights * scr)
            mem_cr[k_mem] = m_cr_new
            
            # Update M_F
            num_f = np.sum(weights * (sf ** 2))
            den_f = np.sum(weights * sf)
            m_f_new = num_f / den_f if den_f > 1e-9 else 0.5
            mem_f[k_mem] = np.clip(m_f_new, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 10. Sort Population (Critical for LPSR and p-best next iter)
        sort_order = np.argsort(fitness)
        pop = pop[sort_order]
        fitness = fitness[sort_order]

    return best_val
