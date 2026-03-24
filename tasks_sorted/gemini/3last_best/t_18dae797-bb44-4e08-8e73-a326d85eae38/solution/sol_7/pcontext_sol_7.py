#An algorithm that implements **SHADE (Success-History based Adaptive Differential Evolution) with Midpoint-Target Bound Handling and Stagnation-Based Restarts**.
#
#### Reasoning for Improvements
#1.  **Discarding LPSR**: Previous attempts using Linear Population Size Reduction (LPSR) performed poorly (~40 vs ~1.1). LPSR can be detrimental when the exact computational cost is unknown or when the time limit is short, as it may reduce diversity too aggressively. This implementation maintains a **fixed, healthy population size** to sustain diversity.
#2.  **Adaptive Restarts**: Instead of a complex reduction schedule, the algorithm monitors population variance. If the population stagnates (standard deviation drops below a threshold), it **restarts** the population while preserving the global best solution (Elitism). This effectively utilizes the remaining time to explore new basins of attraction.
#3.  **Midpoint Bound Handling**: Rather than simple clipping (which biases search to boundaries), violated components are set to the **midpoint** between the parent and the bound `(parent + bound) / 2`. This preserves search capability near the edges.
#4.  **Optimized Vectorization**: The selection of `r2` from the union of Population and Archive is handled via masked indexing rather than expensive array stacking (`np.vstack`), improving iteration speed.
#
#### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using SHADE with Adaptive Restarts and 
    Midpoint Bound Handling.
    """
    
    # --- Time Management ---
    t_start = time.time()
    # Reserve a 2% buffer to ensure we return within the strict limit
    t_limit = max_time * 0.98

    def check_timeout():
        return (time.time() - t_start) >= t_limit

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Configuration
    # Population size: Fixed size based on dimension.
    # A range of [40, 200] is robust for most problems without LPSR.
    pop_size = int(max(40, min(200, 15 * dim)))
    
    # External Archive: Stores good solutions replaced by better ones.
    # Size = 2.0 * pop_size
    max_arc_size = int(2.0 * pop_size)
    archive = np.zeros((max_arc_size, dim))
    arc_count = 0
    
    # SHADE Memory Parameters (History size H=6)
    H = 6
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0

    # Global Best Tracking
    best_val = float('inf')
    best_sol = None 

    # --- Main Optimization Loop (Restarts) ---
    while not check_timeout():
        
        # --- Session Initialization ---
        # Generate initial population (Uniform Random)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject the global best solution into the new population
        if best_sol is not None:
            pop[0] = best_sol
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_timeout(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
        
        # Reset Archive and Memory for the new restart
        # This prevents bias from the previous local optimum
        arc_count = 0
        mem_cr.fill(0.5)
        mem_f.fill(0.5)
        k_mem = 0
        
        # --- Evolutionary Cycle ---
        while not check_timeout():
            
            # 1. Stagnation Check
            # If population variance is negligible, restart to explore elsewhere.
            if np.std(fitness) < 1e-6:
                break 
            
            # 2. Sort Population (Crucial for p-best selection)
            # We sort by fitness so best individuals are at indices [0, 1, ...]
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            # 3. Parameter Generation
            # Randomly select a memory index for each individual
            r_idxs = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idxs]
            m_f = mem_f[r_idxs]
            
            # Generate CR: Normal(mean=m_cr, std=0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            np.clip(cr, 0, 1, out=cr)
            
            # Generate F: Cauchy(loc=m_f, scale=0.1)
            # F = m_f + 0.1 * tan(pi * (rand - 0.5))
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Repair F
            f[f > 1] = 1.0
            # Resample non-positive F values
            while True:
                mask_bad = f <= 0
                if not np.any(mask_bad): break
                n_bad = np.sum(mask_bad)
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
                f[f > 1] = 1.0
                
            # 4. Mutation: current-to-pbest/1
            # Select p-best indices: Randomly from top p%
            # p is dynamic/random in [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            n_pbest = int(max(2, p_val * pop_size))
            
            pbest_indices = np.random.randint(0, n_pbest, pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (distinct from i)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            # Ensure r1 != i
            mask_col = r1_indices == np.arange(pop_size)
            r1_indices[mask_col] = (r1_indices[mask_col] + 1) % pop_size
            x_r1 = pop[r1_indices]
            
            # Select r2 (distinct from r1 and i, from Pop U Archive)
            # Efficient selection without vstacking
            r2_indices = np.random.randint(0, pop_size + arc_count, pop_size)
            x_r2 = np.empty((pop_size, dim))
            
            mask_from_pop = r2_indices < pop_size
            mask_from_arc = ~mask_from_pop
            
            if np.any(mask_from_pop):
                x_r2[mask_from_pop] = pop[r2_indices[mask_from_pop]]
            if np.any(mask_from_arc):
                # Map index to archive range
                arc_idxs = r2_indices[mask_from_arc] - pop_size
                x_r2[mask_from_arc] = archive[arc_idxs]
            
            # Compute Mutant Vector V
            f_col = f[:, None]
            v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            # Ensure at least one parameter is changed (j_rand)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cr = np.random.rand(pop_size, dim) < cr[:, None]
            mask_cr[np.arange(pop_size), j_rand] = True
            
            u = np.where(mask_cr, v, pop)
            
            # 6. Bound Constraint Handling (Midpoint)
            # If a variable violates a bound, place it halfway between the parent and the bound.
            # This preserves diversity better than clipping.
            mask_l = u < min_b
            mask_u = u > max_b
            
            if np.any(mask_l):
                cols = np.where(mask_l)[1]
                u[mask_l] = (pop[mask_l] + min_b[cols]) / 2.0
            if np.any(mask_u):
                cols = np.where(mask_u)[1]
                u[mask_u] = (pop[mask_u] + max_b[cols]) / 2.0
                
            # 7. Selection and Evaluation
            new_fitness = np.empty(pop_size)
            success_mask = np.zeros(pop_size, dtype=bool)
            diff_fitness = np.zeros(pop_size)
            
            for i in range(pop_size):
                if check_timeout(): return best_val
                
                val = func(u[i])
                new_fitness[i] = val
                
                if val <= fitness[i]:
                    success_mask[i] = True
                    diff_fitness[i] = fitness[i] - val
                    
                    if val < best_val:
                        best_val = val
                        best_sol = u[i].copy()
            
            # 8. Update Archive, Population, Memory
            success_idx = np.where(success_mask)[0]
            if len(success_idx) > 0:
                # Add parents to Archive
                for idx in success_idx:
                    if arc_count < max_arc_size:
                        archive[arc_count] = pop[idx]
                        arc_count += 1
                    else:
                        rand_k = np.random.randint(0, max_arc_size)
                        archive[rand_k] = pop[idx]
                
                # Update Population
                pop[success_idx] = u[success_idx]
                fitness[success_idx] = new_fitness[success_idx]
                
                # Update Memory (only using strict improvements)
                better_idx = success_idx[diff_fitness[success_idx] > 0]
                if len(better_idx) > 0:
                    df = diff_fitness[better_idx]
                    total_df = np.sum(df)
                    weights = df / total_df
                    
                    # Update M_CR
                    scr = cr[better_idx]
                    m_cr_new = np.sum(weights * scr)
                    mem_cr[k_mem] = m_cr_new
                    
                    # Update M_F (Weighted Lehmer Mean)
                    sf = f[better_idx]
                    num = np.sum(weights * (sf**2))
                    den = np.sum(weights * sf)
                    m_f_new = num / den if den > 1e-9 else 0.5
                    mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                    
                    k_mem = (k_mem + 1) % H

    return best_val
