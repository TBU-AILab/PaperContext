#Here is an improved algorithm. It implements **L-SHADE with Linear Population Size Reduction (LPSR) and Nelder-Mead Polish**.
#
#### Key Improvements
#1.  **Linear Population Size Reduction (LPSR):** Instead of fixed restart sizes, the population size shrinks linearly from a high number (exploration) to a small number (exploitation) based on elapsed time. This forces convergence and increases efficiency as the search progresses, mimicking the behavior of state-of-the-art CEC competition winners.
#2.  **Nelder-Mead Simplex Polish:** Replaced the coordinate-descent (Hooke-Jeeves) with the Nelder-Mead algorithm. Nelder-Mead can adapt its shape to the local landscape (stretching along valleys), making it significantly better at refining the final solution in complex, non-separable functions.
#3.  **Adaptive Archive:** The external archive size now scales dynamically with the population size, ensuring the diversity preservation mechanism remains balanced throughout the reduction process.
#4.  **Robust Bound Handling:** Implements a rigorous reflective bound constraint to prevent the optimizer from getting stuck at the edges of the search space.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Linear Population Size Reduction (LPSR)
    followed by a Nelder-Mead Simplex polish.
    """
    
    # --- Configuration & Constants ---
    start_time = time.time()
    
    # Time allocation
    # Reserve small portion for final Nelder-Mead polish (5% or min 0.5s)
    polish_budget = max(0.5, max_time * 0.05)
    shade_time = max_time - polish_budget
    
    # Pre-process bounds
    bounds = np.array(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    bound_diff = max_b - min_b
    
    # L-SHADE Parameters
    # Population Size Schedule (LPSR)
    # Initial size: ample for exploration (e.g., 18*D)
    # Final size: minimal for convergence (4)
    r_N_init = 18
    p_init = int(r_N_init * dim)
    p_init = np.clip(p_init, 20, 500) # Safety clips
    p_min = 4
    
    pop_size = p_init
    current_pop_size = p_init
    
    # Memory Parameters
    H = 6 # Memory size
    arc_rate = 2.6 # Archive size factor
    
    # Adaptation Memories
    memory_sf = np.full(H, 0.5)
    memory_scr = np.full(H, 0.5)
    mem_k = 0
    
    # --- Initialization ---
    # Random initial population
    pop = min_b + np.random.rand(pop_size, dim) * bound_diff
    fitness = np.array([float('inf')] * pop_size)
    
    # Archive
    archive = []
    
    # Global Best Tracking
    global_best_x = None
    global_best_f = float('inf')
    
    # --- Helper: Boundary Constraint (Reflective) ---
    def check_bounds(candidates):
        # Reflect lower bounds
        mask_l = candidates < min_b
        if np.any(mask_l):
            candidates[mask_l] = 2 * min_b[mask_l] - candidates[mask_l]
            # If still out, clip
            mask_l_2 = candidates < min_b
            candidates[mask_l_2] = min_b[mask_l_2]
            
        # Reflect upper bounds
        mask_u = candidates > max_b
        if np.any(mask_u):
            candidates[mask_u] = 2 * max_b[mask_u] - candidates[mask_u]
            # If still out, clip
            mask_u_2 = candidates > max_b
            candidates[mask_u_2] = max_b[mask_u_2]
            
        return candidates

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if (time.time() - start_time) > shade_time: break
        val = func(pop[i])
        fitness[i] = val
        if val < global_best_f:
            global_best_f = val
            global_best_x = pop[i].copy()

    # --- Main L-SHADE-LPSR Loop ---
    while (time.time() - start_time) < shade_time:
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate max possible generations roughly or use time ratio
        # Since we use time, we map Time -> PopSize
        elapsed = time.time() - start_time
        time_ratio = min(1.0, elapsed / shade_time)
        
        # Target size based on time progress
        target_size = int(round(p_init + (p_min - p_init) * time_ratio))
        target_size = max(p_min, target_size)
        
        # If current pop is too big, reduce it by removing worst individuals
        if current_pop_size > target_size:
            # Sort by fitness
            sort_idx = np.argsort(fitness)
            fitness = fitness[sort_idx]
            pop = pop[sort_idx]
            
            # Truncate
            remove_count = current_pop_size - target_size
            current_pop_size = target_size
            
            pop = pop[:current_pop_size]
            fitness = fitness[:current_pop_size]
            
            # Archive size also shrinks dynamically
            arc_size_max = int(current_pop_size * arc_rate)
            if len(archive) > arc_size_max:
                del archive[arc_size_max:]

        # 2. Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Update best
        if fitness[0] < global_best_f:
            global_best_f = fitness[0]
            global_best_x = pop[0].copy()

        # 3. Parameter Generation
        # Random memory indices
        r_idx = np.random.randint(0, H, current_pop_size)
        m_sf = memory_sf[r_idx]
        m_scr = memory_scr[r_idx]
        
        # Generate F (Cauchy)
        # If F <= 0, retry. If F > 1, clip to 1.
        sf = m_sf + 0.1 * np.random.standard_cauchy(current_pop_size)
        while np.any(sf <= 0):
            mask_bad = sf <= 0
            sf[mask_bad] = m_sf[mask_bad] + 0.1 * np.random.standard_cauchy(np.sum(mask_bad))
        sf = np.clip(sf, 0, 1.0)
        
        # Generate CR (Normal)
        scr = np.random.normal(m_scr, 0.1)
        scr = np.clip(scr, 0.0, 1.0)
        
        # 4. Mutation: current-to-pbest/1
        # p-best selection (top p%)
        p_val = np.random.uniform(2/current_pop_size, 0.2, size=current_pop_size)
        p_indices = (p_val * current_pop_size).astype(int)
        p_indices = np.maximum(p_indices, 1)
        
        # Vectorized index selection
        r_best_ids = [np.random.randint(0, pi) for pi in p_indices]
        x_pbest = pop[r_best_ids]
        
        # r1 != i
        r1_ids = np.random.randint(0, current_pop_size, current_pop_size)
        # Handle collision r1 == i
        col_r1 = (r1_ids == np.arange(current_pop_size))
        r1_ids[col_r1] = (r1_ids[col_r1] + 1) % current_pop_size
        x_r1 = pop[r1_ids]
        
        # r2 != i, != r1, from Pop U Archive
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
            
        len_union = len(union_pop)
        r2_ids = np.random.randint(0, len_union, current_pop_size)
        # Simple check for r2==r1 or r2==i is skipped for speed in vectorized form, 
        # DE is robust enough to noise.
        x_r2 = union_pop[r2_ids]
        
        # Compute Mutation Vectors V
        sf_col = sf[:, None]
        v = pop + sf_col * (x_pbest - pop) + sf_col * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, current_pop_size)
        mask_rand = np.random.rand(current_pop_size, dim) < scr[:, None]
        mask_j = np.zeros((current_pop_size, dim), dtype=bool)
        mask_j[np.arange(current_pop_size), j_rand] = True
        final_mask = np.logical_or(mask_rand, mask_j)
        
        u = np.where(final_mask, v, pop)
        u = check_bounds(u)
        
        # 6. Selection & Archive Update
        # Must evaluate individually
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        success_sf = []
        success_scr = []
        diffs = []
        
        for k in range(current_pop_size):
            if (time.time() - start_time) > shade_time: break
            
            f_new = func(u[k])
            
            if f_new <= fitness[k]:
                new_pop[k] = u[k]
                new_fitness[k] = f_new
                
                if f_new < fitness[k]:
                    # Keep successful parameters
                    success_sf.append(sf[k])
                    success_scr.append(scr[k])
                    diffs.append(fitness[k] - f_new)
                    
                    # Add replaced parent to archive
                    archive.append(pop[k].copy())
                    
                if f_new < global_best_f:
                    global_best_f = f_new
                    global_best_x = u[k].copy()
        
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Limit
        arc_max = int(current_pop_size * arc_rate)
        while len(archive) > arc_max:
            # Remove random
            idx_rm = np.random.randint(0, len(archive))
            archive.pop(idx_rm)
            
        # 7. Update Memory (Weighted Lehmer Mean)
        if len(success_sf) > 0:
            s_sf = np.array(success_sf)
            s_scr = np.array(success_scr)
            w_diff = np.array(diffs)
            
            total_diff = np.sum(w_diff)
            if total_diff == 0:
                weights = np.ones(len(w_diff)) / len(w_diff)
            else:
                weights = w_diff / total_diff
                
            # Mean F (Lehmer)
            mean_sf = np.sum(weights * (s_sf ** 2)) / np.sum(weights * s_sf)
            # Mean CR (Arithmetic - Weighted)
            mean_scr = np.sum(weights * s_scr)
            
            memory_sf[mem_k] = np.clip(mean_sf, 0, 1)
            memory_scr[mem_k] = np.clip(mean_scr, 0, 1)
            mem_k = (mem_k + 1) % H

    # --- Phase 2: Nelder-Mead Simplex Polish ---
    # Apply to the best found solution to drain remaining time
    # Nelder-Mead is better than Hooke-Jeeves for navigating non-separable valleys.
    
    if global_best_x is None:
        return float('inf')

    # Initial Simplex
    # x0 = global best. Other N points generated by perturbing x0
    nm_dim = dim
    simplex = np.zeros((nm_dim + 1, nm_dim))
    simplex_f = np.zeros(nm_dim + 1)
    
    simplex[0] = global_best_x
    simplex_f[0] = global_best_f
    
    # Step size for simplex creation: 5% of domain or 0.05
    step = np.minimum(0.05, (max_b - min_b) * 0.05)
    
    for i in range(nm_dim):
        point = global_best_x.copy()
        # Perturb
        if point[i] + step[i] <= max_b[i]:
            point[i] += step[i]
        else:
            point[i] -= step[i]
            
        simplex[i+1] = point
        # Just in case of crash, check time
        if (time.time() - start_time) >= max_time: 
            return global_best_f
        simplex_f[i+1] = func(point)

    # Nelder-Mead Parameters
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5    # Contraction
    sigma = 0.5  # Shrink
    
    while (time.time() - start_time) < max_time:
        # Sort
        order = np.argsort(simplex_f)
        simplex = simplex[order]
        simplex_f = simplex_f[order]
        
        best_f = simplex_f[0]
        worst_f = simplex_f[-1]
        
        # Update Global
        if best_f < global_best_f:
            global_best_f = best_f
            
        # Centroid of the best n points
        x0 = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = x0 + alpha * (x0 - simplex[-1])
        xr = np.clip(xr, min_b, max_b) # Simple clip for polish
        fr = func(xr)
        
        if simplex_f[0] <= fr < simplex_f[-2]:
            simplex[-1] = xr
            simplex_f[-1] = fr
        elif fr < simplex_f[0]:
            # Expansion
            xe = x0 + gamma * (xr - x0)
            xe = np.clip(xe, min_b, max_b)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                simplex_f[-1] = fe
            else:
                simplex[-1] = xr
                simplex_f[-1] = fr
        else:
            # Contraction
            xc = x0 + rho * (simplex[-1] - x0)
            xc = np.clip(xc, min_b, max_b)
            fc = func(xc)
            if fc < simplex_f[-1]:
                simplex[-1] = xc
                simplex_f[-1] = fc
            else:
                # Shrink
                for i in range(1, nm_dim + 1):
                    if (time.time() - start_time) >= max_time: break
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], min_b, max_b)
                    simplex_f[i] = func(simplex[i])
                    
    return global_best_f
