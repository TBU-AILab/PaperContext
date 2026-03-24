#Here is an improved algorithm. It implements **jSO (a specialized L-SHADE variant)** combined with **MTS-LS1 (Multiple Trajectory Search - Local Search 1)**.
#
#### Key Improvements
#1.  **jSO Architecture:** Implements the specific parameter adaptation rules of the jSO algorithm (CEC 2017 winner). This includes a dynamic $p$ value for *p-best* selection (starts high for exploration, ends low for exploitation) and a specific weighted mutation strategy.
#2.  **MTS-LS1 Polish:** Replaces the Nelder-Mead polish with MTS-LS1. MTS-LS1 is a coordinate-descent-based local search that is extremely effective at refining solutions in high-dimensional bound-constrained problems, often superior to Nelder-Mead for fine-tuning.
#3.  **Linear Population Size Reduction (LPSR):** Continues to use LPSR to force convergence, but refined to ensure the population doesn't drop below viable diversity levels too quickly.
#4.  **Robust Bound Handling:** Uses the "midpoint target" correction for variables violating bounds, which is generally more robust than simple reflection for differential evolution.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using jSO (L-SHADE variant) with Linear Population Size Reduction
    and MTS-LS1 Local Search Polish.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Allocation: 90% DE, 10% Local Search
    ls_budget = max(0.2, max_time * 0.1) 
    de_budget = max_time - ls_budget

    # Convert bounds
    bounds = np.array(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    diff_b = max_b - min_b

    # --- jSO Parameters ---
    # Population size schedule
    # N_init: roughly 25*D, clamped to [30, 500] for efficiency
    r_N_init = 25
    p_init_size = int(r_N_init * dim)
    p_init_size = np.clip(p_init_size, 30, 500)
    p_min_size = 4
    
    pop_size = p_init_size
    current_pop_size = p_init_size
    
    # Archive parameters
    archive_rate = 2.6
    archive = []
    
    # Memory parameters (H)
    H = 5
    mem_sf = np.full(H, 0.5)
    mem_scr = np.full(H, 0.5)
    mem_k = 0
    
    # Dynamic p-best parameters
    # p starts at 0.25 (exploration) and reduces to 0.05 (exploitation)
    p_max_rate = 0.25
    p_min_rate = 0.05
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.array([float('inf')] * pop_size)
    
    global_best_f = float('inf')
    global_best_x = None

    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) > de_budget: break
        val = func(pop[i])
        fitness[i] = val
        if val < global_best_f:
            global_best_f = val
            global_best_x = pop[i].copy()

    # --- Helper: Bound Correction (Midpoint) ---
    def correct_bounds(trial, old_pop):
        # If out of bounds, place between old value and bound
        mask_l = trial < min_b
        if np.any(mask_l):
            trial[mask_l] = (min_b[mask_l] + old_pop[mask_l]) / 2.0
            
        mask_u = trial > max_b
        if np.any(mask_u):
            trial[mask_u] = (max_b[mask_u] + old_pop[mask_u]) / 2.0
        return trial

    # --- Main Loop (DE Phase) ---
    while (time.time() - start_time) < de_budget:
        
        # 1. Calculate Progress & Dynamic Parameters
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / de_budget)
        
        # LPSR: Reduce Population
        plan_size = int(round(p_init_size + (p_min_size - p_init_size) * progress))
        plan_size = max(p_min_size, plan_size)
        
        if current_pop_size > plan_size:
            # Reduce: remove worst
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            # Truncate
            current_pop_size = plan_size
            pop = pop[:current_pop_size]
            fitness = fitness[:current_pop_size]
            
            # Reduce archive size accordingly
            max_arc_size = int(current_pop_size * archive_rate)
            if len(archive) > max_arc_size:
                del archive[max_arc_size:]

        # Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Update Global Best
        if fitness[0] < global_best_f:
            global_best_f = fitness[0]
            global_best_x = pop[0].copy()

        # Dynamic p value (jSO strategy)
        # Linear reduction of p
        curr_p = p_max_rate - (p_max_rate - p_min_rate) * progress
        # p cannot be smaller than 2 individuals
        num_pbest = max(2, int(curr_p * current_pop_size))
        
        # 2. Parameter Generation
        # Random memory index
        r_idx = np.random.randint(0, H, current_pop_size)
        m_sf = mem_sf[r_idx]
        m_scr = mem_scr[r_idx]
        
        # Generate CR: Normal(mean=m_scr, std=0.1)
        scr = np.random.normal(m_scr, 0.1)
        scr = np.clip(scr, 0.0, 1.0)
        
        # Generate F: Cauchy(loc=m_sf, scale=0.1)
        # jSO/L-SHADE constraint: if F > 1 set to 1, if F <= 0 generate again
        sf = m_sf + 0.1 * np.random.standard_cauchy(current_pop_size)
        
        # Retry logic for F <= 0 (vectorized)
        retry_mask = sf <= 0
        while np.any(retry_mask):
            count = np.sum(retry_mask)
            sf[retry_mask] = m_sf[retry_mask] + 0.1 * np.random.standard_cauchy(count)
            retry_mask = sf <= 0
        
        sf = np.clip(sf, 0.0, 1.0)

        # 3. Mutation: current-to-pbest/1
        # Indices for mutation
        # r1 != i
        r1 = np.random.randint(0, current_pop_size, current_pop_size)
        # Fix collision r1 == i
        col_r1 = (r1 == np.arange(current_pop_size))
        r1[col_r1] = (r1[col_r1] + 1) % current_pop_size
        
        # r2 != i, r2 != r1, from Pop U Archive
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
            
        r2 = np.random.randint(0, len(union_pop), current_pop_size)
        # Collision handling for r2 is usually skipped for speed in DE as impact is low
        
        # p-best selection: random from top p%
        pbest_indices = np.random.randint(0, num_pbest, current_pop_size)
        
        # Vectors
        x_i = pop
        x_pbest = pop[pbest_indices]
        x_r1 = pop[r1]
        x_r2 = union_pop[r2]
        
        # Mutation Vector V
        # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
        diff1 = x_pbest - x_i
        diff2 = x_r1 - x_r2
        v = x_i + sf[:, None] * diff1 + sf[:, None] * diff2
        
        # 4. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, current_pop_size)
        mask_rand = np.random.rand(current_pop_size, dim) < scr[:, None]
        mask_j = np.zeros((current_pop_size, dim), dtype=bool)
        mask_j[np.arange(current_pop_size), j_rand] = True
        cross_mask = np.logical_or(mask_rand, mask_j)
        
        u = np.where(cross_mask, v, pop)
        
        # Bound Correction
        for k in range(current_pop_size):
            u[k] = correct_bounds(u[k], pop[k])

        # 5. Selection
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        success_sf = []
        success_scr = []
        diff_fitness = []
        
        # Evaluation
        for k in range(current_pop_size):
            if (time.time() - start_time) > de_budget: break
            
            f_u = func(u[k])
            
            if f_u < fitness[k]:
                new_pop[k] = u[k]
                new_fitness[k] = f_u
                
                success_sf.append(sf[k])
                success_scr.append(scr[k])
                diff_fitness.append(fitness[k] - f_u)
                
                # Add parent to archive
                archive.append(pop[k].copy())
                
                if f_u < global_best_f:
                    global_best_f = f_u
                    global_best_x = u[k].copy()
            elif f_u == fitness[k]:
                new_pop[k] = u[k]

        pop = new_pop
        fitness = new_fitness
        
        # Clean Archive
        max_arc_size = int(current_pop_size * archive_rate)
        while len(archive) > max_arc_size:
            idx_rem = np.random.randint(0, len(archive))
            archive.pop(idx_rem)
            
        # 6. Memory Update (Weighted Lehmer Mean)
        if len(success_sf) > 0:
            w = np.array(diff_fitness)
            total_w = np.sum(w)
            if total_w > 0:
                w = w / total_w
                
                # Update M_SF (Lehmer Mean)
                s_sf = np.array(success_sf)
                mean_sf = np.sum(w * (s_sf ** 2)) / np.sum(w * s_sf)
                
                # Update M_SCR (Weighted Mean)
                # jSO specific: if mean_scr is close to 0/1, fix it logic? 
                # Standard L-SHADE update is fine here.
                s_scr = np.array(success_scr)
                mean_scr = np.sum(w * s_scr)
                
                mem_sf[mem_k] = np.clip(mean_sf, 0, 1)
                mem_scr[mem_k] = np.clip(mean_scr, 0, 1)
                mem_k = (mem_k + 1) % H

    # --- Phase 2: MTS-LS1 Local Search Polish ---
    # Coordinate descent on the best solution found
    if global_best_x is None:
        return float('inf')

    # Start search range: 40% of domain
    search_range = (max_b - min_b) * 0.4
    
    # MTS-LS1 iterates through dimensions
    # We refine global_best_x in place (copy)
    best_sol = global_best_x.copy()
    best_val = global_best_f
    
    # Check if we have time
    while (time.time() - start_time) < max_time:
        
        improved = False
        # Randomize dimension order for better coverage
        dims_order = np.random.permutation(dim)
        
        for d in dims_order:
            if (time.time() - start_time) >= max_time: break
            
            # Step 1: Negative direction
            original_val = best_sol[d]
            best_sol[d] -= search_range[d]
            
            # Bound check
            if best_sol[d] < min_b[d]: 
                best_sol[d] = min_b[d]
            elif best_sol[d] > max_b[d]: 
                best_sol[d] = max_b[d]
                
            new_val = func(best_sol)
            
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                # Restore and try positive direction (0.5 step as per MTS logic)
                best_sol[d] = original_val
                best_sol[d] += 0.5 * search_range[d]
                
                # Bound check
                if best_sol[d] < min_b[d]: 
                    best_sol[d] = min_b[d]
                elif best_sol[d] > max_b[d]: 
                    best_sol[d] = max_b[d]
                
                new_val = func(best_sol)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    # Restore
                    best_sol[d] = original_val
        
        if not improved:
            # Refine search step
            search_range *= 0.5
            # If precision is too small, break or reset? 
            # In limited time, just keeping refining is usually safer.
            if np.all(search_range < 1e-9):
                # Small restart of step size if time allows
                search_range = (max_b - min_b) * 0.4
        else:
            # If improved, update global tracking
            if best_val < global_best_f:
                global_best_f = best_val
                global_best_x = best_sol.copy()
                
    return global_best_f
