import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    SHADE (Success-History based Adaptive Differential Evolution) 
    with Restarts and Local Polish.
    
    Improvements over standard DE/L-SHADE:
    1. Restarts: Prevents getting stuck in local optima by resetting the population 
       when convergence is detected.
    2. Local Polish: A lightweight coordinate descent is applied to the best solution 
       before restarting to refine precision.
    3. SHADE Adaptation: Uses history-based memory for F and CR parameters, 
       adapting to the function landscape dynamically.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Balanced for speed and diversity.
    # We clip it to ensure it doesn't get too large for high dimensions within strict time.
    pop_size = int(np.clip(10 * dim, 30, 80))
    
    # SHADE Memory Parameters
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive Parameters
    arc_rate = 2.0
    max_arc_size = int(pop_size * arc_rate)
    archive = np.zeros((max_arc_size, dim))
    n_arc = 0
    
    # Bounds Processing
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    x = min_b + np.random.rand(pop_size, dim) * diff_b
    fit = np.full(pop_size, np.inf)
    
    best_val = np.inf
    best_sol = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        val = func(x[i])
        fit[i] = val
        if val < best_val:
            best_val = val
            best_sol = x[i].copy()
            
    # --- Helper: Local Search (Coordinate Descent) ---
    def local_search(curr_sol, curr_val):
        # Creates a copy to not mess with original unless better
        sol = curr_sol.copy()
        val = curr_val
        
        # Initial step size (10% of domain)
        step_size = (max_b - min_b) * 0.1
        
        # Limit effort: 2 passes or timeout
        max_ls_evals = dim * 2 
        eval_count = 0
        
        improved = True
        while improved and eval_count < max_ls_evals:
            improved = False
            for d in range(dim):
                if time.time() - start_time >= max_time:
                    return sol, val
                
                # Try negative direction
                temp_sol = sol.copy()
                temp_sol[d] = np.clip(sol[d] - step_size[d], min_b[d], max_b[d])
                temp_val = func(temp_sol)
                eval_count += 1
                
                if temp_val < val:
                    val = temp_val
                    sol = temp_sol
                    improved = True
                else:
                    # Try positive direction (half step to refine)
                    temp_sol[d] = np.clip(sol[d] + 0.5 * step_size[d], min_b[d], max_b[d])
                    temp_val = func(temp_sol)
                    eval_count += 1
                    
                    if temp_val < val:
                        val = temp_val
                        sol = temp_sol
                        improved = True
                
                if eval_count >= max_ls_evals:
                    break
            
            # Decay step size for finer search next pass
            step_size *= 0.5
            
        return sol, val

    # --- Main Optimization Loop ---
    while True:
        # Time check
        if time.time() - start_time >= max_time:
            return best_val
        
        # 1. Parameter Generation (SHADE)
        # Randomly select memory slots
        r_idx = np.random.randint(0, H, pop_size)
        m_f = mem_f[r_idx]
        m_cr = mem_cr[r_idx]
        
        # Generate F (Cauchy) and CR (Normal)
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        cr = np.random.normal(m_cr, 0.1, pop_size)
        
        # Repair F: if <= 0, retry; if > 1, clip.
        # Vectorized retry for efficiency
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            # Regenerate bad ones
            count = np.sum(mask_bad)
            f[mask_bad] = m_f[mask_bad] + 0.1 * np.random.standard_cauchy(count)
        f = np.clip(f, 0, 1)
        cr = np.clip(cr, 0, 1)
        
        # 2. Mutation: current-to-pbest/1
        # Sort population to identify p-best
        sorted_indices = np.argsort(fit)
        
        # Select p uniformly from [2/NP, 0.2]
        p_val = np.random.uniform(2/pop_size, 0.2)
        n_pbest = int(pop_size * p_val)
        n_pbest = max(2, n_pbest)
        
        top_indices = sorted_indices[:n_pbest]
        pbest_indices = np.random.choice(top_indices, pop_size)
        x_pbest = x[pbest_indices]
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Simple heuristic to reduce collisions
        for _ in range(2):
            col = r1_indices == np.arange(pop_size)
            if not np.any(col): break
            r1_indices[col] = np.random.randint(0, pop_size, np.sum(col))
        x_r1 = x[r1_indices]
        
        # Select r2 (distinct from i and r1, from Union(Pop, Archive))
        n_total = pop_size + n_arc
        r2_indices = np.random.randint(0, n_total, pop_size)
        for _ in range(2):
            col = (r2_indices == np.arange(pop_size)) | (r2_indices == r1_indices)
            if not np.any(col): break
            r2_indices[col] = np.random.randint(0, n_total, np.sum(col))
            
        # Build x_r2
        x_r2 = np.zeros((pop_size, dim))
        mask_pop = r2_indices < pop_size
        mask_arc = ~mask_pop
        
        x_r2[mask_pop] = x[r2_indices[mask_pop]]
        if np.any(mask_arc):
            arc_idx = r2_indices[mask_arc] - pop_size
            x_r2[mask_arc] = archive[arc_idx]
            
        # Mutation Vector
        # v = x + F*(pbest - x) + F*(r1 - r2)
        v = x + f[:, None] * (x_pbest - x) + f[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        # Generate mask
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        # Ensure at least one dimension
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        u = np.where(mask_cross, v, x)
        u = np.clip(u, min_b, max_b)
        
        # 4. Selection and Memory Update
        success_f = []
        success_cr = []
        diff_fitness = []
        
        new_x = x.copy()
        new_fit = fit.copy()
        
        # Sequential evaluation to respect time limit strictly
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            val_u = func(u[i])
            
            if val_u <= fit[i]:
                # Update Archive
                if n_arc < max_arc_size:
                    archive[n_arc] = x[i].copy()
                    n_arc += 1
                else:
                    rand_idx = np.random.randint(0, max_arc_size)
                    archive[rand_idx] = x[i].copy()
                
                # Update Success Lists
                if val_u < fit[i]:
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_fitness.append(fit[i] - val_u)
                
                new_x[i] = u[i]
                new_fit[i] = val_u
                
                if val_u < best_val:
                    best_val = val_u
                    best_sol = u[i].copy()
        
        x = new_x
        fit = new_fit
        
        # Update Memory
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            s_df = np.array(diff_fitness)
            
            total_diff = np.sum(s_df)
            if total_diff > 0:
                # Weighted Lehmer Mean
                weights = s_df / total_diff
                mean_f = np.sum(weights * (s_f ** 2)) / np.sum(weights * s_f)
                mean_cr = np.sum(weights * s_cr)
                
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                k_mem = (k_mem + 1) % H
                
        # 5. Convergence Check & Restart
        # Check if population has converged
        fit_std = np.std(fit)
        
        if fit_std < 1e-6:
            # -- Phase 1: Local Polish --
            # Squeeze best solution
            if time.time() - start_time < max_time:
                p_sol, p_val = local_search(best_sol, best_val)
                if p_val < best_val:
                    best_val = p_val
                    best_sol = p_sol.copy()
            
            # -- Phase 2: Restart --
            # Reset Population
            x = min_b + np.random.rand(pop_size, dim) * diff_b
            # Elitism
            x[0] = best_sol.copy()
            fit = np.full(pop_size, np.inf)
            fit[0] = best_val
            
            # Evaluate new population (skip index 0)
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(x[i])
                fit[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = x[i].copy()
            
            # Reset Internals
            n_arc = 0
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            
    return best_val
