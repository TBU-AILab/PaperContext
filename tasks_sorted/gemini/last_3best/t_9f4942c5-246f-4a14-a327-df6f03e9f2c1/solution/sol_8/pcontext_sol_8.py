#The following algorithm implements **SHADE with Midpoint-Target Bound Handling and Restart-Polish Strategy**.
#
#**Key Improvements:**
#1.  **SHADE Adaptation**: Utilizes historical memory ($M_F, M_{CR}$) to adapt mutation factor $F$ and crossover rate $CR$, using a Cauchy distribution for $F$ to maintain exploration potential.
#2.  **Vectorized Operations**: Mutations, crossovers, and parameter generation are fully vectorized using NumPy to minimize interpreter overhead.
#3.  **Midpoint Bound Correction**: Instead of simple clipping (which biases the search to the edges), particles violating bounds are reset to the midpoint between their previous valid position and the bound. This preserves diversity near boundaries.
#4.  **Archive Strategy**: Maintains an external archive of successful past solutions to ensure diversity in the difference vector generation.
#5.  **Restart with Local Polish**: Monitors population variance. Upon convergence, it executes a lightweight coordinate descent (Local Polish) on the global best solution to refine precision, then triggers a soft restart (keeping the best solution) to escape local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Adaptive based on dimension. 
    # Capped at 100 to ensure rapid generations, min 20 for statistical validity.
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # SHADE Memory parameters
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive parameters
    archive_size = int(pop_size * 2.0)
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, np.inf)
    
    best_val = np.inf
    best_sol = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
            
    # --- Helper: Local Polish (Coordinate Descent) ---
    def local_polish(curr_sol, curr_val):
        sol = curr_sol.copy()
        val = curr_val
        # Initial step size relative to domain
        step = diff_b * 0.05 
        # Budget for local search to avoid hogging time
        ls_budget = dim * 10
        evals = 0
        
        improved = True
        while improved and evals < ls_budget:
            improved = False
            # Randomize dimension order
            idxs = np.random.permutation(dim)
            
            for d in idxs:
                if time.time() - start_time >= max_time:
                    return sol, val
                
                origin = sol[d]
                
                # Check Positive Step
                sol[d] = np.clip(origin + step[d], min_b[d], max_b[d])
                v_new = func(sol)
                evals += 1
                
                if v_new < val:
                    val = v_new
                    improved = True
                    continue
                
                # Check Negative Step (if positive failed)
                sol[d] = np.clip(origin - step[d], min_b[d], max_b[d])
                v_new = func(sol)
                evals += 1
                
                if v_new < val:
                    val = v_new
                    improved = True
                else:
                    # Revert if no improvement
                    sol[d] = origin
            
            # Decay step size
            step *= 0.5
            
        return sol, val

    # --- Main Loop ---
    while True:
        if time.time() - start_time >= max_time:
            return best_val
            
        # 1. Parameter Generation
        # Select memory index uniformly
        r_idx = np.random.randint(0, H, pop_size)
        m_f = mem_f[r_idx]
        m_cr = mem_cr[r_idx]
        
        # Generate F (Cauchy) and CR (Normal)
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        # Repair F: retry if <= 0
        retry_mask = f <= 0
        while np.any(retry_mask):
            n_retry = np.sum(retry_mask)
            f[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(n_retry)
            retry_mask = f <= 0
        f = np.minimum(f, 1.0)
        
        cr = np.random.normal(m_cr, 0.1, pop_size)
        cr = np.clip(cr, 0.0, 1.0)
        
        # 2. Mutation: current-to-pbest/1
        # Sort population by fitness
        sort_idx = np.argsort(fitness)
        
        # Select pbest (top p%, where p in [2/N, 0.2])
        p_val = np.random.uniform(2.0/pop_size, 0.2, pop_size)
        n_pbest = (p_val * pop_size).astype(int)
        n_pbest = np.maximum(n_pbest, 2)
        
        # Vectorized pbest selection
        rand_p = (np.random.rand(pop_size) * n_pbest).astype(int)
        pbest_ptr = sort_idx[rand_p]
        x_pbest = population[pbest_ptr]
        
        # Select r1 (distinct from i)
        r1_ptr = np.random.randint(0, pop_size, pop_size)
        # Handle collision with i
        mask_col = r1_ptr == np.arange(pop_size)
        if np.any(mask_col):
            r1_ptr[mask_col] = (r1_ptr[mask_col] + 1) % pop_size
        x_r1 = population[r1_ptr]
        
        # Select r2 (distinct from i, r1; from Union(Pop, Archive))
        pool_size = pop_size + n_archive
        r2_ptr = np.random.randint(0, pool_size, pop_size)
        # Handle collision (simple retry loop)
        for _ in range(2):
            mask_c = (r2_ptr == np.arange(pop_size)) | (r2_ptr == r1_ptr)
            if not np.any(mask_c): break
            r2_ptr[mask_c] = np.random.randint(0, pool_size, np.sum(mask_c))
            
        x_r2 = np.zeros((pop_size, dim))
        mask_pop = r2_ptr < pop_size
        x_r2[mask_pop] = population[r2_ptr[mask_pop]]
        mask_arc = ~mask_pop
        if np.any(mask_arc):
            idx_arc = r2_ptr[mask_arc] - pop_size
            x_r2[mask_arc] = archive[idx_arc]
            
        # Compute Mutant Vectors
        # v = x + F*(pbest - x) + F*(r1 - r2)
        mutant = population + f[:, None] * (x_pbest - population) + f[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        mask_rand = np.random.rand(pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask_rand[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_rand, mutant, population)
        
        # 4. Bound Handling (Midpoint Correction)
        # If trial value is out of bounds, set it to the midpoint 
        # between the parent and the bound. Better than clipping.
        mask_l = trial < min_b
        trial[mask_l] = (min_b[mask_l] + population[mask_l]) * 0.5
        mask_h = trial > max_b
        trial[mask_h] = (max_b[mask_h] + population[mask_h]) * 0.5
        
        # 5. Selection and Updates
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        succ_f = []
        succ_cr = []
        succ_w = []
        
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Add parent to Archive before replacement
                if n_archive < archive_size:
                    archive[n_archive] = population[i].copy()
                    n_archive += 1
                else:
                    ridx = np.random.randint(0, archive_size)
                    archive[ridx] = population[i].copy()
                
                new_pop[i] = trial[i]
                new_fit[i] = f_trial
                
                if f_trial < fitness[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    succ_w.append(fitness[i] - f_trial)
                    
                if f_trial < best_val:
                    best_val = f_trial
                    best_sol = trial[i].copy()
                    
        population = new_pop
        fitness = new_fit
        
        # 6. Memory Update (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            succ_f = np.array(succ_f)
            succ_cr = np.array(succ_cr)
            w = np.array(succ_w)
            w = w / np.sum(w)
            
            mean_f = np.sum(w * succ_f**2) / np.sum(w * succ_f)
            mean_cr = np.sum(w * succ_cr)
            
            mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
            mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
            k_mem = (k_mem + 1) % H
            
        # 7. Convergence Check & Restart
        # If population variance is negligible, we are stuck.
        if np.std(fitness) < 1e-8 or (np.max(fitness) - np.min(fitness)) < 1e-8:
            # 7a. Local Polish on Best
            if time.time() - start_time < max_time:
                p_sol, p_val = local_polish(best_sol, best_val)
                if p_val < best_val:
                    best_val = p_val
                    best_sol = p_sol
            
            # 7b. Restart
            # Soft restart: Keep best, re-initialize others
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_sol.copy()
            fitness = np.full(pop_size, np.inf)
            fitness[0] = best_val
            
            # Reset Archive and Memory
            n_archive = 0
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            
            # Evaluate new individuals
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(population[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = population[i].copy()

    return best_val
