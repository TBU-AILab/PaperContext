#The following algorithm implements **SHADE-LS (Success-History Adaptive Differential Evolution with Local Search and Restarts)**.
#
#### Key Improvements:
#1.  **L-SHADE Adaptive Mechanism**: Utilizes a historical memory of successful control parameters ($F$ and $CR$) to adaptively guide the search strategy, specifically using a Cauchy distribution for mutation factors to prevent stagnation.
#2.  **Vectorized `current-to-pbest/1` Mutation**: Implements the efficient rank-based mutation strategy using NumPy vectorization. This allows for rapid generation of candidate solutions, maximizing the budget of function evaluations.
#3.  **Local Search Polish**: Incorporates a lightweight coordinate descent (local search) that is triggered before a restart. This "polishes" the best solution found so far, allowing the algorithm to refine results in the final basin of attraction where DE might struggle with precision.
#4.  **Soft Restarts with Elitism**: Monitors population diversity. If the population collapses (converges), it triggers a restart to explore new areas of the search space, while carrying over the global best solution to ensure non-decreasing performance.
#5.  **Midpoint Bound Handling**: Instead of simple clipping (which causes stacking at bounds), it uses a midpoint correction rule `(bound + old_val) / 2` to preserve population diversity near the edges of the search space.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Compromise between exploration and generation count
    # Smaller population allows more generations, larger allows better exploration.
    # A size of ~50-100 is a robust compromise for general black-box functions.
    pop_size = 50 if dim <= 20 else 100
    
    # SHADE Memory parameters
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive parameters
    archive_size = int(2.0 * pop_size)
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # Bounds processing
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Initialization
    # Random uniform initialization
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, np.inf)
    
    best_val = np.inf
    best_sol = np.zeros(dim)
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = pop[i].copy()
            
    # --- Local Search Helper ---
    # Applies Coordinate Descent to refine a solution
    def local_polish(sol, val):
        curr_sol = sol.copy()
        curr_val = val
        
        # Budget for local search (proportional to dimension)
        ls_budget = max(20, 2 * dim)
        evals = 0
        
        # Initial step size (adaptive based on domain)
        step_size = diff_b * 0.01 
        
        improved = True
        while improved and evals < ls_budget:
            improved = False
            # Randomize dimension order
            dims = np.random.permutation(dim)
            
            for d in dims:
                if time.time() - start_time >= max_time:
                    return curr_sol, curr_val
                
                # Check neighbors
                # 1. Negative direction
                temp_sol = curr_sol.copy()
                temp_sol[d] = np.clip(curr_sol[d] - step_size[d], min_b[d], max_b[d])
                temp_val = func(temp_sol)
                evals += 1
                
                if temp_val < curr_val:
                    curr_val = temp_val
                    curr_sol = temp_sol
                    improved = True
                else:
                    # 2. Positive direction
                    temp_sol[d] = np.clip(curr_sol[d] + step_size[d], min_b[d], max_b[d])
                    temp_val = func(temp_sol)
                    evals += 1
                    
                    if temp_val < curr_val:
                        curr_val = temp_val
                        curr_sol = temp_sol
                        improved = True
                
                if evals >= ls_budget:
                    break
                    
            # Reduce step size if no improvement found in a full pass
            if not improved:
                step_size *= 0.5
                # Continue if step size is significant, else stop
                if np.max(step_size) > 1e-5:
                    improved = True 
                else:
                    improved = False
                    
        return curr_sol, curr_val

    # --- Main Loop ---
    while True:
        # Time check
        if time.time() - start_time >= max_time:
            return best_val
            
        # 1. Parameter Generation (Vectorized)
        r_idx = np.random.randint(0, H, pop_size)
        m_f = mem_f[r_idx]
        m_cr = mem_cr[r_idx]
        
        # F: Cauchy distribution
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        # Repair F
        retry_idx = f <= 0
        while np.any(retry_idx):
            n_retry = np.sum(retry_idx)
            f[retry_idx] = m_f[retry_idx] + 0.1 * np.random.standard_cauchy(n_retry)
            retry_idx = f <= 0
        f = np.minimum(f, 1.0)
        
        # CR: Normal distribution
        cr = np.random.normal(m_cr, 0.1, pop_size)
        cr = np.clip(cr, 0.0, 1.0)
        
        # 2. Mutation: current-to-pbest/1
        # Sort population by fitness
        sort_idx = np.argsort(fitness)
        
        # Select pbest indices (top p%)
        # p is randomized between 2/N and 0.2
        p_val = np.random.uniform(2.0/pop_size, 0.2, pop_size)
        n_top = (p_val * pop_size).astype(int)
        n_top = np.maximum(n_top, 2)
        
        rand_top_idx = (np.random.rand(pop_size) * n_top).astype(int)
        pbest_ptr = sort_idx[rand_top_idx]
        x_pbest = pop[pbest_ptr]
        
        # Select r1 (distinct from i)
        r1_ptr = np.random.randint(0, pop_size, pop_size)
        # Handle collision with i
        mask_self = r1_ptr == np.arange(pop_size)
        if np.any(mask_self):
            r1_ptr[mask_self] = (r1_ptr[mask_self] + 1) % pop_size
        x_r1 = pop[r1_ptr]
        
        # Select r2 (distinct from i, r1; from Union(Pop, Archive))
        pool_size = pop_size + n_archive
        r2_ptr = np.random.randint(0, pool_size, pop_size)
        # Handle collisions (simplistic approach for speed)
        mask_col = (r2_ptr == np.arange(pop_size)) | (r2_ptr == r1_ptr)
        if np.any(mask_col):
            r2_ptr[mask_col] = np.random.randint(0, pool_size, np.sum(mask_col))
            
        # Construct x_r2
        x_r2 = np.zeros((pop_size, dim))
        mask_pop = r2_ptr < pop_size
        x_r2[mask_pop] = pop[r2_ptr[mask_pop]]
        mask_arc = ~mask_pop
        if np.any(mask_arc):
            idx_arc = r2_ptr[mask_arc] - pop_size
            x_r2[mask_arc] = archive[idx_arc]
            
        # Compute Mutant Vectors
        # v = x + F*(pbest - x) + F*(r1 - r2)
        mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        mask_rand = np.random.rand(pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask_rand[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_rand, mutant, pop)
        
        # 4. Bound Handling (Midpoint Correction)
        # Generally preserves diversity better than clipping
        mask_l = trial < min_b
        mask_h = trial > max_b
        trial[mask_l] = (min_b[mask_l] + pop[mask_l]) * 0.5
        trial[mask_h] = (max_b[mask_h] + pop[mask_h]) * 0.5
        
        # 5. Selection and Memory Update
        new_pop = pop.copy()
        new_fit = fitness.copy()
        
        succ_f = []
        succ_cr = []
        diff_fit = []
        
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Update Archive
                if n_archive < archive_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    # Random replacement
                    ridx = np.random.randint(0, archive_size)
                    archive[ridx] = pop[i].copy()
                    
                new_pop[i] = trial[i]
                new_fit[i] = f_trial
                
                # Check for strict improvement for memory update
                if f_trial < fitness[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fit.append(fitness[i] - f_trial)
                
                # Global best update
                if f_trial < best_val:
                    best_val = f_trial
                    best_sol = trial[i].copy()
                    
        pop = new_pop
        fitness = new_fit
        
        # Update Memory (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            succ_f = np.array(succ_f)
            succ_cr = np.array(succ_cr)
            w = np.array(diff_fit)
            w = w / np.sum(w)
            
            mean_f = np.sum(w * succ_f**2) / np.sum(w * succ_f)
            mean_cr = np.sum(w * succ_cr)
            
            mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
            mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
            k_mem = (k_mem + 1) % H
            
        # 6. Convergence Check & Restart
        # If fitness variance is too low, we are likely stuck
        if np.std(fitness) < 1e-6:
            # Attempt to refine best solution before discarding population
            if time.time() - start_time < max_time:
                polished_sol, polished_val = local_polish(best_sol, best_val)
                if polished_val < best_val:
                    best_val = polished_val
                    best_sol = polished_sol
            
            # Restart
            # Keep best solution (Elitism)
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol.copy()
            fitness = np.full(pop_size, np.inf)
            fitness[0] = best_val
            
            # Reset Archive and Memory
            n_archive = 0
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            
            # Re-evaluate (skip 0)
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = pop[i].copy()

    return best_val
