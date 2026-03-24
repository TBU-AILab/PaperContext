#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution) with an External Archive and Soft Restarts**.
#
#**Key Improvements:**
#1.  **SHADE Strategy**: Unlike simple JADE, this uses a historical memory ($M_F, M_{CR}$) to adapt mutation factor $F$ and crossover rate $CR$. This allows the algorithm to learn from successful updates over recent generations, balancing exploration (high F) and exploitation (low F).
#2.  **Vectorized Operations**: All parameter generation, mutation (`current-to-pbest/1`), crossover, and bound handling are fully vectorized using NumPy. This minimizes Python overhead and maximizes the number of generations within the `max_time`.
#3.  **External Archive**: An archive stores recently replaced inferior solutions. This effectively increases the diversity of the difference vectors ($x_{r1} - x_{r2}$) without evaluating extra points, preventing premature convergence.
#4.  **Soft Restarts**: If the population fitness variance collapses (convergence), the algorithm triggers a restart. It keeps the global best solution (elitism) and re-initializes the rest of the population to explore new basins of attraction.
#5.  **Robust Bound Handling**: It uses simple clipping to keep solutions within bounds, which is robust for general black-box optimization problems.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Adaptive to dimension.
    # SHADE performs well with N around 100. We scale it with dim but clip to safe limits.
    pop_size = int(np.clip(18 * dim, 60, 200))
    
    # Archive parameters: Stores historic bad vectors to preserve diversity
    archive_size = int(pop_size * 2.6) 
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # SHADE Memory parameters
    H = 5 # Size of the historical memory
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Bounds preprocessing
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
            
    # --- Main Optimization Loop ---
    while True:
        # Time Check
        if time.time() - start_time >= max_time:
            return best_val
            
        # 1. Parameter Generation
        # Randomly select memory index
        r_idx = np.random.randint(0, H, pop_size)
        m_f = mem_f[r_idx]
        m_cr = mem_cr[r_idx]
        
        # Generate F using Cauchy distribution
        # F must be positive. If F <= 0, retry.
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        retry_mask = f <= 0
        while np.any(retry_mask):
            n_retry = np.sum(retry_mask)
            if n_retry > 0:
                f[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(n_retry)
                retry_mask = f <= 0
        f = np.minimum(f, 1.0) # Clip upper bound to 1.0
        
        # Generate CR using Normal distribution
        cr = np.random.normal(m_cr, 0.1, pop_size)
        cr = np.clip(cr, 0.0, 1.0)
        
        # 2. Mutation: current-to-pbest/1 with Archive
        # Sort population by fitness to identify p-best
        sort_idx = np.argsort(fitness)
        sorted_pop = population[sort_idx]
        
        # Randomize 'p' in [2/N, 0.2] to balance greediness
        p_min = 2.0 / pop_size
        p = np.random.uniform(p_min, 0.2)
        n_pbest = int(p * pop_size)
        n_pbest = max(n_pbest, 2)
        
        # Select pbest indices
        pbest_indices = np.random.randint(0, n_pbest, pop_size)
        x_pbest = sorted_pop[pbest_indices]
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Ensure r1 != i
        col_mask = (r1_indices == np.arange(pop_size))
        while np.any(col_mask):
            r1_indices[col_mask] = np.random.randint(0, pop_size, np.sum(col_mask))
            col_mask = (r1_indices == np.arange(pop_size))
        x_r1 = population[r1_indices]
        
        # Select r2 (distinct from i and r1, from Union(Pop, Archive))
        pool_size = pop_size + n_archive
        r2_indices = np.random.randint(0, pool_size, pop_size)
        # Ensure r2 != i and r2 != r1
        col_mask = (r2_indices == np.arange(pop_size)) | (r2_indices == r1_indices)
        while np.any(col_mask):
            r2_indices[col_mask] = np.random.randint(0, pool_size, np.sum(col_mask))
            col_mask = (r2_indices == np.arange(pop_size)) | (r2_indices == r1_indices)
            
        # Build x_r2
        x_r2 = np.zeros((pop_size, dim))
        mask_pop = r2_indices < pop_size
        x_r2[mask_pop] = population[r2_indices[mask_pop]]
        
        mask_arc = ~mask_pop
        if np.any(mask_arc):
            idx_arc = r2_indices[mask_arc] - pop_size
            x_r2[mask_arc] = archive[idx_arc]
            
        # Compute Mutant Vectors
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        mutant = population + f[:, None] * (x_pbest - population) + f[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        # Guarantee at least one dimension is changed
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # 4. Bound Handling (Clip)
        trial = np.clip(trial, min_b, max_b)
        
        # 5. Evaluation and Selection
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        succ_f = []
        succ_cr = []
        succ_diff = []
        
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Update Archive: Add replaced parent
                if n_archive < archive_size:
                    archive[n_archive] = population[i].copy()
                    n_archive += 1
                else:
                    rand_idx = np.random.randint(0, archive_size)
                    archive[rand_idx] = population[i].copy()
                
                # Record success for memory update (if strict improvement)
                if f_trial < fitness[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    succ_diff.append(fitness[i] - f_trial)
                
                new_pop[i] = trial[i]
                new_fit[i] = f_trial
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_sol = trial[i].copy()
                    
        population = new_pop
        fitness = new_fit
        
        # 6. Memory Update (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            s_f = np.array(succ_f)
            s_cr = np.array(succ_cr)
            s_diff = np.array(succ_diff)
            
            # Weights based on improvement magnitude
            w = s_diff / np.sum(s_diff)
            
            # Mean F (Lehmer Mean)
            mean_f = np.sum(w * s_f**2) / np.sum(w * s_f)
            # Mean CR (Weighted Arithmetic Mean)
            mean_cr = np.sum(w * s_cr)
            
            mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
            mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
            k_mem = (k_mem + 1) % H
            
        # 7. Restart Mechanism
        # If population diversity is lost (std dev of fitness is negligible), restart.
        if np.std(fitness) < 1e-8:
            # Re-init population
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            # Keep elite (best solution found so far)
            population[0] = best_sol.copy()
            fitness = np.full(pop_size, np.inf)
            fitness[0] = best_val
            
            # Reset Archive and Memory
            n_archive = 0
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            
            # Evaluate new population (skip index 0)
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(population[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = population[i].copy()

    return best_val
