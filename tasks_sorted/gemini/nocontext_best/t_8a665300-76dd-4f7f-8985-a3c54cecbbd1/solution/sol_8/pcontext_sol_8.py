#To improve upon the previous JADE implementation, we will implement **L-SHADE** (Linear Success-History based Adaptive Differential Evolution). L-SHADE is a state-of-the-art improvement over JADE that introduces **Linear Population Size Reduction (LPSR)** and a historical memory for adaptive parameters. 
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR):** The population size is dynamically reduced from a large initial size (exploration) to a small size (exploitation) as time progresses. This allocates computational resources more efficiently.
#2.  **Success-History Memory:** Instead of a single adaptive parameter, a memory of successful parameter configurations is maintained, allowing the algorithm to learn the landscape's properties over time.
#3.  **Coordinate Descent Polishing:** A robust, gradient-free local search (similar to Powell's method) is applied at the end or when the population converges, to refine the best solution with high precision.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE (Linear Success-History Adaptive Differential Evolution)
    with Linear Population Size Reduction and Coordinate Descent Polishing.
    """
    
    # --- 1. Initialization & Configuration ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Precompute bound arrays for fast access
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # L-SHADE Configuration
    # Initial population size: High for exploration. 
    # Capped for performance within limited time.
    pop_size_init = int(min(500, max(30, 18 * dim)))
    pop_size_min = 4 # Minimum population size at the end
    
    # Memory for adaptive parameters (H = memory size)
    H = 5
    M_cr = np.full(H, 0.5) # Memory for Crossover Rate
    M_f = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0              # Memory index pointer
    
    # Archive for inferior solutions (maintains diversity)
    archive = []
    
    # Initial Population Generation (Latin Hypercube Sampling)
    population = np.zeros((pop_size_init, dim))
    for d in range(dim):
        # Divide dimension range into N intervals and pick one point per interval
        edges = np.linspace(min_b[d], max_b[d], pop_size_init + 1)
        points = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(points)
        population[:, d] = points
        
    # Evaluate Initial Population
    pop_fitness = np.array([func(ind) for ind in population])
    
    # Track Global Best
    best_idx = np.argmin(pop_fitness)
    best_fitness = pop_fitness[best_idx]
    best_solution = population[best_idx].copy()
    
    current_pop_size = pop_size_init

    # --- 2. Helper Functions ---
    
    def get_remaining_seconds():
        return (time_limit - (datetime.now() - start_time)).total_seconds()

    def local_search(x_center, current_best_val, budget_time):
        """
        Coordinate Descent (Pattern Search) for final polishing.
        Robust to lack of gradients and handles diagonal valleys reasonably well.
        """
        ls_start = datetime.now()
        ls_limit = timedelta(seconds=budget_time)
        
        x = x_center.copy()
        val = current_best_val
        
        # Initial step sizes relative to bounds
        step_scale = 0.05
        steps = diff_b * step_scale
        
        while (datetime.now() - ls_start) < ls_limit:
            improved = False
            for i in range(dim):
                if (datetime.now() - ls_start) >= ls_limit: break
                
                # Probe positive direction
                x_new = x.copy()
                x_new[i] += steps[i]
                if x_new[i] > max_b[i]: x_new[i] = max_b[i]
                
                v_new = func(x_new)
                if v_new < val:
                    val = v_new
                    x = x_new
                    steps[i] *= 1.2 # Accelerate on success
                    improved = True
                    continue
                    
                # Probe negative direction
                x_new = x.copy()
                x_new[i] -= steps[i]
                if x_new[i] < min_b[i]: x_new[i] = min_b[i]
                
                v_new = func(x_new)
                if v_new < val:
                    val = v_new
                    x = x_new
                    steps[i] *= 1.2
                    improved = True
                else:
                    steps[i] *= 0.5 # Decelerate on failure
            
            # Convergence check
            if not improved and np.max(steps) < 1e-9:
                break
                    
        return x, val

    # --- 3. Main Optimization Loop ---
    while True:
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Stop if time is up
        if elapsed >= max_time:
            break
            
        progress = elapsed / max_time
        
        # Reserve a small buffer (5% or 0.5s) for final polishing
        if max_time - elapsed < max(0.5, max_time * 0.05):
            break

        # -- Linear Population Size Reduction (LPSR) --
        # Linearly reduce population from init to min based on time progress
        target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        target_size = max(pop_size_min, target_size)
        
        if current_pop_size > target_size:
            # Sort population by fitness
            sorted_indices = np.argsort(pop_fitness)
            
            # Keep the top 'target_size' individuals
            keep_indices = sorted_indices[:target_size]
            population = population[keep_indices]
            pop_fitness = pop_fitness[keep_indices]
            
            current_pop_size = target_size
            
            # Resize archive to match current population size
            if len(archive) > current_pop_size:
                del archive[current_pop_size:]

        # -- Parameter Generation --
        # Assign an index from memory to each individual
        rand_mem_indices = np.random.randint(0, H, current_pop_size)
        
        # Generate Crossover Rates (Normal Distribution)
        mean_crs = M_cr[rand_mem_indices]
        crs = np.random.normal(mean_crs, 0.1)
        crs = np.clip(crs, 0, 1)
        
        # Generate Scaling Factors (Cauchy Distribution)
        mean_fs = M_f[rand_mem_indices]
        fs = np.zeros(current_pop_size)
        # Cauchy generation loop to handle constraints
        for i in range(current_pop_size):
            while True:
                # Cauchy: location + scale * tan(pi * (rand - 0.5))
                f_val = mean_fs[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                if f_val > 0:
                    fs[i] = min(f_val, 1.0)
                    break
        
        # -- Mutation (current-to-pbest/1) --
        # Sort to identify the top p-best individuals
        sorted_indices = np.argsort(pop_fitness)
        
        # p_best_rate is typically 0.11 or adaptive. We use 0.11 (top 11%)
        p_count = max(2, int(current_pop_size * 0.11))
        p_best_pool = sorted_indices[:p_count]
        
        new_pop = np.zeros_like(population)
        new_fitness = np.zeros_like(pop_fitness)
        
        success_f = []
        success_cr = []
        success_improvement = []
        
        # Archive Size
        n_archive = len(archive)
        
        for i in range(current_pop_size):
            if get_remaining_seconds() <= 0: break
            
            x_i = population[i]
            
            # 1. Select p-best from top p%
            idx_pbest = np.random.choice(p_best_pool)
            x_pbest = population[idx_pbest]
            
            # 2. Select r1 (from population, distinct from i)
            r1 = np.random.randint(0, current_pop_size)
            while r1 == i:
                r1 = np.random.randint(0, current_pop_size)
            x_r1 = population[r1]
            
            # 3. Select r2 (from Union of Population and Archive, distinct from i and r1)
            limit_pool = current_pop_size + n_archive
            r2_idx = np.random.randint(0, limit_pool)
            
            # Validate r2 logic to ensure distinctness
            # We map the virtual index r2_idx to actual data
            x_r2 = None
            if r2_idx < current_pop_size:
                r2 = r2_idx
                # Resample if collision
                while r2 == i or r2 == r1:
                    r2_idx = np.random.randint(0, limit_pool)
                    if r2_idx < current_pop_size: r2 = r2_idx
                    else: break 
                
                if r2_idx < current_pop_size:
                    x_r2 = population[r2_idx]
                else:
                    x_r2 = archive[r2_idx - current_pop_size]
            else:
                x_r2 = archive[r2_idx - current_pop_size]
            
            # Compute Mutant Vector
            v_i = x_i + fs[i] * (x_pbest - x_i) + fs[i] * (x_r1 - x_r2)
            
            # -- Crossover (Binomial) --
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < crs[i]
            mask[j_rand] = True # Ensure at least one dimension changes
            
            u_i = np.where(mask, v_i, x_i)
            
            # -- Bound Handling (Reflection/Correction) --
            # Instead of clipping, set to average of current and bound (preserves evolution direction better)
            below_bounds = u_i < min_b
            above_bounds = u_i > max_b
            
            u_i[below_bounds] = (x_i[below_bounds] + min_b[below_bounds]) / 2.0
            u_i[above_bounds] = (x_i[above_bounds] + max_b[above_bounds]) / 2.0
            
            # -- Selection --
            fit_u = func(u_i)
            
            if fit_u < pop_fitness[i]:
                new_pop[i] = u_i
                new_fitness[i] = fit_u
                
                # Record success for memory update
                success_f.append(fs[i])
                success_cr.append(crs[i])
                success_improvement.append(pop_fitness[i] - fit_u)
                
                # Add replaced parent to archive
                archive.append(x_i.copy())
                
                # Update Global Best
                if fit_u < best_fitness:
                    best_fitness = fit_u
                    best_solution = u_i.copy()
            else:
                new_pop[i] = x_i
                new_fitness[i] = pop_fitness[i]
        
        # Update Population arrays
        population = new_pop
        pop_fitness = new_fitness
        
        # -- Update History Memory --
        if len(success_f) > 0:
            success_f = np.array(success_f)
            success_cr = np.array(success_cr)
            imp = np.array(success_improvement)
            
            # Weighted Lehmer Mean based on improvement magnitude
            total_imp = np.sum(imp)
            if total_imp > 0:
                weights = imp / total_imp
                
                # Update Mean F (Lehmer Mean)
                mean_f_new = np.sum(weights * (success_f ** 2)) / (np.sum(weights * success_f) + 1e-15)
                M_f[k_mem] = mean_f_new
                
                # Update Mean CR (Weighted Arithmetic Mean)
                mean_cr_new = np.sum(weights * success_cr)
                M_cr[k_mem] = mean_cr_new
                
                # Advance memory index
                k_mem = (k_mem + 1) % H

    # --- 4. Final Polish ---
    # Use any remaining time to refine the best found solution using Local Search
    rem_time = get_remaining_seconds()
    if rem_time > 0.05 and best_solution is not None:
        polished_sol, polished_val = local_search(best_solution, best_fitness, rem_time)
        if polished_val < best_fitness:
            best_fitness = polished_val
            
    return best_fitness
