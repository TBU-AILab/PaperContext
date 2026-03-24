#Here is an improved algorithm.
#
#**Improvements Explanation:**
#1.  **Efficiency via Vectorization:** The previous implementation used explicit Python loops for selection and updates, which created a bottleneck. This version vectorizes parameter generation and selection, significantly increasing the number of function evaluations per second.
#2.  **L-SHADE-RSP / jSO Hybrid:** I implemented features from the top-performing Differential Evolution variants (jSO and L-SHADE-RSP):
#    *   **Linear Population Size Reduction (LPSR):** Linearly reduces population size from an initial high value (for exploration) to a minimum (for exploitation) based on elapsed time.
#    *   **Weighted Lehmer Mean:** Updates the memory (History) using weighted improvements, giving more influence to mutations that resulted in larger fitness gains.
#3.  **Continuous "Best-Solution" Polishing:** Instead of waiting for stagnation, the algorithm periodically applies a lightweight Coordinate Descent (Local Search) on the current best solution to refine it rapidly.
#4.  **Diversity-Based Restart:** Monitors population variance. If the population collapses (converges) before time is up, it triggers a restart while preserving the best solution, allowing the algorithm to escape local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimized L-SHADE with Linear Population Size Reduction (LPSR),
    Vectorized Operations, and Coordinate Descent Polishing.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Check if we have enough time left (with a small safety buffer)
    def check_time(buffer_seconds=0.05):
        return (datetime.now() - start_time) < (time_limit - timedelta(seconds=buffer_seconds))

    # --- Initialization ---
    rng = np.random.default_rng()
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing: Start large for exploration, reduce linearly
    # Cap max size to ensure speed on high dimensions
    pop_size_max = int(min(500, 25 * dim)) 
    pop_size_min = 4
    pop_size = pop_size_max
    
    # Initial Population
    pop = min_b + rng.random((pop_size, dim)) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initial Evaluation
    best_idx = 0
    best_val = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        if not check_time(): return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i
            
    best_sol = pop[best_idx].copy()
    
    # L-SHADE Memory
    h_mem = 5
    mem_sf = np.full(h_mem, 0.5)
    mem_cr = np.full(h_mem, 0.8)
    mem_k = 0
    
    # External Archive (stores inferior solutions to maintain diversity in mutation)
    arc_rate = 2.0
    archive = np.zeros((0, dim))
    
    # Local Search Parameters
    ls_radius = diff_b * 0.05 # Initial step size
    
    # --- Main Optimization Loop ---
    while check_time(0.1): # Ensure at least 0.1s remains for a batch
        
        # 1. Linear Population Size Reduction (LPSR)
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        
        # Calculate target population size based on time progress
        plan_pop_size = int(round((pop_size_min - pop_size_max) * progress + pop_size_max))
        plan_pop_size = max(pop_size_min, plan_pop_size)
        
        if pop_size > plan_pop_size:
            # Sort population by fitness to keep the best
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            # Truncate
            pop_size = plan_pop_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Reduce archive size if necessary
            target_arc_size = int(pop_size * arc_rate)
            if archive.shape[0] > target_arc_size:
                del_count = archive.shape[0] - target_arc_size
                # Remove random elements from archive
                del_indices = rng.choice(archive.shape[0], del_count, replace=False)
                archive = np.delete(archive, del_indices, axis=0)

            # Reset best pointer (index 0 is best after sort)
            best_idx = 0
            best_val = fitness[0]
            best_sol = pop[0].copy()

        # 2. Parameter Generation
        # Randomly select memory slots
        r_indices = rng.integers(0, h_mem, pop_size)
        mu_sf = mem_sf[r_indices]
        mu_cr = mem_cr[r_indices]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = rng.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_sf, 0.1)
        # Vectorized retry for F <= 0
        f = mu_sf + 0.1 * np.tan(np.pi * (rng.random(pop_size) - 0.5))
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            f[bad_mask] = mu_sf[bad_mask] + 0.1 * np.tan(np.pi * (rng.random(np.sum(bad_mask)) - 0.5))
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: current-to-pbest/1
        # p-best parameter linearly decreases from 0.2 to 0.05 (focusing on exploitation later)
        p_val = max(2/pop_size, 0.2 * (1 - progress))
        
        # Identify p-best individuals
        # Partial sort is faster than full sort for finding top k
        p_count = max(2, int(p_val * pop_size))
        sorted_indices = np.argsort(fitness)
        pbest_candidates = sorted_indices[:p_count]
        
        # Select pbest for each individual
        pbest_idxs = rng.choice(pbest_candidates, pop_size)
        x_pbest = pop[pbest_idxs]
        
        # Select r1 != i
        r1 = rng.integers(0, pop_size, pop_size)
        collision = (r1 == np.arange(pop_size))
        while np.any(collision):
            r1[collision] = rng.integers(0, pop_size, np.sum(collision))
            collision = (r1 == np.arange(pop_size))
        x_r1 = pop[r1]
        
        # Select r2 != i, r2 != r1 from (Pop U Archive)
        if archive.shape[0] > 0:
            union_pop = np.vstack((pop, archive))
        else:
            union_pop = pop
            
        r2 = rng.integers(0, union_pop.shape[0], pop_size)
        collision = (r2 == np.arange(pop_size)) | (r2 == r1)
        while np.any(collision):
            r2[collision] = rng.integers(0, union_pop.shape[0], np.sum(collision))
            collision = (r2 == np.arange(pop_size)) | (r2 == r1)
        x_r2 = union_pop[r2]
        
        # Calculate Mutant Vector
        f_vec = f[:, np.newaxis]
        mutant = pop + f_vec * (x_pbest - pop) + f_vec * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        j_rand = rng.integers(0, dim, pop_size)
        cross_mask = rng.random((pop_size, dim)) < cr[:, np.newaxis]
        cross_mask[np.arange(pop_size), j_rand] = True
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Boundary Handling (Reflection)
        below = trial < min_b
        above = trial > max_b
        
        # Reflect back into bounds
        trial[below] = (min_b + (min_b - trial[below]))
        # Fallback if reflection is still out
        trial[trial < min_b] = min_b[np.where(trial < min_b)[1]]

        trial[above] = (max_b - (trial[above] - max_b))
        trial[trial > max_b] = max_b[np.where(trial > max_b)[1]]
        
        # 6. Evaluation and Selection
        success_f = []
        success_cr = []
        success_diff = []
        
        pop_changed = False
        
        # Evaluate loop
        # Check time periodically inside loop to safely handle large populations
        check_interval = max(1, pop_size // 10)
        
        for i in range(pop_size):
            if i % check_interval == 0 and not check_time(): 
                return best_val
            
            f_val = func(trial[i])
            
            if f_val <= fitness[i]:
                # Successful trial
                if f_val < fitness[i]:
                    # Add old vector to archive
                    if archive.shape[0] < int(pop_size * arc_rate):
                        archive = np.vstack((archive, pop[i]))
                    else:
                        # Random replacement
                        ridx = rng.integers(0, archive.shape[0])
                        archive[ridx] = pop[i]
                        
                    # Store params for memory update
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    success_diff.append(fitness[i] - f_val)
                    pop_changed = True
                
                # Update population
                pop[i] = trial[i]
                fitness[i] = f_val
                
                if f_val < best_val:
                    best_val = f_val
                    best_sol = trial[i].copy()
                    best_idx = i
        
        # 7. Memory Update (Weighted Lehmer Mean)
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            s_diff = np.array(success_diff)
            
            weights = s_diff / np.sum(s_diff)
            
            mean_f = np.sum(weights * (s_f**2)) / np.sum(weights * s_f)
            mean_cr = np.sum(weights * s_cr)
            
            mem_sf[mem_k] = 0.5 * mem_sf[mem_k] + 0.5 * mean_f
            mem_cr[mem_k] = 0.5 * mem_cr[mem_k] + 0.5 * mean_cr
            mem_k = (mem_k + 1) % h_mem
            
        # 8. Local Search (Coordinate Descent on Best)
        # Probabilistically run to refine best solution, taking small time budget
        if pop_changed and check_time(0.2):
            # Dynamic radius
            radius = ls_radius * (1.0 - progress)
            if radius < 1e-8: radius = 1e-5
            
            # Check random dimensions (limit to 5 to be fast)
            dims_to_check = rng.permutation(dim)[:min(dim, 5)]
            
            for d in dims_to_check:
                if not check_time(): break
                
                x_curr = best_sol[d]
                
                # Try negative step
                best_sol[d] = np.clip(x_curr - radius, min_b[d], max_b[d])
                v_neg = func(best_sol)
                
                if v_neg < best_val:
                    best_val = v_neg
                    fitness[best_idx] = v_neg
                    pop[best_idx, d] = best_sol[d]
                else:
                    # Try positive step
                    best_sol[d] = np.clip(x_curr + 0.5 * radius, min_b[d], max_b[d])
                    v_pos = func(best_sol)
                    
                    if v_pos < best_val:
                        best_val = v_pos
                        fitness[best_idx] = v_pos
                        pop[best_idx, d] = best_sol[d]
                    else:
                        # Revert
                        best_sol[d] = x_curr

        # 9. Restart Mechanism
        # If population diversity is lost (convergence) and time remains
        # Standard deviation of population < threshold
        if progress < 0.85:
            std_dev = np.mean(np.std(pop, axis=0))
            if std_dev < 1e-6:
                # Restart: Keep best, re-initialize others
                pop[1:] = min_b + rng.random((pop_size - 1, dim)) * diff_b
                # Reset memory
                mem_sf.fill(0.5)
                mem_cr.fill(0.5)
                
                # Evaluate new population
                for i in range(1, pop_size):
                    if not check_time(): return best_val
                    val = func(pop[i])
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_idx = i
                        best_sol = pop[i].copy()
                        
    return best_val
