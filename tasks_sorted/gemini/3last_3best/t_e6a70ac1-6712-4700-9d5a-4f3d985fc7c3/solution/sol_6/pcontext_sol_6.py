#The following is a fully operational and self-contained algorithm implementing **L-SHADE with jDE Parameter Adaptation**.
#
#### Algorithm Description
#To achieve the minimum output value within the time limit, this algorithm combines several state-of-the-art strategies from evolutionary computation:
#
#1.  **Linear Population Size Reduction (LPSR)**: The population size starts large (to explore the search space) and linearly decreases to a minimum size as time progresses. This naturally transitions the search from exploration to exploitation.
#2.  **DE/current-to-pbest/1/bin Strategy**: A robust mutation strategy that guides individuals toward the top $p\%$ of the population ($p$-best) while utilizing an **External Archive** of inferior solutions to maintain diversity.
#3.  **jDE Parameter Adaptation**: Instead of fixed parameters, each individual has its own Mutation Factor ($F$) and Crossover Rate ($CR$). These values are evolved along with the solution; successful parameters are retained, while failing ones are discarded or re-sampled.
#4.  **Soft Restarts**: If the population converges (low variance) before the time limit is reached, a "soft restart" is triggered. The best solution is kept, while the rest of the population is re-initialized to escape local optima.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using an advanced Self-Adaptive Differential Evolution 
    algorithm (jDE) combined with Linear Population Size Reduction (LPSR) and an 
    external archive. This approach is modeled after L-SHADE, a state-of-the-art 
    optimizer, adapting it for time-limited execution.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Initial Population Size:
    # Scale with dimension to ensure adequate coverage.
    # Heuristic: ~25 * sqrt(D), clamped to [40, 200] for efficiency.
    initial_pop_size = int(round(25 * np.sqrt(dim)))
    initial_pop_size = max(40, min(200, initial_pop_size))
    
    # Minimum Population Size for the final convergence phase
    min_pop_size = 4
    
    # Current population variables
    pop_size = initial_pop_size
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # --- jDE Parameter Initialization ---
    # Each individual carries its own F and CR parameters.
    # F: Mutation factor, CR: Crossover probability.
    # Initial values: F=0.5, CR=0.9 are standard starting points.
    F_arr = np.full(pop_size, 0.5)
    CR_arr = np.full(pop_size, 0.9)
    
    # External Archive
    # Stores potentially useful solutions that were replaced.
    # Enhances diversity for the 'current-to-pbest' mutation.
    archive = []
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_sol = None
    
    # Helper for Time Check
    def check_time():
        return (time.time() - start_time) >= max_time

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if check_time(): return best_fitness
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # --- Optimization Loop ---
    while not check_time():
        
        # 1. Linear Population Size Reduction (LPSR)
        # Linearly reduce population size from initial to min based on elapsed time.
        # This shifts focus from exploration (start) to exploitation (end).
        elapsed = time.time() - start_time
        progress = elapsed / max_time
        
        # Calculate target size
        # We target reaching min_pop_size at 95% of max_time to allow a final squeeze.
        reduction_ratio = min(1.0, progress / 0.95)
        target_size = int(round(initial_pop_size + reduction_ratio * (min_pop_size - initial_pop_size)))
        target_size = max(min_pop_size, target_size)
        
        # Apply reduction if needed
        if target_size < pop_size:
            # Sort population to keep the best individuals
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            F_arr = F_arr[sort_indices]
            CR_arr = CR_arr[sort_indices]
            
            # Truncate
            pop_size = target_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            F_arr = F_arr[:pop_size]
            CR_arr = CR_arr[:pop_size]
            
            # Resize Archive (Maintain size relative to population, e.g., 200%)
            max_arc_size = max(5, int(2.0 * pop_size))
            if len(archive) > max_arc_size:
                # Randomly remove elements
                idxs_keep = np.random.choice(len(archive), max_arc_size, replace=False)
                archive = [archive[i] for i in idxs_keep]

        # 2. Sort Population
        # Required for 'current-to-pbest' strategy (selecting top p%).
        sort_indices = np.argsort(fitness)
        pop = pop[sort_indices]
        fitness = fitness[sort_indices]
        F_arr = F_arr[sort_indices]
        CR_arr = CR_arr[sort_indices]
        
        # 3. Convergence / Stagnation Check
        # If population variance is near zero, we are stuck.
        # Strategy: "Soft Restart" - Keep best, scatter the rest.
        if np.std(fitness) < 1e-9 and progress < 0.85:
            # Reset all but the best (index 0)
            pop[1:] = min_b + np.random.rand(pop_size-1, dim) * diff_b
            
            # Re-evaluate scattered individuals
            for k in range(1, pop_size):
                if check_time(): return best_fitness
                v = func(pop[k])
                fitness[k] = v
                if v < best_fitness:
                    best_fitness = v
                    best_sol = pop[k].copy()
            
            # Re-sort
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            # Reset Parameters for new individuals
            F_arr[1:] = 0.5
            CR_arr[1:] = 0.9

        # 4. Parameter Adaptation (jDE)
        # Create candidate F and CR values
        # F: with prob 0.1, new random (0.1, 1.0)
        rand_F = np.random.rand(pop_size)
        candidate_F = np.where(rand_F < 0.1, 0.1 + 0.9 * np.random.rand(pop_size), F_arr)
        
        # CR: with prob 0.1, new random (0.0, 1.0)
        rand_CR = np.random.rand(pop_size)
        candidate_CR = np.where(rand_CR < 0.1, np.random.rand(pop_size), CR_arr)
        
        # 5. Mutation Strategy: DE/current-to-pbest/1/bin
        # v = x + F(x_pbest - x) + F(x_r1 - x_r2)
        
        # Select p-best: Random from top p%
        # p is typically 5% to 20%. 
        p_count = max(2, int(pop_size * 0.11))
        idxs_pbest = np.random.randint(0, p_count, pop_size)
        x_pbest = pop[idxs_pbest]
        
        # Select r1: Random from population, r1 != i
        idxs_r1 = np.random.randint(0, pop_size, pop_size)
        # Handle collisions simply
        for i in range(pop_size):
            while idxs_r1[i] == i:
                idxs_r1[i] = np.random.randint(0, pop_size)
        x_r1 = pop[idxs_r1]
        
        # Select r2: Random from Union(Population, Archive), r2 != i, r2 != r1
        # Prepare Union Access
        n_arch = len(archive)
        if n_arch > 0:
            arr_archive = np.array(archive)
            # Generate raw indices into the union
            total_size = pop_size + n_arch
            idxs_r2_raw = np.random.randint(0, total_size, pop_size)
            
            x_r2 = np.zeros_like(pop)
            
            # Map indices
            mask_pop = idxs_r2_raw < pop_size
            mask_arc = ~mask_pop
            
            if np.any(mask_pop):
                x_r2[mask_pop] = pop[idxs_r2_raw[mask_pop]]
            if np.any(mask_arc):
                # archive indices start at 0
                arc_indices = idxs_r2_raw[mask_arc] - pop_size
                x_r2[mask_arc] = arr_archive[arc_indices]
        else:
            # Only population available
            idxs_r2 = np.random.randint(0, pop_size, pop_size)
            for i in range(pop_size):
                while idxs_r2[i] == i or idxs_r2[i] == idxs_r1[i]:
                    idxs_r2[i] = np.random.randint(0, pop_size)
            x_r2 = pop[idxs_r2]
            
        # Calculate Mutant Vector
        F_col = candidate_F[:, None]
        mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # 6. Crossover (Binomial)
        CR_col = candidate_CR[:, None]
        rand_cross = np.random.rand(pop_size, dim)
        cross_mask = rand_cross < CR_col
        
        # Ensure at least one dimension is changed
        j_rand = np.random.randint(0, dim, pop_size)
        j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
        j_rand_mask[np.arange(pop_size), j_rand] = True
        cross_mask = cross_mask | j_rand_mask
        
        trial = np.where(cross_mask, mutant, pop)
        
        # Boundary Handling
        trial = np.clip(trial, min_b, max_b)
        
        # 7. Evaluation & Selection
        for i in range(pop_size):
            if check_time(): return best_fitness
            
            f_trial = func(trial[i])
            f_target = fitness[i]
            
            if f_trial <= f_target:
                # Success
                
                # Add parent to archive if trial is strictly better
                if f_trial < f_target:
                    archive.append(pop[i].copy())
                
                # Update Population
                pop[i] = trial[i]
                fitness[i] = f_trial
                
                # Adopt new parameters
                F_arr[i] = candidate_F[i]
                CR_arr[i] = candidate_CR[i]
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
            else:
                # Failure: Discard trial, keep parent and old parameters
                pass
                
        # Maintain Archive Limit (Clean up occasionally or per gen)
        limit_arc = max(5, int(2.5 * pop_size))
        if len(archive) > limit_arc:
            # Shuffle and slice is fast
            keep_idxs = np.random.choice(len(archive), limit_arc, replace=False)
            archive = [archive[k] for k in keep_idxs]
            
    return best_fitness
