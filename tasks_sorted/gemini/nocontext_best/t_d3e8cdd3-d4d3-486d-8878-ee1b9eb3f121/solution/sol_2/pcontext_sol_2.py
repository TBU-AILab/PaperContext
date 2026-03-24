#Here is a fully operational, self-contained implementation.
#
#This algorithm improves upon the previous submission by implementing **Vectorized L-SHADE** (Success-History based Adaptive Differential Evolution with Linear Population Reduction) combined with a **Coordinate Descent Local Search** for final polishing.
#
#**Key Improvements:**
#1.  **Vectorization**: Instead of iterating through the population to create mutants (which is slow in Python), this code uses NumPy matrix operations to generate the entire next generation at once. This significantly increases the number of generations possible within `max_time`.
#2.  **Linear Population Reduction (L-SHADE)**: The population size linearly decreases from $N_{init}$ to $N_{min}$ over the duration of the search. This encourages exploration early on and forces fast convergence in later stages.
#3.  **Coordinate Descent Polishing**: Differential Evolution is excellent at finding the "basin" of the global minimum but can be noisy at the very bottom. A deterministic local search (Coordinate Descent) is triggered in the final moments (or upon stagnation) to refine the best solution to high precision.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized L-SHADE (Linear Population Size Reduction
    Success-History Adaptive Differential Evolution) followed by Coordinate Descent polishing.
    """
    
    # --- Helper: Bounds Handling ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    diff = ub - lb

    def clip_to_bounds(vec):
        return np.clip(vec, lb, ub)

    # --- Initialization ---
    start_time = time.time()
    
    # L-SHADE Parameters
    # Initial pop size (higher is better for exploration)
    pop_size_init = max(30, int(15 * dim))
    # Minimum pop size (small for fast convergence at the end)
    pop_size_min = 4 
    pop_size = pop_size_init
    
    # Memory size for historical success of parameters
    H = 6
    mem_M_cr = np.full(H, 0.5)
    mem_M_f  = np.full(H, 0.5)
    k_mem = 0 # Memory index pointer

    # Archive
    archive = []
    
    # Initialize Population
    # We maintain population in range [0, 1] internally for simpler math, 
    # then scale to bounds for evaluation.
    # Actually, standard DE logic works best on real values, we just vector-clip.
    population = lb + np.random.rand(pop_size, dim) * diff
    fitness = np.full(pop_size, float('inf'))
    
    global_best_val = float('inf')
    global_best_vec = None

    # Evaluate Initial Population
    for i in range(pop_size):
        if (time.time() - start_time) > max_time:
            return global_best_val if global_best_vec is not None else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_vec = population[i].copy()

    # Calculate Max Generations (Estimate based on first eval time)
    # We treat max_time as a hard stop, but we use the elapsed time to drive the linear reduction
    
    # --- Main Loop ---
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # TIME CHECK: Reserve last 5% or 0.5 seconds for Local Search Polishing
        if elapsed >= max_time * 0.95:
            break

        # 1. Linear Population Size Reduction (LPSR)
        # Reduce population size linearly relative to consumed time
        progress = elapsed / max_time
        plan_pop_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
        
        if pop_size > plan_pop_size:
            # Sort by fitness and truncate worst
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Remove worst individuals
            num_to_remove = pop_size - plan_pop_size
            population = population[:-num_to_remove]
            fitness = fitness[:-num_to_remove]
            pop_size = plan_pop_size
            
            # Resize archive if it exceeds new pop_size
            if len(archive) > pop_size:
                # Randomly remove
                del_indices = np.random.choice(len(archive), len(archive) - pop_size, replace=False)
                archive = [arr for i, arr in enumerate(archive) if i not in del_indices]

        # 2. Parameter Generation (Vectorized)
        # Select random memory slot for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_M_cr[r_idx]
        m_f  = mem_M_f[r_idx]

        # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        # Use -1 to denote specialized constraint handling if needed, usually just clip is fine.
        
        # Generate F: Cauchy(m_f, 0.1), clipped [0, 1]
        # Cauchy = standard_cauchy * scale + loc
        f = np.random.standard_cauchy(pop_size) * 0.1 + m_f
        # Check constraints: if f > 1 -> 1, if f <= 0 -> regenerate
        # Fast vector correction for F
        while np.any(f <= 0):
            mask = f <= 0
            f[mask] = np.random.standard_cauchy(np.sum(mask)) * 0.1 + m_f[mask]
        f = np.minimum(f, 1.0)

        # 3. Mutation Strategy: current-to-pbest/1 (Vectorized)
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        
        # Sort to find p-best
        sorted_indices = np.argsort(fitness)
        
        # p-best selection: random from top p% (p depends on random factor, usually 0.05 to 0.2)
        p = np.random.uniform(0.05, 0.2) # Dynamic p
        top_p_count = max(1, int(p * pop_size))
        
        # Assign pbest indices for each vector
        pbest_indices = np.random.randint(0, top_p_count, pop_size) # relative to sorted list
        pbest_indices = sorted_indices[pbest_indices] # convert to actual indices
        x_pbest = population[pbest_indices]

        # Select r1 (distinct from i)
        # We generate a permutation for r1
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Simple fix for collisions: usually rare enough in DE not to break logic, 
        # but strictly:
        collision_mask = (r1_indices == np.arange(pop_size))
        r1_indices[collision_mask] = (r1_indices[collision_mask] + 1) % pop_size
        x_r1 = population[r1_indices]

        # Select r2 (distinct from i and r1, from Union(Pop, Archive))
        # Archive management as numpy array for vectorization
        if len(archive) > 0:
            archive_np = np.array(archive)
            pop_and_archive = np.vstack((population, archive_np))
        else:
            pop_and_archive = population
            
        len_total = len(pop_and_archive)
        r2_indices = np.random.randint(0, len_total, pop_size)
        # Collision logic omitted for speed; F-dithering handles diversity naturally
        x_r2 = pop_and_archive[r2_indices]

        # Calculate Mutant Vectors (Vectorized Math)
        # shape: (pop_size, dim)
        diff_best = x_pbest - population
        diff_r1r2 = x_r1 - x_r2
        
        # Broadcasting F: (pop_size,) -> (pop_size, 1)
        F_col = f[:, np.newaxis]
        mutant = population + F_col * diff_best + F_col * diff_r1r2

        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        CR_col = cr[:, np.newaxis]
        
        # Ensure at least one parameter comes from mutant (j_rand)
        j_rand = np.random.randint(0, dim, pop_size)
        j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
        j_rand_mask[np.arange(pop_size), j_rand] = True
        
        mask = (rand_vals <= CR_col) | j_rand_mask
        trial_pop = np.where(mask, mutant, population)
        
        # Bound Constraints
        trial_pop = clip_to_bounds(trial_pop)

        # 5. Selection & Evaluation
        # We must iterate to evaluate func, but we track successful updates
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        succ_mask = np.zeros(pop_size, dtype=bool)
        diff_fitness = np.zeros(pop_size)

        for i in range(pop_size):
            if (time.time() - start_time) > max_time:
                 return global_best_val
            
            t_val = func(trial_pop[i])
            
            if t_val < fitness[i]:
                new_population[i] = trial_pop[i]
                new_fitness[i] = t_val
                succ_mask[i] = True
                diff_fitness[i] = fitness[i] - t_val
                
                # Update Global Best
                if t_val < global_best_val:
                    global_best_val = t_val
                    global_best_vec = trial_pop[i].copy()
                
                # Add overwritten particle to archive
                archive.append(population[i].copy())

        # Archive Maintenance (Remove random if too full)
        while len(archive) > pop_size:
            idx_remove = np.random.randint(0, len(archive))
            archive.pop(idx_remove)

        # 6. Update Memory (History)
        # Weighted Lehmer Mean based on fitness improvement amount
        if np.any(succ_mask):
            succ_f = f[succ_mask]
            succ_cr = cr[succ_mask]
            weights = diff_fitness[succ_mask]
            weight_sum = np.sum(weights)
            
            if weight_sum > 0:
                weights /= weight_sum
                
                # Mean for CR
                mean_cr = np.sum(weights * succ_cr)
                
                # Lehmer Mean for F
                f_num = np.sum(weights * (succ_f ** 2))
                f_den = np.sum(weights * succ_f)
                mean_f = f_num / f_den if f_den > 0 else 0.5

                # Update memory slot
                mem_M_cr[k_mem] = 0.5 * mem_M_cr[k_mem] + 0.5 * mean_cr # Using a learning rate
                mem_M_f[k_mem]  = 0.5 * mem_M_f[k_mem]  + 0.5 * mean_f
                
                k_mem = (k_mem + 1) % H

        # Apply new population
        population = new_population
        fitness = new_fitness
        
        # Check Convergence (Optional reset, but L-SHADE relies on convergence)
        # If std dev is tiny, we might break early to Polishing
        if np.mean(np.std(population, axis=0)) < 1e-12:
            break

    # --- Phase 2: Coordinate Descent Polishing (Local Search) ---
    # Use remaining time to refine the global best vector dimension by dimension.
    # This cleans up the final precision that stochastic DE often misses.
    
    current_best = global_best_vec.copy()
    current_score = global_best_val
    
    # Initial step size for local search (small percentage of bounds)
    step_size = (ub - lb) * 0.005 
    
    while (time.time() - start_time) < max_time and np.max(step_size) > 1e-9:
        improved_any = False
        
        for d in range(dim):
            if (time.time() - start_time) >= max_time:
                return current_score
                
            # Try positive step
            original_val = current_best[d]
            current_best[d] = np.clip(original_val + step_size[d], lb[d], ub[d])
            val = func(current_best)
            
            if val < current_score:
                current_score = val
                improved_any = True
                continue # Keep change, move to next dim
            
            # Try negative step
            current_best[d] = np.clip(original_val - step_size[d], lb[d], ub[d])
            val = func(current_best)
            
            if val < current_score:
                current_score = val
                improved_any = True
            else:
                # Revert
                current_best[d] = original_val
        
        if not improved_any:
            # Reduce step size if no improvement found in any dimension
            step_size *= 0.5
            
    return current_score
