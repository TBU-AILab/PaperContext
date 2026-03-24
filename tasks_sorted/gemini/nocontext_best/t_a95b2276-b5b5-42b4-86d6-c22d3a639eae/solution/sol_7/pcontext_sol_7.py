#Here is the improved algorithm.
#
#**Improvements over the previous version:**
#1.  **SHADE Architecture (Success-History Adaptation):** Unlike JADE (which uses a single running mean), this implementation uses a historical memory ($H=5$) for control parameters $F$ and $Cr$. This allows the algorithm to learn multimodal adaptation strategies for complex landscapes.
#2.  **External Archive:** It implements a diversity archive. When solutions are overwritten in the population, they are moved to an archive. The mutation strategy selects the second difference vector ($r2$) from the union of the Population and the Archive. This significantly boosts exploration and prevents premature convergence.
#3.  **Fully Vectorized Mutation:** The mutation and crossover steps are fully vectorized using NumPy, removing Python loops from the critical path (except for the objective function evaluation itself).
#4.  **Weighted Lehmer Mean:** The parameter updates use the theoretically superior weighted Lehmer mean, giving more weight to highly successful improvements.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History Adaptive Differential Evolution).
    
    Key Features:
    - External Archive for diversity maintenance.
    - Historical Memory for parameter adaptation (F and CR).
    - Current-to-pbest/1 mutation strategy.
    - Fully vectorized evolution steps for performance.
    """
    start_time = time.time()
    
    # --- hyperparameters ---
    # Population size: 18 * dim is a robust heuristic for SHADE
    pop_size = 18 * dim
    # Memory size for history adaptation
    H_capacity = 5 
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # We use random uniform initialization within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            # If time is up during init, return best found so far (or inf)
            if i > 0: return np.min(fitness[:i])
            return float('inf')
        fitness[i] = func(population[i])
        
    # Sort for convenience (though not strictly required at this step)
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # Best Solution Tracking
    best_idx = 0 # Since it's sorted
    global_best_val = fitness[best_idx]
    global_best_vec = population[best_idx].copy()

    # Memory Initialization
    # M_CR and M_F hold historical successful parameters
    # Initialized to 0.5 as a neutral starting point
    memory_cr = np.full(H_capacity, 0.5)
    memory_f = np.full(H_capacity, 0.5)
    k_mem = 0 # Index for memory update

    # Archive Initialization
    # Stores inferior solutions to maintain diversity in mutation
    archive = [] 
    max_archive_size = pop_size

    # Main Loop
    while True:
        # Time Check
        if (time.time() - start_time) >= max_time:
            return global_best_val

        # --- 1. Parameter Generation ---
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H_capacity, pop_size)
        m_cr = memory_cr[r_idx]
        m_f = memory_f[r_idx]

        # Generate CR ~ Normal(M_CR, 0.1)
        cr_g = np.random.normal(m_cr, 0.1)
        cr_g = np.clip(cr_g, 0, 1)
        
        # Generate F ~ Cauchy(M_F, 0.1)
        # Cauchy generation: location + scale * tan(pi * (rand - 0.5))
        # Or using numpy's standard_cauchy
        f_g = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraint: if F > 1 set to 1, if F <= 0 regenerate
        # Vectorized correction for F <= 0
        while True:
            bad_f = f_g <= 0
            if not np.any(bad_f):
                break
            f_g[bad_f] = m_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
        
        f_g = np.clip(f_g, 0, 1) # Clip upper bound

        # --- 2. Mutation (current-to-pbest/1) ---
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # Select p-best (top 5% to 20% random selection)
        p_min = 2 / pop_size
        p_val = np.random.uniform(p_min, 0.2) # Randomized p-value for robustness
        top_p_cnt = max(1, int(p_val * pop_size))
        
        # Indices for pbest
        # Since population is sorted by fitness at the end of loop, top is at beginning
        pbest_indices = np.random.randint(0, top_p_cnt, pop_size)
        x_pbest = population[pbest_indices]

        # Indices for r1 (random from population, r1 != i)
        # We simplify by choosing random and accepting small collision chance for speed
        # or performing a rotation. Standard DE implies distinct, but in large Pop it matters less.
        # Permutation is a fast way to ensure r1 != i implies mixing.
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_indices]

        # Indices for r2 (random from Population Union Archive)
        # Union population and archive
        if len(archive) > 0:
            archive_np = np.array(archive)
            pop_archive_pool = np.vstack((population, archive_np))
        else:
            pop_archive_pool = population
            
        pool_size = pop_archive_pool.shape[0]
        r2_indices = np.random.randint(0, pool_size, pop_size)
        x_r2 = pop_archive_pool[r2_indices]

        # Calculate Mutant Vectors (Vectorized)
        # reshape F for broadcasting: (pop_size, 1)
        F_b = f_g[:, np.newaxis]
        
        # V = X_curr + F*(X_pbest - X_curr) + F*(X_r1 - X_r2)
        # This formula guides towards best (exploitation) while using difference (exploration)
        mutants = population + F_b * (x_pbest - population) + F_b * (x_r1 - x_r2)
        
        # Boundary Handling (Clip)
        mutants = np.clip(mutants, min_b, max_b)

        # --- 3. Crossover (Binomial) ---
        # Generate crossover mask
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr_g[:, np.newaxis]
        
        # Ensure at least one dimension is taken from mutant (j_rand)
        j_rand = np.random.randint(0, dim, pop_size)
        # Advanced indexing to set the j_rand index to True
        row_indices = np.arange(pop_size)
        cross_mask[row_indices, j_rand] = True
        
        trials = np.where(cross_mask, mutants, population)

        # --- 4. Selection ---
        # We must evaluate trials sequentially because 'func' might not be vectorized
        # We collect successful parameters
        
        trial_fitness = np.zeros(pop_size)
        
        # Containers for successful updates
        success_indices = []
        success_diff = [] # fitness improvement amount
        
        # Optimized Loop for Evaluation
        for i in range(pop_size):
            # Check time strictly inside the loop
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            f_trial = func(trials[i])
            trial_fitness[i] = f_trial
            
            # Greedy Selection
            if f_trial <= fitness[i]: # Better or equal
                # Store improvement for weighted mean
                diff = fitness[i] - f_trial
                
                # If strictly better, mark for archive and memory update
                if f_trial < fitness[i]:
                    # Add parent to archive
                    archive.append(population[i].copy())
                    success_indices.append(i)
                    success_diff.append(diff)
                
                # Update population immediately
                fitness[i] = f_trial
                population[i] = trials[i]
                
                # Update global best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trials[i].copy()

        # --- 5. Archive Maintenance ---
        # If archive exceeds size, remove random elements
        while len(archive) > max_archive_size:
            # Remove random elements to maintain diversity
            rem_idx = np.random.randint(0, len(archive))
            archive.pop(rem_idx)

        # --- 6. Memory Update (Adaptive Logic) ---
        if len(success_indices) > 0:
            # Get successful F and CR
            s_f = f_g[success_indices]
            s_cr = cr_g[success_indices]
            s_diff = np.array(success_diff)
            
            # Weighted Lehmer Mean for F (biases towards high-impact updates)
            # Weights based on fitness improvement
            weights = s_diff / (np.sum(s_diff) + 1e-15)
            
            # Lehmer mean: sum(w * x^2) / sum(w * x)
            mean_lehmer_f = np.sum(weights * (s_f ** 2)) / (np.sum(weights * s_f) + 1e-15)
            
            # Weighted Arithmetic Mean for CR
            mean_arith_cr = np.sum(weights * s_cr)
            
            # Update Memory at pointer k
            # Using a learning rate for the memory (e.g., 0.5 blend with old, or strict replacement)
            # SHADE usually strict replacement of the slot
            memory_f[k_mem] = mean_lehmer_f
            memory_cr[k_mem] = mean_arith_cr
            
            # Increment pointer
            k_mem = (k_mem + 1) % H_capacity

        # --- 7. Sorting & Convergence Check ---
        # Sort population to ensure p-best logic works in next iter
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Soft Restart: If std dev of fitness is tiny, we are stuck.
        # However, SHADE with archive is resilient. 
        # Only restart worst 50% if extremely converged.
        if np.std(fitness) < 1e-9:
             # Keep top half, randomize bottom half to re-inject energy
             half_pt = pop_size // 2
             population[half_pt:] = min_b + np.random.rand(pop_size - half_pt, dim) * diff_b
             # Reset fitness for these to ensure they get evaluated or treated as high
             # (In next loop, they will be parents, so their fitness will be compared)
             # To force evaluation, we can set their fitness to inf, 
             # but we need valid fitness for difference calc. 
             # Re-evaluating takes time. Just mutating them is cheaper.
             # Let's simple-mutate them using the Best 1
             population[half_pt:] = global_best_vec + (np.random.rand(pop_size - half_pt, dim) - 0.5) * diff_b * 0.1
             # We must re-evaluate these to keep arrays valid
             for k in range(half_pt, pop_size):
                 if (time.time() - start_time) >= max_time: return global_best_val
                 fitness[k] = func(population[k])
             
             # Re-sort
             sorted_idx = np.argsort(fitness)
             population = population[sorted_idx]
             fitness = fitness[sorted_idx]

    return global_best_val
