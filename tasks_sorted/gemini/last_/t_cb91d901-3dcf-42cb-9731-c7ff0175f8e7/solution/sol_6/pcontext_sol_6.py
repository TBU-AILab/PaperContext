#Here is a fully operational, self-contained Python algorithm implementing **L-SHADE** (Linear Success-History based Adaptive Differential Evolution) with vectorized operations and Linear Population Size Reduction.
#
##### Improvement Logic
#1.  **Vectorization**: The previous implementation used Python loops for mutation and crossover. This version uses NumPy vector operations to generate the entire population of trial vectors at once, significantly reducing overhead and increasing the number of evaluations per second.
#2.  **Boundary Handling**: Instead of simple clipping (which reduces diversity at the edges), this implementation uses a "midpoint bounce-back" strategy (common in CEC competition winners like jSO), placing particles between the boundary and the previous position if they violate bounds.
#3.  **LPSR (Linear Population Size Reduction)**: The population size linearly decreases from a high initial value to a minimum, ensuring global search at the beginning and fast convergence at the end.
#4.  **Robust Initialization**: Uses standard uniform initialization but calculates initial population size based on dimension `dim` (capped for performance) to ensure adequate coverage.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Linear Population Size Reduction (LPSR) and Vectorization.
    
    Optimized for high-dimensional and expensive black-box functions within a time limit.
    """
    # --- Configuration & Constants ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Extract bounds
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    
    # 1. Initialization Parameters
    # Initial population size: 18 * dim is a strong baseline (L-SHADE literature)
    # We cap it to prevent sluggishness in very high dimensions.
    N_init = int(round(max(20, 18 * dim)))
    if N_init > 200: 
        N_init = 200
    
    N_min = 4
    pop_size = N_init
    
    # Archive parameters
    archive = [] # List of numpy arrays
    arc_rate = 2.6
    
    # Memory for adaptive parameters (History length H=6)
    H = 6
    M_cr = np.full(H, 0.5)
    M_f = np.full(H, 0.5)
    k_mem = 0
    
    # --- Initial Population ---
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    
    # Evaluate Initial Population
    # We perform a check inside the loop to ensure we don't timeout during init
    for i in range(pop_size):
        if (datetime.now() - start_time) >= limit:
            return best_fitness
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            
    # --- Main Optimization Loop ---
    while True:
        # Time Check
        elapsed = datetime.now() - start_time
        if elapsed >= limit:
            break
            
        # Calculate Progress (0.0 -> 1.0)
        progress = elapsed.total_seconds() / max_time
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target population size
        next_pop_size = int(round(((N_min - N_init) * progress) + N_init))
        next_pop_size = max(N_min, next_pop_size)
        
        if pop_size > next_pop_size:
            # Sort by fitness (best to worst)
            sort_idx = np.argsort(fitness)
            # Keep the top 'next_pop_size' individuals
            keep_idx = sort_idx[:next_pop_size]
            pop = pop[keep_idx]
            fitness = fitness[keep_idx]
            pop_size = next_pop_size
            
            # Reduce Archive Size accordingly
            target_arc_size = int(pop_size * arc_rate)
            if len(archive) > target_arc_size:
                # Randomly remove excess
                # Using shuffle is robust
                random.shuffle(archive)
                archive = archive[:target_arc_size]

        # 2. Adaptive Parameter Generation (Vectorized)
        # Pick random memory slot for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = M_cr[r_idx]
        m_f = M_f[r_idx]
        
        # Generate Crossover Rates (Cr) ~ Normal(M_cr, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate Mutation Factors (F) ~ Cauchy(M_f, 0.1)
        # We need to handle F <= 0 (regenerate) and F > 1 (clip to 1)
        f = np.zeros(pop_size)
        todo_mask = np.ones(pop_size, dtype=bool)
        
        while np.any(todo_mask):
            n_todo = np.sum(todo_mask)
            # Cauchy generation: loc + scale * tan(pi * (rand - 0.5))
            f_gen = m_f[todo_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_todo) - 0.5))
            
            # Valid F must be > 0
            valid = f_gen > 0
            
            # Map valid results back to the main array
            # Get indices of the rows we are currently fixing
            curr_indices = np.where(todo_mask)[0]
            good_indices = curr_indices[valid]
            
            f[good_indices] = f_gen[valid]
            
            # Update mask (disable rows that are now valid)
            todo_mask[good_indices] = False
            
        f[f > 1] = 1.0
        
        # 3. Mutation Strategy: current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Sort population for p-best selection
        sorted_indices = np.argsort(fitness)
        
        # p varies from 0.2 (exploration) to 2/N (exploitation) based on progress
        p_val = max(2.0/pop_size, 0.2 * (1 - progress))
        p_count = int(round(p_val * pop_size))
        p_count = max(2, p_count)
        
        # Select p-best indices
        pbest_pool = sorted_indices[:p_count]
        pbest_idx = np.random.choice(pbest_pool, pop_size)
        
        # Select r1 (distinct from current i)
        r1_idx = np.random.randint(0, pop_size, pop_size)
        # Fix collisions (r1 == i)
        confl = (r1_idx == np.arange(pop_size))
        while np.any(confl):
            r1_idx[confl] = np.random.randint(0, pop_size, np.sum(confl))
            confl = (r1_idx == np.arange(pop_size))
            
        # Select r2 (distinct from i and r1, from Pop U Archive)
        n_arc = len(archive)
        if n_arc > 0:
            # Create combined population for indexing
            arr_arc = np.array(archive)
            pop_all = np.vstack((pop, arr_arc))
        else:
            pop_all = pop
            
        n_all = len(pop_all)
        r2_idx = np.random.randint(0, n_all, pop_size)
        
        # Fix collisions (r2 == i or r2 == r1)
        # Note: i and r1 are indices in 'pop', r2 is index in 'pop_all'.
        # We only check collision if r2 points to a member of 'pop' (i.e. < pop_size)
        c1 = (r2_idx == np.arange(pop_size))
        c2 = (r2_idx == r1_idx)
        confl = c1 | c2
        while np.any(confl):
            r2_idx[confl] = np.random.randint(0, n_all, np.sum(confl))
            c1 = (r2_idx == np.arange(pop_size))
            c2 = (r2_idx == r1_idx)
            confl = c1 | c2
            
        # Create Mutant Vectors
        x = pop
        x_pbest = pop[pbest_idx]
        x_r1 = pop[r1_idx]
        x_r2 = pop_all[r2_idx]
        
        f_v = f[:, None] # Reshape for broadcasting
        mutant = x + f_v * (x_pbest - x) + f_v * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals < cr[:, None]
        # Ensure at least one dimension is taken from mutant
        mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask, mutant, x)
        
        # 5. Bound Handling (Midpoint Correction)
        # If out of bounds, place halfway between limit and previous position
        # This preserves diversity better than clipping.
        
        # Lower bound violations
        below = trial < lb
        if np.any(below):
            # We access lb via broadcasting or indexing. lb is (dim,)
            # We need the specific column index for the correct lb value.
            # np.where(below) returns (row_indices, col_indices)
            cols = np.where(below)[1]
            trial[below] = (lb[cols] + x[below]) / 2.0
            
        # Upper bound violations
        above = trial > ub
        if np.any(above):
            cols = np.where(above)[1]
            trial[above] = (ub[cols] + x[above]) / 2.0
            
        # Final clip to ensure precision errors don't leave us out of bounds
        trial = np.clip(trial, lb, ub)
        
        # 6. Evaluation and Selection
        trial_fitness = np.zeros(pop_size)
        
        # Loop for evaluation (func is black-box, usually not vectorized)
        for i in range(pop_size):
            # Check time periodically (every 10 evals) to ensure responsiveness
            if (i % 10 == 0) and (datetime.now() - start_time) >= limit:
                return best_fitness
            
            tf = func(trial[i])
            trial_fitness[i] = tf
            
            if tf < best_fitness:
                best_fitness = tf

        # 7. Update Archive and Memory
        # Identify improvements
        imp = trial_fitness < fitness
        
        if np.any(imp):
            # -- Archive Update --
            # Add parents that are about to be replaced to the archive
            old_parents = pop[imp]
            for op in old_parents:
                archive.append(op.copy())
            
            # Maintain Archive Size
            curr_arc_limit = int(pop_size * arc_rate)
            if len(archive) > curr_arc_limit:
                # If significantly over, shuffle and slice
                if len(archive) > curr_arc_limit + 50:
                    random.shuffle(archive)
                    archive = archive[:curr_arc_limit]
                else:
                    # Randomly remove elements until size is met
                    while len(archive) > curr_arc_limit:
                        idx = random.randint(0, len(archive)-1)
                        archive.pop(idx)
            
            # -- Memory Update --
            # Calculate fitness difference
            diff = np.abs(fitness[imp] - trial_fitness[imp])
            
            # Normalize weights based on improvement magnitude
            w = diff / np.sum(diff)
            
            s_f = f[imp]
            s_cr = cr[imp]
            
            # Weighted Lehmer Mean for F
            mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
            
            # Weighted Arithmetic Mean for Cr
            mean_cr = np.sum(w * s_cr)
            
            # Update Memory (Ring Buffer)
            M_f[k_mem] = mean_f
            M_cr[k_mem] = mean_cr
            k_mem = (k_mem + 1) % H
            
            # -- Population Update --
            pop[imp] = trial[imp]
            fitness[imp] = trial_fitness[imp]

    return best_fitness
