#The best standard improvement over a SHADE-based algorithm, particularly for limited-time black-box optimization, is **L-SHADE** (Linear Success-History based Adaptive Differential Evolution).
#
#The key improvements in this implementation over the previous code are:
#1.  **LPSR (Linear Population Size Reduction):** The algorithm starts with a large population to explore the global landscape and linearly reduces the population size over time to force convergence and exploit the best found basin. This aligns the search pressure perfectly with the `max_time` constraint.
#2.  **External Archive:** It maintains an archive of inferior solutions recently replaced. These are used in the mutation strategy to preserve diversity and prevent the population from collapsing into a local optimum too early.
#3.  **Vectorized Trial Generation:** The heavy math (mutation and crossover) is vectorized using NumPy for maximum speed, performing loops only where necessary (function evaluation).
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    L-SHADE implementation (Linear Success-History Adaptive Differential Evolution).
    Features:
    - Linear Population Size Reduction (LPSR) based on time.
    - External Archive for diversity maintenance.
    - Adaptive F and CR parameters based on historical success.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Initial population size: Start high for exploration
    # Typical L-SHADE uses N_init = 18 * dim, but we cap it for very high dims/slow funcs
    pop_size_init = int(max(30, min(500, 18 * dim)))
    pop_size_min = 4
    
    # LPSR: Population size will decrease linearly to this value
    pop_size = pop_size_init
    
    # Memory for adaptive parameters
    h_mem = 5  # Memory size
    mem_cr = np.full(h_mem, 0.5)
    mem_f = np.full(h_mem, 0.5)
    k_mem = 0
    
    # Archive parameters
    archive_rate = 2.0
    archive = [] # Stores solution vectors
    
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_idx = -1
    
    # Initial evaluation
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            # Return best found so far if time runs out during init
            return best_val if best_val != float('inf') else func(pop[0])
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Loop ---
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            return best_val
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on time elapsed
        # We target reaching min size slightly before max_time (95%) to allow final polishing
        progress = elapsed / max_time
        target_size = int(round((pop_size_min - pop_size_init) * (progress / 0.95) + pop_size_init))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            # Sort by fitness and truncate the worst individuals
            sort_indices = np.argsort(fitness)
            n_survivors = target_size
            
            # Keep top n_survivors
            pop = pop[sort_indices[:n_survivors]]
            fitness = fitness[sort_indices[:n_survivors]]
            
            # Update Archive size limit
            current_archive_cap = int(target_size * archive_rate)
            if len(archive) > current_archive_cap:
                # Randomly remove elements if archive is too big
                del_count = len(archive) - current_archive_cap
                # Simply truncate for speed, or remove random
                archive = archive[:current_archive_cap]
                
            pop_size = target_size
            # Recalculate best index after sort/resize
            best_idx = np.argmin(fitness)
            best_val = fitness[best_idx]

        # 2. Parameter Adaptation
        # Generate CR and F for each individual
        r_idx = np.random.randint(0, h_mem, pop_size)
        
        # CR: Normal distribution ~ N(mem_cr, 0.1)
        cr = np.random.normal(mem_cr[r_idx], 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        # Fix CR = -1 (not possible with clip 0,1 but good practice in SHADE) to 0
        
        # F: Cauchy distribution ~ C(mem_f, 0.1)
        # Cauchy can generate negative or > 1 values
        f = mem_f[r_idx] + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F boundaries
        # If F > 1, cap at 1. If F <= 0, regenerate until > 0
        too_small = f <= 0
        while np.any(too_small):
            f[too_small] = mem_f[r_idx][too_small] + 0.1 * np.random.standard_cauchy(np.sum(too_small))
            too_small = f <= 0
        f = np.clip(f, 0.0, 1.0) # Standard clip for upper bound

        # 3. Mutation: current-to-pbest/1
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        
        # Sort to find p-best
        sorted_indices = np.argsort(fitness)
        
        # p-best selection size (top 11% is standard for SHADE)
        p_num = max(2, int(pop_size * 0.11))
        p_best_indices = sorted_indices[:p_num]
        
        # Generate indices
        pbest_idxs = np.random.choice(p_best_indices, pop_size)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        
        # Ensure r1 != i
        # Vectorized correction is messy, simpler to loop or just accept rare collision in large pops.
        # Strict implementation:
        for i in range(pop_size):
            while r1_idxs[i] == i:
                r1_idxs[i] = np.random.randint(0, pop_size)
                
        # r2 selection: Union of Pop and Archive
        # We handle archive as a numpy array for vectorization
        if len(archive) > 0:
            arc_np = np.array(archive)
            union_pop = np.vstack((pop, arc_np))
        else:
            union_pop = pop
            
        r2_idxs = np.random.randint(0, len(union_pop), pop_size)
        
        # Ensure r2 != i and r2 != r1
        for i in range(pop_size):
            while r2_idxs[i] == i or r2_idxs[i] == r1_idxs[i]:
                r2_idxs[i] = np.random.randint(0, len(union_pop))

        # Calculate Mutation Vectors
        x_curr = pop
        x_pbest = pop[pbest_idxs]
        x_r1 = pop[r1_idxs]
        x_r2 = union_pop[r2_idxs]
        
        # F needs to be reshaped for broadcasting (pop_size, 1)
        f_col = f.reshape(-1, 1)
        
        mutant = x_curr + f_col * (x_pbest - x_curr) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover: Binomial
        j_rand = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        
        # Ensure at least one dimension is taken from mutant
        # We can't easily use 'arange' vectorization for setting j_rand per row,
        # but the boolean mask update is fast.
        for i in range(pop_size):
            mask[i, j_rand[i]] = True
            
        trial = np.where(mask, mutant, pop)
        
        # Bound Constraints
        trial = np.clip(trial, min_b, max_b)
        
        # 5. Selection
        succ_scr = []
        succ_sf = []
        diff_fitness = []
        
        # Store for archive update
        archive_candidates = []
        
        for i in range(pop_size):
            # Strict time check inside generation loop
            if time.time() - start_time >= max_time:
                return best_val
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Improvement
                if f_trial < fitness[i]:
                    # Only store memory updates if strictly better
                    succ_scr.append(cr[i])
                    succ_sf.append(f[i])
                    diff_fitness.append(fitness[i] - f_trial)
                    
                    # Add parent to archive candidates
                    archive_candidates.append(pop[i].copy())
                
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i
        
        # 6. Update Archive
        if len(archive_candidates) > 0:
            for cand in archive_candidates:
                if len(archive) < int(pop_size * archive_rate):
                    archive.append(cand)
                else:
                    # Random replacement
                    idx = np.random.randint(0, len(archive))
                    archive[idx] = cand

        # 7. Update Memory (Weighted Lehmer Mean)
        if len(succ_scr) > 0:
            succ_scr = np.array(succ_scr)
            succ_sf = np.array(succ_sf)
            diff_fitness = np.array(diff_fitness)
            
            # Weights based on fitness improvement
            total_diff = np.sum(diff_fitness)
            if total_diff > 0:
                weights = diff_fitness / total_diff
                
                # Update Mem CR
                mean_cr = np.sum(weights * succ_scr)
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                
                # Update Mem F (Lehmer Mean)
                mean_f_num = np.sum(weights * (succ_sf ** 2))
                mean_f_den = np.sum(weights * succ_sf)
                if mean_f_den > 0:
                    mean_f = mean_f_num / mean_f_den
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                
                k_mem = (k_mem + 1) % h_mem

    return best_val
