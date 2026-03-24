#Here is a fully operational, self-contained Python implementation of the **SHADE (Success-History based Adaptive Differential Evolution)** algorithm. 
#
#This algorithm improves upon the previous submission by implementing:
#1.  **Historical Memory**: It stores successful $F$ and $CR$ parameters to adaptively guide the generation of new parameters, rather than simple random resetting.
#2.  **External Archive**: It maintains an archive of inferior solutions to maintain diversity in the difference vector generation, significantly boosting exploration.
#3.  **LHS Initialization**: It uses Latin Hypercube Sampling for the initial population to cover the search space more evenly.
#4.  **Robust Restart**: It detects convergence stagnation using standard deviation and performs a soft restart while preserving the global best.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution).
    """
    start_time = time.time()

    # --- Pre-process Bounds ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b

    # --- Configuration ---
    # Population size: Adapted for efficiency within limited time
    # SHADE typically uses ~18*dim, but we cap it to ensure enough generations run.
    pop_size = int(max(30, min(150, 18 * dim)))

    # Memory configuration for adaptive parameters
    # H: History size
    H = 5
    mem_M_F = np.full(H, 0.5) # Memory for Mutation Factor F
    mem_M_CR = np.full(H, 0.5) # Memory for Crossover Rate CR
    k_mem = 0 # Memory index pointer

    # External Archive (A) to maintain diversity
    archive = []
    
    # --- Initialization (Latin Hypercube Sampling) ---
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        # Generate stratified samples
        points = (np.arange(pop_size) + np.random.rand(pop_size)) / pop_size
        np.random.shuffle(points)
        pop[:, d] = min_b[d] + diff_b[d] * points

    fitness = np.zeros(pop_size)
    best_val = float('inf')
    best_vec = np.zeros(dim)

    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            # If time runs out immediately, return best found so far
            return best_val if best_val != float('inf') else func(pop[i])
        
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()

    # --- Main Optimization Loop ---
    while True:
        # Global Time Check
        if (time.time() - start_time) >= max_time:
            return best_val

        # 1. Parameter Adaptation
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_F = mem_M_F[r_idx]
        mu_CR = mem_M_CR[r_idx]

        # Generate CR: Normal(mu_CR, 0.1), clipped to [0, 1]
        CR = np.random.normal(mu_CR, 0.1)
        CR = np.clip(CR, 0, 1)

        # Generate F: Cauchy(mu_F, 0.1), clipped to (0, 1]
        # Cauchy generation: loc + scale * tan(pi * (rand - 0.5))
        # If F <= 0, regenerate. If F > 1, clamp to 1.
        rand_c = np.random.rand(pop_size)
        F = mu_F + 0.1 * np.tan(np.pi * (rand_c - 0.5))
        
        # Repair F values
        for i in range(pop_size):
            while F[i] <= 0:
                F[i] = mu_F[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            if F[i] > 1:
                F[i] = 1.0

        # 2. Mutation Strategy: current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        
        # Select p-best from top 11% (standard SHADE heuristic)
        p_rate = 0.11
        num_pbest = max(2, int(pop_size * p_rate))
        top_p_indices = sorted_indices[:num_pbest]
        
        pbest_indices = np.random.choice(top_p_indices, pop_size)
        x_pbest = pop[pbest_indices]

        # Select r1 (distinct from current index i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        for i in range(pop_size):
            if r1_indices[i] == i:
                r1_indices[i] = (r1_indices[i] + 1) % pop_size
        x_r1 = pop[r1_indices]

        # Select r2 (distinct from i and r1, from Union of Pop and Archive)
        if len(archive) > 0:
            arr_archive = np.array(archive)
            pop_all = np.vstack((pop, arr_archive))
        else:
            pop_all = pop
        
        len_all = len(pop_all)
        r2_indices = np.random.randint(0, len_all, pop_size)
        
        # Collision handling for r2
        for i in range(pop_size):
            # We need r2 != i (if r2 is in pop) and r2 != r1
            while (r2_indices[i] < pop_size and r2_indices[i] == i) or r2_indices[i] == r1_indices[i]:
                r2_indices[i] = np.random.randint(0, len_all)
        x_r2 = pop_all[r2_indices]

        # Compute mutation vectors
        diff_pbest = x_pbest - pop
        diff_r1_r2 = x_r1 - x_r2
        F_col = F.reshape(-1, 1)
        
        mutant = pop + F_col * diff_pbest + F_col * diff_r1_r2

        # 3. Crossover (Binomial)
        # mask = rand < CR
        mask = np.random.rand(pop_size, dim) < CR.reshape(-1, 1)
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        for i in range(pop_size):
            mask[i, j_rand[i]] = True
        
        trial = np.where(mask, mutant, pop)

        # 4. Bound Constraints (Clipping)
        trial = np.clip(trial, min_b, max_b)

        # 5. Selection and Memory Update Prep
        winning_F = []
        winning_CR = []
        winning_diff = []

        # Evaluate Trial Vectors
        for i in range(pop_size):
            # Granular time check
            if (time.time() - start_time) >= max_time:
                return best_val

            f_trial = func(trial[i])

            # Greedy Selection
            if f_trial <= fitness[i]:
                # If improvement, calculate magnitude
                diff = fitness[i] - f_trial
                
                # If strictly better, store parameter success data
                if f_trial < fitness[i]:
                    # Add parent to archive before replacing
                    archive.append(pop[i].copy())
                    winning_F.append(F[i])
                    winning_CR.append(CR[i])
                    winning_diff.append(diff)

                pop[i] = trial[i]
                fitness[i] = f_trial

                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial[i].copy()

        # Maintain Archive Size (Clamp to pop_size)
        while len(archive) > pop_size:
            idx_rem = np.random.randint(0, len(archive))
            archive.pop(idx_rem)

        # 6. Update Memory (Weighted Lehmer Mean)
        if len(winning_diff) > 0:
            w_diff = np.array(winning_diff)
            w_F = np.array(winning_F)
            w_CR = np.array(winning_CR)

            total_diff = np.sum(w_diff)
            if total_diff > 0:
                weights = w_diff / total_diff
                
                # Weighted Lehmer Mean for F
                mean_F_L = np.sum(weights * (w_F ** 2)) / np.sum(weights * w_F)
                
                # Weighted Mean for CR
                mean_CR_W = np.sum(weights * w_CR)
                
                # Update current memory slot
                mem_M_F[k_mem] = mean_F_L
                mem_M_CR[k_mem] = mean_CR_W
                
                # Move pointer
                k_mem = (k_mem + 1) % H

        # 7. Restart Mechanism
        # If population diversity is extremely low, restart to escape local optima
        if np.std(fitness) < 1e-9:
            # Keep the best, re-initialize the rest
            pop = min_b + diff_b * np.random.rand(pop_size, dim)
            pop[0] = best_vec
            fitness[:] = float('inf')
            fitness[0] = best_val
            
            # Reset Archive and Memory
            archive = []
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            
            # Re-evaluate new population (skipping index 0)
            for i in range(1, pop_size):
                if (time.time() - start_time) >= max_time: return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()

    return best_val
