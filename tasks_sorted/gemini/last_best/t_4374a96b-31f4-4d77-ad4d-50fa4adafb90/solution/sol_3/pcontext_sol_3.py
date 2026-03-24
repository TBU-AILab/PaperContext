Here is the improved, fully operational, and self-contained algorithm.

# This implementation uses a **Restart SHADE (Success-History based Adaptive Differential Evolution)** algorithm.
# It improves upon previous versions by:
# 1. Using **'current-to-pbest/1'** mutation, which balances exploitation of the best solutions with exploration.
# 2. Implementing **History-based Parameter Adaptation**, which learns the optimal scale factor (F) and crossover rate (CR) for the specific function landscape over time.
# 3. Utilizing **Latin Hypercube Sampling** for restarts to maximize coverage.
# 4. Employing strict vectorized operations for maximum throughput.

import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Restart SHADE (Success-History based Adaptive DE).
    """
    start_time = time.time()
    # Buffer to ensure clean exit
    limit_time = max_time - 0.05

    # --- Pre-computation ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # SHADE Parameters
    # History size for memory
    H = 6 
    # Population size: 15*dim is a good balance for SHADE, clamped for efficiency
    pop_size = int(np.clip(15 * dim, 20, 150))
    # p-best factor (top 5% - 20%)
    p_best_rate = 0.11 
    
    global_best_fit = float('inf')
    
    # Helper for random number generation
    rng = np.random.default_rng()

    # --- Main Loop (Restarts) ---
    while True:
        if (time.time() - start_time) > limit_time:
            return global_best_fit

        # 1. Initialization: Latin Hypercube Sampling
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
            pop[:, d] = rng.uniform(edges[:-1], edges[1:])
        
        # Shuffle dimensions to break correlations in initialization
        for d in range(dim):
            rng.shuffle(pop[:, d])

        # Evaluate initial population
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if (time.time() - start_time) > limit_time:
                return global_best_fit
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_fit:
                global_best_fit = val

        # Initialize Memory for Adaptive Parameters
        # M_CR and M_F hold historical successful means
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k_mem = 0  # Memory index

        # 2. Evolutionary Loop
        while True:
            if (time.time() - start_time) > limit_time:
                return global_best_fit
            
            # Check for stagnation/convergence to trigger restart
            # If population variance is extremely low, we are stuck
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break

            # --- Parameter Adaptation ---
            # Select random memory index for each individual
            r_idx = rng.integers(0, H, pop_size)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]

            # Generate CR (Normal distribution, clipped [0, 1])
            CR = rng.normal(m_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            # Fix CR=0 to small epsilon to ensure some crossover
            CR[CR == 0] = 1e-6

            # Generate F (Cauchy distribution)
            # Cauchy(loc, scale) = loc + scale * tan(pi * (rand - 0.5))
            # Or use standard_cauchy: loc + scale * standard_cauchy
            F = m_f + 0.1 * rng.standard_cauchy(pop_size)
            
            # Handle F constraints: If F > 1 clip to 1, if F <= 0 regenerate
            # Vectorized correction for F <= 0
            while np.any(F <= 0):
                mask = F <= 0
                F[mask] = m_f[mask] + 0.1 * rng.standard_cauchy(np.sum(mask))
            F = np.clip(F, 0.0, 1.0)

            # --- Mutation: DE/current-to-pbest/1 ---
            # V = X_i + F*(X_pbest - X_i) + F*(X_r1 - X_r2)
            
            # 1. Identify p-best for each individual
            # Sort population by fitness to find top p_best_rate
            sorted_indices = np.argsort(fitness)
            num_pbest = max(2, int(p_best_rate * pop_size))
            top_indices = sorted_indices[:num_pbest]
            
            # Randomly select one pbest for each individual
            pbest_indices = rng.choice(top_indices, pop_size)
            x_pbest = pop[pbest_indices]

            # 2. Select r1 and r2
            # Generate random indices
            # We need r1 != r2 != i. 
            # Efficient approach: generate 3 random columns, pick distinct ones
            rand_indices = rng.integers(0, pop_size, (pop_size, 3))
            
            # Adjust indices to ensure uniqueness relative to 'i' (row index) is harder strictly vectorized, 
            # but collisions are rare in reasonably sized populations. 
            # We strictly enforce r1 != r2 using shifting.
            r1 = rand_indices[:, 0]
            r2 = rand_indices[:, 1]
            
            # If r1 == i, shift
            collision_r1 = (r1 == np.arange(pop_size))
            r1[collision_r1] = (r1[collision_r1] + 1) % pop_size
            
            # If r2 == i or r2 == r1, shift
            collision_r2 = (r2 == np.arange(pop_size)) | (r2 == r1)
            r2[collision_r2] = (r2[collision_r2] + 2) % pop_size
            
            x_r1 = pop[r1]
            x_r2 = pop[r2]

            # Compute mutant vectors
            # Reshape F for broadcasting
            F_col = F[:, None]
            
            # Mutation equation
            # V = Current + F(pBest - Current) + F(r1 - r2)
            mutants = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # Bound Handling (Clip)
            mutants = np.clip(mutants, min_b, max_b)

            # --- Crossover (Binomial) ---
            rand_vals = rng.random((pop_size, dim))
            cross_mask = rand_vals < CR[:, None]
            
            # Ensure at least one variable is taken from mutant
            j_rand = rng.integers(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trials = np.where(cross_mask, mutants, pop)

            # --- Selection & Memory Update ---
            # To update memory, we need to track successful F and CR
            successful_F = []
            successful_CR = []
            improvement_diff = []
            
            for i in range(pop_size):
                if (time.time() - start_time) > limit_time:
                    return global_best_fit
                
                f_trial = func(trials[i])
                
                if f_trial <= fitness[i]:
                    # Improvement or equal
                    diff = fitness[i] - f_trial
                    
                    # Store for memory update if strictly better
                    if diff > 0:
                        successful_F.append(F[i])
                        successful_CR.append(CR[i])
                        improvement_diff.append(diff)
                    
                    fitness[i] = f_trial
                    pop[i] = trials[i]
                    
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial

            # --- Update History Memory ---
            if len(successful_F) > 0:
                s_F = np.array(successful_F)
                s_CR = np.array(successful_CR)
                diffs = np.array(improvement_diff)
                
                # Weighted Lehmer Mean
                # Weights based on fitness improvement amount
                total_diff = np.sum(diffs)
                if total_diff > 0:
                    weights = diffs / total_diff
                    
                    # Update M_CR (Weighted Mean)
                    mean_cr = np.sum(weights * s_CR)
                    M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr # Slight smoothing
                    
                    # Update M_F (Weighted Lehmer Mean: sum(w*f^2) / sum(w*f))
                    mean_f_numer = np.sum(weights * (s_F ** 2))
                    mean_f_denom = np.sum(weights * s_F)
                    if mean_f_denom > 1e-9:
                        mean_f = mean_f_numer / mean_f_denom
                        M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                    
                    # Increment memory index
                    k_mem = (k_mem + 1) % H

    return global_best_fit