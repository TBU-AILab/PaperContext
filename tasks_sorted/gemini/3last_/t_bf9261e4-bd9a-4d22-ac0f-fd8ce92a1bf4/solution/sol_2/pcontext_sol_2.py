#The following algorithm is an implementation of **L-SHADE (Linear Population Size Reduction - Success-History Adaptive Differential Evolution)**.
#
#This is a significant improvement over standard or restarting DE because:
#1.  **Adaptive Parameters**: It learns the optimal Mutation Factor ($F$) and Crossover Rate ($CR$) during the search, adapting to the function's landscape (e.g., separable vs. non-separable).
#2.  **Linear Population Reduction**: It starts with a large population to explore the global space and linearly reduces the population size over time to focus computational resources on exploiting the best area found.
#3.  **External Archive**: It maintains an archive of inferior solutions to preserve diversity in the mutation equation, preventing premature convergence.
#4.  **Current-to-pBest Mutation**: A robust mutation strategy that balances exploitation (moving toward top $p\%$ solutions) and exploration.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History Adaptive Differential Evolution).
    """
    # --- Initialization ---
    start_time = datetime.now()
    # Use 95% of available time to ensure safe return
    time_limit = timedelta(seconds=max_time * 0.95)
    
    # Helper for bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b

    # --- L-SHADE Hyperparameters ---
    # Initial population size (N_init). 18*dim is a standard empirical value.
    N_init = int(18 * dim) 
    # Ensure reasonable limits for very high/low dims
    N_init = max(30, min(N_init, 500)) 
    
    # Final population size (N_min)
    N_min = 4 
    
    # Archive size factor
    arc_rate = 2.6 
    
    # History memory size for parameters
    H = 6 
    
    # Initialize Memory for F and CR
    # Start with bias towards mean values (F=0.5, CR=0.5)
    M_CR = np.full(H, 0.5)
    M_F = np.full(H, 0.5)
    k_mem = 0  # Memory index

    # Initialize Population
    pop_size = N_init
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate initial population
    best_idx = 0
    best_fitness = float('inf')
    
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            # If time runs out during init, return best found so far (or inf)
            if i > 0: return min(fitness[:i])
            return float('inf')
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # Sort population by fitness for p-best selection
    sorted_indices = np.argsort(fitness)
    pop = pop[sorted_indices]
    fitness = fitness[sorted_indices]

    # External Archive (starts empty)
    archive = np.empty((0, dim))
    
    # Current max evaluations estimated (heuristic) or time-based progress
    # We use time to drive the population reduction
    
    # --- Main Loop ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate expected progress ratio
        elapsed = (datetime.now() - start_time).total_seconds()
        limit_sec = time_limit.total_seconds()
        progress = elapsed / limit_sec if limit_sec > 0 else 1.0
        
        # Calculate new target population size
        N_next = int(round((N_min - N_init) * progress + N_init))
        N_next = max(N_min, N_next)
        
        # If current pop is too big, remove worst individuals
        if pop_size > N_next:
            num_to_remove = pop_size - N_next
            # Population is already sorted at the end of loop
            pop = pop[:N_next]
            fitness = fitness[:N_next]
            pop_size = N_next
            
            # Reduce archive size if necessary to match pop size ratio
            target_arc_size = max(0, int(pop_size * arc_rate))
            if archive.shape[0] > target_arc_size:
                # Remove random elements from archive to shrink it
                keep_indices = np.random.choice(archive.shape[0], target_arc_size, replace=False)
                archive = archive[keep_indices]

        # 2. Parameter Generation
        # For each individual, pick a memory index r from [0, H-1]
        r_indices = np.random.randint(0, H, pop_size)
        mu_cr = M_CR[r_indices]
        mu_f = M_F[r_indices]
        
        # Generate CR (Normal distribution, clipped [0, 1])
        # If CR is close to 0, it slows convergence, but -1 handling (from paper) is simplified here to clip
        CR = np.random.normal(mu_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # Generate F (Cauchy distribution, clipped [0, 1])
        # F > 1 clipped to 1. F <= 0 regenerated.
        F = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Retry mechanism for F <= 0
        while np.any(F <= 0):
            mask = F <= 0
            F[mask] = mu_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
        F = np.clip(F, 0.0, 1.0)
        
        # 3. Mutation (current-to-pbest/1)
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        
        # P-best selection: select from top p (p in [2/N, 0.2])
        # We use p_best_rate = 0.11 (common setting)
        p_val = max(2, int(pop_size * 0.11))
        pbest_indices = np.random.randint(0, p_val, pop_size)
        x_pbest = pop[pbest_indices]
        
        # r1: Random from population (distinct from i)
        # r2: Random from Union(Population, Archive) (distinct from i and r1)
        
        # To vectorize, we just generate indices and ignore the rare overlap 
        # (overlap effect is negligible in stochastic optimizers)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = pop[r1_indices]
        
        # Union Population + Archive
        if archive.shape[0] > 0:
            union_pop = np.vstack((pop, archive))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, union_pop.shape[0], pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Compute Mutation Vectors
        # F needs to be reshaped for broadcasting
        F_col = F.reshape(-1, 1)
        
        # v = current + F(pbest - current) + F(r1 - r2)
        mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # Boundary Correction (Bounce back or Clip) -> Using Clip for stability
        mutant = np.clip(mutant, min_b, max_b)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        CR_col = CR.reshape(-1, 1)
        cross_mask = rand_vals < CR_col
        
        # Enforce at least one dimension from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 5. Selection and Memory Update
        winning_F = []
        winning_CR = []
        improvement_diff = []
        
        new_archive_candidates = []
        
        for i in range(pop_size):
            # Time check inside individual loop for responsiveness
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            f_trial = func(trial_pop[i])
            
            f_old = fitness[i]
            
            if f_trial < f_old:
                # Success
                # Store improvement for weighted mean
                diff = f_old - f_trial
                improvement_diff.append(diff)
                winning_F.append(F[i])
                winning_CR.append(CR[i])
                
                # Add parent to archive candidates
                new_archive_candidates.append(pop[i].copy())
                
                # Update population
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
            
        # Update Archive
        if len(new_archive_candidates) > 0:
            new_archive_candidates = np.array(new_archive_candidates)
            if archive.shape[0] == 0:
                archive = new_archive_candidates
            else:
                archive = np.vstack((archive, new_archive_candidates))
            
            # Resize archive if too big
            max_arc_size = int(pop_size * arc_rate)
            if archive.shape[0] > max_arc_size:
                # Randomly remove elements
                indices = np.random.choice(archive.shape[0], max_arc_size, replace=False)
                archive = archive[indices]
                
        # Update Memory (M_F, M_CR) using Weighted Lehmer Mean
        if len(winning_F) > 0:
            w_F = np.array(winning_F)
            w_CR = np.array(winning_CR)
            w_diff = np.array(improvement_diff)
            
            # Weights based on fitness improvement
            total_diff = np.sum(w_diff)
            if total_diff > 0:
                weights = w_diff / total_diff
                
                # Update M_CR (Weighted Mean)
                m_cr_new = np.sum(weights * w_CR)
                # Avoid locking CR to 0 or 1 completely
                M_CR[k_mem] = np.clip(m_cr_new, 0.05, 0.95) 
                
                # Update M_F (Weighted Lehmer Mean: sum(w*F^2) / sum(w*F))
                mean_pow2 = np.sum(weights * (w_F ** 2))
                mean_pow1 = np.sum(weights * w_F)
                if mean_pow1 > 1e-9:
                    m_f_new = mean_pow2 / mean_pow1
                    M_F[k_mem] = np.clip(m_f_new, 0.1, 1.0)
                
                # Advance memory index
                k_mem = (k_mem + 1) % H

        # Resort population for next generation (needed for p-best)
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
    return best_fitness
