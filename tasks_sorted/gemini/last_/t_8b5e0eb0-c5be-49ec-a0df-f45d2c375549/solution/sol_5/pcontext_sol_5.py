#The following algorithm is a **Fully Vectorized implementation of L-SHADE with Linear Population Size Reduction (LPSR) and a Soft Restart mechanism**.
#
#**Key Improvements over the previous version:**
#1.  **Full Vectorization**: The previous code iterated through the population to generate mutants and perform crossover. This version uses `numpy` matrix operations for all evolutionary operators. This drastically reduces Python overhead, allowing for significantly more function evaluations within the `max_time`.
#2.  **Corrected Weighted Lehmer Mean**: Implements the precise memory update mechanism defined in the L-SHADE/jSO papers, using weighted averages based on fitness improvements.
#3.  **Adaptive Restart**: Instead of a hard restart or continuous local search, this detects stagnation (population variance collapse or lack of improvement) and performs a "Soft Restart": it keeps the global best, scatters the remaining population, and resets the evolutionary memory to encourage new exploration.
#4.  **Reflective Boundaries**: Implements robust boundary handling that preserves the statistical distribution of the population near the edges.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimized Vectorized L-SHADE with LPSR and Soft Restart.
    """
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    def has_time():
        return (datetime.now() - start_time) < limit

    # --- Pre-computation ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Parameters ---
    # Initial population size (N_init)
    # For higher dims we need more people, but for speed we keep it manageable
    pop_size_init = int(round(18 * dim))
    if pop_size_init < 30: pop_size_init = 30
    
    # Minimum population size at the end (N_min)
    pop_size_min = 4
    
    # Archive size parameter
    arc_rate = 2.0 
    
    # Memory size (H)
    mem_size = 5 
    
    # --- Helper: Boundary Handling (Reflection) ---
    def apply_bounds(x_in):
        # Lower bound reflection
        mask_l = x_in < min_b
        if np.any(mask_l):
            x_in[mask_l] = 2.0 * min_b[np.where(mask_l)[1]] - x_in[mask_l]
            # If reflection goes out of upper bound, clip
            mask_l_2 = x_in < min_b
            x_in[mask_l_2] = min_b[np.where(mask_l_2)[1]]
            
        # Upper bound reflection
        mask_u = x_in > max_b
        if np.any(mask_u):
            x_in[mask_u] = 2.0 * max_b[np.where(mask_u)[1]] - x_in[mask_u]
            # If reflection goes out of lower bound, clip
            mask_u_2 = x_in > max_b
            x_in[mask_u_2] = max_b[np.where(mask_u_2)[1]]
            
        return np.clip(x_in, min_b, max_b)

    # --- Initialization ---
    pop_size = pop_size_init
    # Latin Hypercube Sampling for better initial spread
    rng = np.random.default_rng()
    grid = np.arange(pop_size).reshape(-1, 1)
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        pop[:, d] = rng.permutation(grid).flatten()
    jitter = rng.random((pop_size, dim))
    pop = min_b + (pop + jitter) / pop_size * diff_b
    
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluation
    best_idx = 0
    best_fitness = float('inf')
    
    for i in range(pop_size):
        if not has_time(): return best_fitness
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_idx = i
            
    # Best Solution Found
    best_sol = pop[best_idx].copy()
    
    # L-SHADE Memory
    mem_M_sf = np.full(mem_size, 0.5)
    mem_M_cr = np.full(mem_size, 0.5)
    mem_k = 0
    
    # Archive
    archive = np.zeros((0, dim))
    
    # Stagnation counter for restart
    stagnation_counter = 0
    last_best_fitness = best_fitness

    # --- Main Loop ---
    while has_time():
        
        # 1. Linear Population Size Reduction (LPSR)
        elapsed_sec = (datetime.now() - start_time).total_seconds()
        progress = elapsed_sec / max_time
        
        # Calculate target population size
        n_target = int(round(((pop_size_min - pop_size_init) * progress) + pop_size_init))
        n_target = max(pop_size_min, n_target)
        
        if pop_size > n_target:
            # Sort by fitness
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            # Reduce
            num_to_remove = pop_size - n_target
            pop = pop[:-num_to_remove]
            fitness = fitness[:-num_to_remove]
            pop_size = n_target
            
            # Archive resizing
            if archive.shape[0] > pop_size * arc_rate:
                # Randomly remove elements from archive to fit
                keep_indices = rng.choice(archive.shape[0], int(pop_size * arc_rate), replace=False)
                archive = archive[keep_indices]
                
            # Update best index after sort/cut
            best_idx = 0 # Since we sorted, best is at 0
            best_fitness = fitness[0]
            best_sol = pop[0].copy()

        # 2. Memory Update Parameter Generation
        # Generate indices for memory
        r_idx = rng.integers(0, mem_size, pop_size)
        m_sf = mem_M_sf[r_idx]
        m_cr = mem_M_cr[r_idx]
        
        # Generate CR (Normal Distribution)
        cr = rng.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        # In case of terminal value -1 in memory (not implemented here, but standard practice safe-guard)
        # We stick to clip 0,1.
        
        # Generate F (Cauchy Distribution)
        # Cauchy: location + scale * standard_cauchy
        f = m_sf + 0.1 * np.tan(np.pi * (rng.random(pop_size) - 0.5))
        
        # Constraints on F
        f = np.where(f > 1.0, 1.0, f)
        # If F <= 0, regenerate until > 0 (Standard SHADE logic)
        # Vectorized retry for F <= 0
        bad_f = f <= 0
        retry_count = 0
        while np.any(bad_f) and retry_count < 10:
            f[bad_f] = m_sf[bad_f] + 0.1 * np.tan(np.pi * (rng.random(np.sum(bad_f)) - 0.5))
            f = np.where(f > 1.0, 1.0, f)
            bad_f = f <= 0
            retry_count += 1
        f[f <= 0] = 0.5 # Fallback
        
        # 3. Mutation: current-to-pbest/1
        # Calculate p (linearly decreasing from 0.11 to 0.02 approx, standard jSO/SHADE tuning)
        # actually standard SHADE is constant p_min = 2/N to 0.2
        p_val = 0.2 * (1 - progress)
        p_val = max(2.0/pop_size, p_val) 
        p_best_num = max(1, int(round(pop_size * p_val)))
        
        # Sort current population to find p-best
        sorted_indices = np.argsort(fitness)
        # We need the actual vectors, not just indices
        pop_sorted = pop[sorted_indices]
        
        # Select x_pbest
        pbest_indices = rng.integers(0, p_best_num, pop_size)
        x_pbest = pop_sorted[pbest_indices]
        
        # Select x_r1 from P
        # Vectorized index selection ensuring r1 != i
        idxs = np.arange(pop_size)
        r1 = rng.integers(0, pop_size, pop_size)
        # Retry collisions
        mask_col = (r1 == idxs)
        while np.any(mask_col):
            r1[mask_col] = rng.integers(0, pop_size, np.sum(mask_col))
            mask_col = (r1 == idxs)
        x_r1 = pop[r1]
        
        # Select x_r2 from P U Archive
        # Union population
        if archive.shape[0] > 0:
            pop_all = np.vstack((pop, archive))
        else:
            pop_all = pop
            
        r2 = rng.integers(0, pop_all.shape[0], pop_size)
        # Retry collisions: r2 != r1 and r2 != i
        # Note: r2 index is into pop_all, r1/i are into pop.
        # r2 < pop_size means it points to current pop.
        mask_col = (r2 == idxs) | (r2 == r1)
        # If r2 >= pop_size, it's in archive, so no collision with i or r1 (assuming unique pointers)
        # But generally we just check index values if r2 is in P
        
        while np.any(mask_col):
            r2[mask_col] = rng.integers(0, pop_all.shape[0], np.sum(mask_col))
            mask_col = (r2 == idxs) | (r2 == r1)
            
        x_r2 = pop_all[r2]
        
        # Compute Mutant Vectors (Vectorized)
        # v = x + F*(xp - x) + F*(xr1 - xr2)
        # F needs to be column vector for broadcasting
        f_col = f[:, None]
        mutants = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        # Generate random mask
        j_rand = rng.integers(0, dim, pop_size)
        cross_mask = rng.random((pop_size, dim)) < cr[:, None]
        # Force at least one dimension
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trials = np.where(cross_mask, mutants, pop)
        
        # Bound Constraints
        trials = apply_bounds(trials)
        
        # 5. Selection and Updates
        trial_fitness = np.zeros(pop_size)
        
        # We need lists to store successful updates for memory
        success_f = []
        success_cr = []
        df = []
        
        # Archive candidates
        archive_candidates = []
        
        # Evaluation Loop (Cannot be fully vectorized due to func interface)
        for i in range(pop_size):
            if not has_time(): return best_fitness
            
            f_trial = func(trials[i])
            trial_fitness[i] = f_trial
            
            if f_trial <= fitness[i]:
                # Improvement or Equal
                if f_trial < fitness[i]:
                    archive_candidates.append(pop[i].copy())
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    df.append(fitness[i] - f_trial)
                
                # Update population
                pop[i] = trials[i]
                fitness[i] = f_trial
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trials[i].copy()
                    
        # Update Archive
        if archive_candidates:
            cands = np.array(archive_candidates)
            if archive.shape[0] + cands.shape[0] <= int(pop_size * arc_rate):
                archive = np.vstack((archive, cands))
            else:
                # Add and Randomly Trim
                current_len = archive.shape[0]
                space_left = int(pop_size * arc_rate) - current_len
                if space_left > 0:
                    archive = np.vstack((archive, cands[:space_left]))
                    # For the rest, replace random existing
                    remaining = cands[space_left:]
                    if remaining.shape[0] > 0 and archive.shape[0] > 0:
                        replace_idx = rng.choice(archive.shape[0], remaining.shape[0], replace=False)
                        archive[replace_idx] = remaining
                else:
                    # Archive full, replace random
                    if archive.shape[0] > 0:
                        replace_idx = rng.choice(archive.shape[0], cands.shape[0], replace=False)
                        archive[replace_idx] = cands

    
        # 6. Update Memory (Weighted Lehmer Mean)
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            s_df = np.array(df)
            
            # Normalize weights
            total_df = np.sum(s_df)
            if total_df > 0:
                w = s_df / total_df
                
                # Mean WL for F
                # mean_wl = sum(w * f^2) / sum(w * f)
                mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                
                # Mean WL for CR (Standard weighted mean)
                # Note: Some papers use WL for CR, others use Arithmetic. jSO uses WL.
                # Avoid CR=0 lock
                if np.max(s_cr) == 0:
                    mean_cr = 0
                else:
                    mean_cr = np.sum(w * s_cr)
                
                mem_M_sf[mem_k] = 0.5 * mem_M_sf[mem_k] + 0.5 * mean_f
                mem_M_cr[mem_k] = 0.5 * mem_M_cr[mem_k] + 0.5 * mean_cr
                
                mem_k = (mem_k + 1) % mem_size
        
        # 7. Soft Restart / Stagnation Check
        # If population variance is extremely low or no improvement for a while
        fit_std = np.std(fitness)
        
        # Update stagnation counter
        if best_fitness < last_best_fitness:
            last_best_fitness = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # Trigger conditions
        is_converged = fit_std < 1e-8
        # Allow some stagnation, but not forever (e.g., 50 generations)
        is_stagnant = stagnation_counter > 50
        
        if (is_converged or is_stagnant) and has_time():
            # Reset memory
            mem_M_sf.fill(0.5)
            mem_M_cr.fill(0.5)
            
            # Keep best, scramble rest
            # We re-initialize pop around the best or globally? 
            # Global restart is safer to escape local optima.
            
            # Save Best
            saved_best = best_sol.copy()
            saved_fit = best_fitness
            
            # Generate new random population
            new_pop = min_b + rng.random((pop_size, dim)) * diff_b
            
            # Evaluate new population
            new_fitness = np.zeros(pop_size)
            for i in range(pop_size):
                if not has_time(): return best_fitness
                new_fitness[i] = func(new_pop[i])
            
            pop = new_pop
            fitness = new_fitness
            
            # Inject Best back into index 0
            best_idx_new = np.argmin(fitness)
            if saved_fit < fitness[best_idx_new]:
                pop[0] = saved_best
                fitness[0] = saved_fit
            else:
                best_fitness = fitness[best_idx_new]
                best_sol = pop[best_idx_new].copy()
            
            # Reset Archive
            archive = np.zeros((0, dim))
            stagnation_counter = 0

    return best_fitness
