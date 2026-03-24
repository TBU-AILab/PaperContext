#The following algorithm implements a simplified version of **SHADE (Success-History based Adaptive Differential Evolution)**.
#
#**Rationale for Improvement:**
#1.  **Adaptive Parameters (Memory-Based)**: Unlike jDE which resets parameters or jitters them, SHADE maintains a history of successful $F$ and $CR$ values, learning the optimal distribution for the landscape over time.
#2.  **Current-to-pBest Mutation**: This strategy ($V = X_i + F(X_{pbest} - X_i) + F(X_{r1} - X_{r2})$) balances exploitation (pulling towards best solutions) and exploration better than standard DE or jDE strategies.
#3.  **External Archive**: By storing inferior solutions recently replaced, the algorithm preserves diversity and prevents the difference vector $(X_{r1} - X_{r2})$ from vanishing too quickly, avoiding premature convergence.
#4.  **Robust Time Management**: The code checks the time limit before every function evaluation to ensure strict adherence to `max_time`.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using SHADE (Success-History based Adaptive Differential Evolution).
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Adapted to dimension but constrained for performance
    # Standard SHADE uses ~18*dim, but we cap it to ensure many generations fit in max_time.
    pop_size = int(18 * dim)
    if pop_size > 100: pop_size = 100
    if pop_size < 30: pop_size = 30
    
    archive_size = int(pop_size * 2.0)
    
    # SHADE History Parameters
    H = 6  # Size of memory for successful parameters
    mem_F = np.full(H, 0.5)   # Initial memory for F
    mem_CR = np.full(H, 0.5)  # Initial memory for CR
    k_mem = 0  # Memory index pointer
    p_best_rate = 0.11  # Greedy factor for current-to-pbest
    
    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Archive (Pre-allocated, filled up to arc_cnt)
    archive = np.zeros((archive_size, dim))
    arc_cnt = 0
    
    global_best_val = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) > max_time:
            return global_best_val
        
        val = func(pop[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val

    # --- Main Loop ---
    while True:
        # Check time at start of generation
        if (time.time() - start_time) > max_time:
            return global_best_val
            
        # 1. Parameter Generation
        # Randomly select memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mF = mem_F[r_idx]
        mCR = mem_CR[r_idx]
        
        # Generate F using Cauchy distribution: C(mF, 0.1)
        # F = mF + 0.1 * tan(pi * (rand - 0.5))
        F = mF + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Constraint handling for F
        # If F > 1 -> 1. If F <= 0 -> 0.5 (Re-initialize)
        F = np.where(F <= 0, 0.5, F)
        F = np.where(F > 1, 1.0, F)
        
        # Generate CR using Normal distribution: N(mCR, 0.1)
        CR = mCR + 0.1 * np.random.randn(pop_size)
        CR = np.clip(CR, 0.0, 1.0)
        
        # 2. Mutation: current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Identify p-best individuals (top p%)
        sorted_indices = np.argsort(fitness)
        num_pbest = max(2, int(pop_size * p_best_rate))
        top_indices = sorted_indices[:num_pbest]
        
        # Select pbest for each individual
        pbest_indices = np.random.choice(top_indices, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (distinct from target i)
        # Random int, shift if collision with self
        r1_indices = np.random.randint(0, pop_size, pop_size)
        conflict_mask = (r1_indices == np.arange(pop_size))
        r1_indices[conflict_mask] = (r1_indices[conflict_mask] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Population U Archive)
        # Logic: Indices < pop_size are population, >= pop_size are archive
        total_pool_size = pop_size + arc_cnt
        r2_indices = np.random.randint(0, total_pool_size, pop_size)
        
        # Construct pool for r2 retrieval
        if arc_cnt > 0:
            pool = np.vstack((pop, archive[:arc_cnt]))
            x_r2 = pool[r2_indices]
        else:
            x_r2 = pop[r2_indices]
            
        # Calculate Mutant Vector
        # F must be column vector for broadcasting
        F_col = F[:, None]
        mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        rand_matrix = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        j_mask = np.zeros((pop_size, dim), dtype=bool)
        j_mask[np.arange(pop_size), j_rand] = True
        
        cross_mask = (rand_matrix < CR[:, None]) | j_mask
        trial = np.where(cross_mask, mutant, pop)
        
        # 4. Bound Constraint Handling (Midpoint Correction)
        # If violated, set to midpoint between parent and bound to preserve direction
        bad_lower = trial < min_b
        bad_upper = trial > max_b
        
        trial = np.where(bad_lower, (pop + min_b) * 0.5, trial)
        trial = np.where(bad_upper, (pop + max_b) * 0.5, trial)
        
        # 5. Evaluation and Selection
        success_F = []
        success_CR = []
        df_list = []
        parents_to_archive = []
        
        # Prepare next generation containers
        next_pop = pop.copy()
        next_fitness = fitness.copy()
        
        for i in range(pop_size):
            # Strict time check before expensive function call
            if (time.time() - start_time) > max_time:
                return global_best_val
            
            f_trial = func(trial[i])
            
            if f_trial < global_best_val:
                global_best_val = f_trial
            
            # Greedy selection
            if f_trial <= fitness[i]:
                # Improvement or equal
                df = fitness[i] - f_trial
                
                # Record success for parameter adaptation
                success_F.append(F[i])
                success_CR.append(CR[i])
                df_list.append(df)
                
                # Add replaced parent to archive list
                parents_to_archive.append(pop[i].copy())
                
                # Update individual
                next_pop[i] = trial[i]
                next_fitness[i] = f_trial
        
        # Move to next generation
        pop = next_pop
        fitness = next_fitness
        
        # 6. Archive Update
        if len(parents_to_archive) > 0:
            add_vals = np.array(parents_to_archive)
            num_add = len(add_vals)
            
            if arc_cnt + num_add <= archive_size:
                # Append if space exists
                archive[arc_cnt : arc_cnt + num_add] = add_vals
                arc_cnt += num_add
            else:
                # Fill remaining space
                slots = archive_size - arc_cnt
                if slots > 0:
                    archive[arc_cnt:] = add_vals[:slots]
                    add_vals = add_vals[slots:]
                    arc_cnt = archive_size
                
                # Randomly replace existing archive members with remainder
                if len(add_vals) > 0:
                    replace_idx = np.random.randint(0, archive_size, len(add_vals))
                    archive[replace_idx] = add_vals

        # 7. Memory Update (Weighted Lehmer Mean)
        if len(df_list) > 0:
            df = np.array(df_list)
            total_df = np.sum(df)
            
            # Update only if there was actual improvement
            if total_df > 0:
                weights = df / total_df
                sf = np.array(success_F)
                scr = np.array(success_CR)
                
                # Weighted Lehmer Mean for F
                denom = np.sum(weights * sf)
                if denom > 1e-9:
                    mean_F = np.sum(weights * (sf ** 2)) / denom
                    mem_F[k_mem] = np.clip(mean_F, 0, 1)
                
                # Weighted Arithmetic Mean for CR
                mean_CR = np.sum(weights * scr)
                mem_CR[k_mem] = np.clip(mean_CR, 0, 1)
                
                # Increment memory pointer
                k_mem = (k_mem + 1) % H

    return global_best_val
