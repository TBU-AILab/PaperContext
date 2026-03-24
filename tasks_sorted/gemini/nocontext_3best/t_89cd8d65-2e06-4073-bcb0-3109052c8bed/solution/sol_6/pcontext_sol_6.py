#To improve upon the previous results, I propose an algorithm based on **SHADE (Success-History based Adaptive Differential Evolution) with OBL (Opposition-Based Learning) Initialization and Strategic Restarts**.
#
#### Algorithm Explanation
#
#1.  **SHADE Core**: This is one of the most powerful variants of Differential Evolution. It uses a historical memory to adapt the scaling factor ($F$) and crossover rate ($CR$) for each individual based on successful updates in previous generations. This allows it to automatically tune itself to the function's landscape.
#2.  **OBL Initialization (Opposition-Based Learning)**: At the beginning of each run (or restart), the algorithm generates a random population and its "opposite" population ($x' = min + max - x$). It evaluates both and keeps the better half. This simple mathematical trick drastically improves the probability of starting near the global optimum.
#3.  **Elitist Restarts**: If the population converges (low variance) or stagnates, the algorithm triggers a restart. Crucially, it injects the **global best solution found so far** into the new population. This ensures the algorithm never "forgets" its best discovery while exploring new areas of the search space.
#4.  **Robustness**: The implementation handles bounds using clipping, vectorizes operations for speed, and strictly manages the time budget to return the best result safely.
#
#### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History Adaptive Differential Evolution)
    with Opposition-Based Learning (OBL) initialization and Elitist Restarts.
    """
    start_time = datetime.now()
    # Reserve small buffer to ensure we return before timeout
    time_limit = timedelta(seconds=max_time * 0.98)
    
    # --- Pre-processing Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking across restarts
    global_best_fitness = float('inf')
    global_best_solution = None

    # SHADE Parameters
    H_memory_size = 5
    
    # --- Main Restart Loop ---
    while True:
        # Check time at start of restart
        if datetime.now() - start_time >= time_limit:
            return global_best_fitness

        # Population Sizing
        # Adaptive size: roughly 18*dim, constrained for performance
        pop_size = int(np.clip(18 * dim, 40, 150))
        
        # --- OBL Initialization ---
        # 1. Generate random population
        pop_rand = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # 2. Generate opposite population
        pop_opp = min_b + max_b - pop_rand
        # Check bounds for opposite (if out of bounds, randomize)
        invalid_opp = (pop_opp < min_b) | (pop_opp > max_b)
        if np.any(invalid_opp):
            # Replace invalid opposites with random values
            pop_opp = np.where(invalid_opp, min_b + np.random.rand(pop_size, dim) * diff_b, pop_opp)

        # 3. Evaluate both (Limited by time)
        pop_combined = np.vstack((pop_rand, pop_opp))
        fit_combined = np.full(2 * pop_size, float('inf'))
        
        for i in range(2 * pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_fitness
            try:
                val = func(pop_combined[i])
            except:
                val = float('inf')
            fit_combined[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_solution = pop_combined[i].copy()

        # 4. Select best N individuals to form initial population
        sort_indices = np.argsort(fit_combined)
        pop = pop_combined[sort_indices[:pop_size]]
        fitness = fit_combined[sort_indices[:pop_size]]
        
        # --- Elitism Injection ---
        # If we have a global best from a previous run, inject it to ensure 
        # we don't regress. We replace the worst individual.
        if global_best_solution is not None:
            # Check if global best is already in pop (avoid duplicates)
            # Using simple distance check or fitness check
            if fitness[0] > global_best_fitness:
                # This handles the rare case where the new run is purely worse
                pop[0] = global_best_solution
                fitness[0] = global_best_fitness
            elif abs(fitness[0] - global_best_fitness) > 1e-9:
                 # Inject into the worst slot if not already present
                pop[-1] = global_best_solution
                fitness[-1] = global_best_fitness
                # Re-sort after injection
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices]
                fitness = fitness[sort_indices]

        # --- SHADE Memory Initialization ---
        mem_M_F = np.full(H_memory_size, 0.5)
        mem_M_CR = np.full(H_memory_size, 0.5)
        k_mem = 0
        archive = []
        
        # --- Generation Loop ---
        while True:
            # Time check
            if datetime.now() - start_time >= time_limit:
                return global_best_fitness
            
            # 1. Parameter Generation
            # Randomly select memory index for each individual
            r_idx = np.random.randint(0, H_memory_size, pop_size)
            m_f = mem_M_F[r_idx]
            m_cr = mem_M_CR[r_idx]
            
            # Generate CR: Normal distribution, clipped [0, 1]
            # If memory is near terminal value, we keep it close to that
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # Generate F: Cauchy distribution
            # Must be > 0. If > 1, clamp to 1.
            F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Retry loop for F <= 0
            retry_mask = F <= 0
            retries = 0
            while np.any(retry_mask) and retries < 5:
                F[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
                retry_mask = F <= 0
                retries += 1
            F[F <= 0] = 0.5 # Fallback
            F = np.minimum(F, 1.0)
            
            # 2. Mutation: current-to-pbest/1
            # Sort population to find p-best
            # Pop is already sorted at start, but updates happen. 
            # We resort indices for p-best selection efficiency.
            sorted_indices = np.argsort(fitness)
            
            # Adaptive p: range [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_cnt = int(max(2, p_val * pop_size))
            top_indices = sorted_indices[:top_cnt]
            
            # Select pbest
            pbest_idxs = np.random.choice(top_indices, pop_size)
            x_pbest = pop[pbest_idxs]
            
            # Select r1 (distinct from i)
            idxs = np.arange(pop_size)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            # Shift collision
            r1_idxs = np.where(r1_idxs == idxs, (r1_idxs + 1) % pop_size, r1_idxs)
            x_r1 = pop[r1_idxs]
            
            # Select r2 (distinct from i and r1, from Union(Pop, Archive))
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((pop, archive_np))
            else:
                union_pop = pop
            
            union_size = len(union_pop)
            r2_idxs = np.random.randint(0, union_size, pop_size)
            
            # Fix collisions for r2
            # Case r2 == i
            r2_idxs = np.where(r2_idxs == idxs, (r2_idxs + 1) % union_size, r2_idxs)
            # Case r2 == r1 (only if r1 is within pop range)
            # Actually, just ensuring r2 != r1 is enough logic for vectorization
            # We map r1 indices to the union space (they are same 0..pop_size)
            r2_idxs = np.where(r2_idxs == r1_idxs, (r2_idxs + 1) % union_size, r2_idxs)
            
            x_r2 = union_pop[r2_idxs]
            
            # Compute Mutant
            F_col = F[:, np.newaxis]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # Boundary Handling: Clipping
            mutant = np.clip(mutant, min_b, max_b)
            
            # 3. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_jrand = np.zeros((pop_size, dim), dtype=bool)
            mask_jrand[np.arange(pop_size), j_rand] = True
            
            mask_cr = np.random.rand(pop_size, dim) < CR[:, np.newaxis]
            mask_trial = mask_cr | mask_jrand
            
            trial_pop = np.where(mask_trial, mutant, pop)
            
            # 4. Selection & Memory Update Preparations
            success_F = []
            success_CR = []
            diff_f = []
            
            # Archive parameters
            max_archive_size = pop_size
            
            for i in range(pop_size):
                # Strict check in loop
                if datetime.now() - start_time >= time_limit:
                    return global_best_fitness
                
                try:
                    f_trial = func(trial_pop[i])
                except:
                    f_trial = float('inf')
                
                if f_trial <= fitness[i]:
                    # Good trial
                    if f_trial < fitness[i]:
                        diff_f.append(fitness[i] - f_trial)
                        success_F.append(F[i])
                        success_CR.append(CR[i])
                        
                        # Add parent to archive
                        archive.append(pop[i].copy())
                        if len(archive) > max_archive_size:
                            # Randomly remove
                            rm = np.random.randint(0, len(archive))
                            archive.pop(rm)
                    
                    pop[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial
                        global_best_solution = trial_pop[i].copy()
            
            # 5. Update Memory
            if len(diff_f) > 0:
                diff_f = np.array(diff_f)
                success_F = np.array(success_F)
                success_CR = np.array(success_CR)
                
                total_diff = np.sum(diff_f)
                weights = diff_f / total_diff
                
                # Weighted Lehmer Mean for F
                mean_f = np.sum(weights * (success_F ** 2)) / np.sum(weights * success_F)
                
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(weights * success_CR)
                
                # Update Memory slots
                mem_M_F[k_mem] = 0.5 * mem_M_F[k_mem] + 0.5 * mean_f
                mem_M_CR[k_mem] = 0.5 * mem_M_CR[k_mem] + 0.5 * mean_cr
                
                k_mem = (k_mem + 1) % H_memory_size

            # 6. Convergence Check (Restart Trigger)
            # If population fitness variance is extremely low, we are likely stuck.
            # Or if max fitness - min fitness is negligible.
            fit_spread = np.max(fitness) - np.min(fitness)
            if fit_spread < 1e-8:
                break
                
    return global_best_fitness
