#Here is the improved algorithm. It implements **SHADE (Success-History based Adaptive Differential Evolution) with Restarts**. 
#
#**Key Improvements over Previous Algorithms:**
#1.  **Adaptive Parameters (History-Based):** Unlike static parameter choices (Algorithm 1) or simple self-adaptation (Algorithm 2), SHADE utilizes a historical memory ($M_{CR}, M_{F}$) to store successful parameter configurations. It generates new parameters based on what worked well in the past, effectively "learning" the optimization landscape.
#2.  **`current-to-pbest/1` Mutation:** This strategy strikes a superior balance between exploration and exploitation compared to `current-to-best` (too greedy) or `rand` (too slow). It mutates towards a random individual from the top $p\%$ best solutions.
#3.  **External Archive:** Inferior solutions replaced during selection are preserved in an external archive. These are used in the mutation step to maintain diversity without slowing down convergence, preventing the population from bunching up too quickly.
#4.  **Vectorization & Time Management:** The code is heavily vectorized using NumPy for maximum throughput and includes strict time checks to utilize the full `max_time` budget via restarts if the population converges early.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    with Restarts.
    
    SHADE adapts F and CR parameters using a historical memory of successful settings
    and uses an external archive to maintain diversity with current-to-pbest mutation.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: 18 * dim is a robust heuristic for SHADE
    pop_size = int(np.clip(18 * dim, 30, 150))
    
    # Memory size for historical adaptation
    H = 5 
    # p-best parameter (top 5% - 20%)
    p_best_rate = 0.11 
    # Archive size factor (Archive size = arc_rate * pop_size)
    arc_rate = 1.0 
    max_arc_size = int(pop_size * arc_rate)

    # Pre-process bounds
    bounds_arr = np.array(bounds)
    lower_bound = bounds_arr[:, 0]
    upper_bound = bounds_arr[:, 1]
    bound_diff = upper_bound - lower_bound
    
    global_best_fitness = float('inf')

    # Helper for weighted lehmer mean
    def weighted_lehmer_mean(values, weights):
        sum_weights = np.sum(weights)
        if sum_weights == 0: return np.mean(values)
        nw = weights / sum_weights
        num = np.sum(nw * (values ** 2))
        den = np.sum(nw * values)
        return num / den if den != 0 else 0

    # --- Main Loop (Restarts) ---
    while True:
        # Check time budget
        if time.time() - start_time >= max_time:
            return global_best_fitness
        
        # --- Initialization ---
        population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
        fitnesses = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return global_best_fitness
            val = func(population[i])
            fitnesses[i] = val
            if val < global_best_fitness:
                global_best_fitness = val

        # SHADE Memory Initialization
        # M_CR and M_F store historical successful means
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0 # Memory index pointer

        # External Archive (stores replaced individuals)
        archive = np.empty((0, dim))
        
        # Stagnation counter
        last_best_fit = np.min(fitnesses)
        stagnation_count = 0

        # --- Evolution Loop ---
        while True:
            if time.time() - start_time >= max_time:
                return global_best_fitness

            # 1. Parameter Generation
            # Select random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            r_cr = mem_cr[r_idx]
            r_f = mem_f[r_idx]

            # Generate CR using Normal distribution N(mem_cr, 0.1)
            # Clip to [0, 1]
            CR = np.random.normal(r_cr, 0.1, pop_size)
            CR = np.clip(CR, 0, 1)

            # Generate F using Cauchy distribution C(mem_f, 0.1)
            # If F <= 0, regenerate. If F > 1, clip to 1.
            F = r_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Correction for F
            retry_mask = F <= 0
            while np.any(retry_mask):
                F[retry_mask] = r_f[retry_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(retry_mask)) - 0.5))
                retry_mask = F <= 0
            F = np.clip(F, 0, 1)

            # 2. Mutation: current-to-pbest/1
            # Sort population to find p-best
            sorted_indices = np.argsort(fitnesses)
            
            # Determine p-best count (at least 2)
            num_pbest = max(2, int(pop_size * p_best_rate))
            pbest_indices = sorted_indices[:num_pbest]
            
            # Select pbest for each individual randomly
            pbest_choices = np.random.choice(pbest_indices, pop_size)
            x_pbest = population[pbest_choices]
            
            # Select r1 (distinct from current i)
            # We use permutation logic or simple random with check
            r1 = np.random.randint(0, pop_size, pop_size)
            # Ensure r1 != i (simplified: if collision, just reroll once)
            collision = (r1 == np.arange(pop_size))
            if np.any(collision):
                r1[collision] = np.random.randint(0, pop_size, np.sum(collision))
            x_r1 = population[r1]

            # Select r2 from Union(Population, Archive)
            # Union allows maintaining diversity
            if len(archive) > 0:
                pop_archive = np.concatenate((population, archive), axis=0)
            else:
                pop_archive = population
            
            r2 = np.random.randint(0, len(pop_archive), pop_size)
            # Ensure r2 != r1 and r2 != i is preferred, but collisions are rare in Union
            x_r2 = pop_archive[r2]

            # Mutation Vector Calculation
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            F_col = F[:, np.newaxis]
            mutant = population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)

            # 3. Crossover (Binomial)
            # Random mask
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < CR[:, np.newaxis]
            # Ensure at least one dimension
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # Bound Handling
            trial_pop = np.clip(trial_pop, lower_bound, upper_bound)

            # 4. Selection & Archive/Memory Update
            # Lists to store successful parameters
            succ_CR = []
            succ_F = []
            diff_fitness = []
            
            # Vectorized evaluation not strictly possible if func is black-box scalar
            # We iterate
            pop_fitness_improved = False
            
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return global_best_fitness
                
                # Check bounds again just in case (float precision)
                # trial_pop[i] = np.clip(trial_pop[i], lower_bound, upper_bound)
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitnesses[i]:
                    # Improvement or equal
                    df = fitnesses[i] - f_trial
                    
                    # Add replaced parent to archive
                    if f_trial < fitnesses[i]:
                        # Only strictly better ones go to archive/memory usually, 
                        # but SHADE allows equal. We stick to strict for archive logic.
                        if len(archive) < max_arc_size:
                            archive = np.vstack((archive, population[i]))
                        else:
                            # Replace random archive member
                            k = np.random.randint(0, max_arc_size)
                            archive[k] = population[i]
                            
                        # Store successful params
                        succ_CR.append(CR[i])
                        succ_F.append(F[i])
                        diff_fitness.append(df)
                        pop_fitness_improved = True

                    fitnesses[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial

            # 5. Update Memory (if any improvements)
            if len(succ_CR) > 0:
                succ_CR = np.array(succ_CR)
                succ_F = np.array(succ_F)
                diff_fitness = np.array(diff_fitness)
                
                # Weighted Lehmer Mean
                # Weights based on fitness improvement amount
                total_diff = np.sum(diff_fitness)
                if total_diff > 0:
                    weights = diff_fitness / total_diff
                    
                    # Update Memory at pointer k_mem
                    if k_mem < H: # Safety check
                        m_cr_new = 0
                        den_cr = np.sum(weights)
                        if den_cr > 0:
                            m_cr_new = np.sum(weights * succ_CR) # Arithmetic mean for CR
                        else:
                            m_cr_new = mem_cr[k_mem]

                        m_f_new = weighted_lehmer_mean(succ_F, weights) # Lehmer for F
                        
                        # Update (Smoothing could be applied, here we do direct update)
                        mem_cr[k_mem] = m_cr_new
                        mem_f[k_mem] = m_f_new
                        
                        k_mem = (k_mem + 1) % H

            # 6. Convergence / Restart Check
            current_best = np.min(fitnesses)
            if np.abs(current_best - last_best_fit) < 1e-9:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = current_best

            # Restart if variance is low (converged) or stuck too long
            pop_std = np.std(fitnesses)
            if pop_std < 1e-7 or stagnation_count > 40:
                break

    return global_best_fitness
