#The following is a fully operational and self-contained algorithm implementing **SHADE (Success-History Adaptive Differential Evolution) with an External Archive and Aggressive Restarts**.
#
#### Algorithm Description
#
#To achieve a minimal output value within a hard time constraint, this algorithm improves upon the previous best strategies by integrating **Memory-Based Parameter Adaptation** (SHADE) with the **Diversity Preservation** of an external archive, wrapped in a **Restart Framework** to handle stagnation.
#
#1.  **Strategy: DE/current-to-pbest/1/bin with Archive**:
#    *   **Current-to-pbest**: Mutations are guided towards a random solution from the top $p\%$ (top 11%) of the population. This balances convergence speed with exploration better than targeting the single best solution.
#    *   **External Archive**: Solutions that are replaced by superior offspring are stored in a fixed-size archive. The mutation step pulls the second difference vector ($x_{r2}$) from the union of the Population and the Archive. This significantly increases the number of difference vector combinations, preserving diversity without extra function evaluations.
#
#2.  **Success-History Parameter Adaptation (SHADE)**:
#    *   Instead of fixed or purely random parameters, the algorithm maintains a memory ($M_{F}, M_{CR}$) of successful $F$ (Mutation Factor) and $CR$ (Crossover Rate) values from previous generations.
#    *   New parameters are sampled from Cauchy and Normal distributions centered on these memory values. This allows the algorithm to "learn" the specific landscape features (e.g., separability, modality) and converge on optimal control parameters automatically.
#
#3.  **Aggressive Restart Mechanism**:
#    *   **Stagnation Detection**: Tracks the improvement of the global best fitness. If no improvement is seen for a set number of generations (e.g., 25) or the population variance drops effectively to zero, a restart is triggered.
#    *   **Soft Restart**: The best solution found so far is injected into the new random population (Elitism), ensuring monotonic improvement across restarts. The adaptation memory is reset to defaults to allow learning of potentially different local basin properties.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using SHADE (Success-History Adaptive 
    Differential Evolution) with External Archive and Restarts.
    """
    start_time = time.time()
    
    # --- Pre-computation ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Population size: Heuristic 20 * sqrt(dim) offers good balance 
    # between exploration density and iteration speed.
    pop_size = int(20 * np.sqrt(dim))
    pop_size = max(30, min(100, pop_size))
    
    # Archive size: Typically equal to population size
    archive_size = pop_size
    
    # SHADE Memory parameters
    memory_size = 5
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_solution = None

    # Helper for Time Checking
    def check_time():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        
        # 1. Initialize State for Restart
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous runs
        start_eval_idx = 0
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
            start_eval_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_time(): return best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
        
        # Initialize SHADE Memory (F=0.5, CR=0.5 default)
        # Note: History-based adaptation converges parameters faster than pure random
        mem_F = np.full(memory_size, 0.5)
        mem_CR = np.full(memory_size, 0.5)
        k_mem = 0 # Memory index
        
        # Initialize Archive (Pre-allocated for speed)
        # We use a simple ring buffer logic or random replacement
        archive = np.zeros((archive_size, dim))
        archive_cnt = 0
        
        # Stagnation Counters
        stag_count = 0
        last_best_fit_in_run = np.min(fitness)
        
        # --- Evolutionary Generation Loop ---
        while not check_time():
            
            # Sort population for p-best selection
            # (Best individuals at index 0)
            sort_indices = np.argsort(fitness)
            population = population[sort_indices]
            fitness = fitness[sort_indices]
            
            # --- Stagnation Check ---
            current_best = fitness[0]
            if abs(current_best - last_best_fit_in_run) < 1e-10:
                stag_count += 1
            else:
                stag_count = 0
                last_best_fit_in_run = current_best
                
            # Restart if stuck (adjust threshold based on dim/difficulty)
            if stag_count > 25 or np.std(fitness) < 1e-9:
                break
            
            # --- Parameter Generation ---
            # Randomly select memory index for each individual
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_f = mem_F[r_idx]
            mu_cr = mem_CR[r_idx]
            
            # Generate CR: Normal(mu_cr, 0.1), clipped [0, 1]
            # If CR is very close to 0, fix to 0 (though unlikely with 0.1 std)
            cr_g = np.random.normal(mu_cr, 0.1, pop_size)
            cr_g = np.clip(cr_g, 0.0, 1.0)
            
            # Generate F: Cauchy(mu_f, 0.1), clipped [0.1, 1.0]
            # F > 1.0 is clamped to 1.0 (resampling not strictly necessary for time-efficiency)
            # F <= 0 is resampled to 0.1
            f_g = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f_g = np.where(f_g <= 0, 0.1, f_g)
            f_g = np.clip(f_g, 0.0, 1.0)
            
            # --- Mutation: DE/current-to-pbest/1 ---
            # p-best: top 11% (approx JADE/SHADE standard)
            p_top = max(2, int(0.11 * pop_size))
            
            # Vectors: pbest
            pbest_idxs = np.random.randint(0, p_top, pop_size)
            x_pbest = population[pbest_idxs]
            
            # Vectors: r1 (from population)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[r1_idxs]
            
            # Vectors: r2 (from Union(Population, Archive))
            # Create a view of the union
            if archive_cnt > 0:
                # Use valid part of archive
                active_archive = archive[:min(archive_size, archive_cnt)]
                union_set = np.concatenate((population, active_archive), axis=0)
                r2_idxs = np.random.randint(0, len(union_set), pop_size)
                x_r2 = union_set[r2_idxs]
            else:
                r2_idxs = np.random.randint(0, pop_size, pop_size)
                x_r2 = population[r2_idxs]
                
            # Compute Mutant: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            # Since population is sorted, population[i] is the 'x'
            mutant = population + f_g[:, None] * (x_pbest - population) + \
                     f_g[:, None] * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_j = np.random.rand(pop_size, dim)
            mask = rand_j < cr_g[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            mask = mask | j_rand_mask
            
            trial_pop = np.where(mask, mutant, population)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection & Memory Update Lists ---
            succ_F = []
            succ_CR = []
            diff_fitness = []
            
            for i in range(pop_size):
                if check_time(): return best_fitness
                
                f_trial = func(trial_pop[i])
                f_target = fitness[i]
                
                if f_trial <= f_target:
                    # Successful replacement
                    
                    # 1. Update Archive (if strictly better)
                    if f_trial < f_target:
                        if archive_cnt < archive_size:
                            archive[archive_cnt] = population[i].copy()
                            archive_cnt += 1
                        else:
                            # Random replacement if full
                            rem_idx = np.random.randint(0, archive_size)
                            archive[rem_idx] = population[i].copy()
                        
                        # Store success data for memory update
                        succ_F.append(f_g[i])
                        succ_CR.append(cr_g[i])
                        diff_fitness.append(f_target - f_trial)

                    # 2. Update Population
                    population[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    # 3. Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_pop[i].copy()
            
            # --- Update SHADE Memory ---
            if len(succ_F) > 0:
                succ_F = np.array(succ_F)
                succ_CR = np.array(succ_CR)
                w = np.array(diff_fitness)
                
                # Normalize weights
                total_w = np.sum(w)
                if total_w > 0:
                    w = w / total_w
                    
                    # Weighted Lehmer Mean for F
                    m_f_new = np.sum(w * (succ_F**2)) / (np.sum(w * succ_F) + 1e-15)
                    
                    # Weighted Arithmetic Mean for CR
                    m_cr_new = np.sum(w * succ_CR)
                    
                    # Update memory slot (moving average helps stability)
                    mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * m_f_new
                    mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * m_cr_new
                    
                    # Advance pointer
                    k_mem = (k_mem + 1) % memory_size
                    
                    # Ensure minimal mutation strength implies in Memory
                    mem_F[k_mem] = np.clip(mem_F[k_mem], 0.1, 1.0)
                    mem_CR[k_mem] = np.clip(mem_CR[k_mem], 0.0, 1.0)

    return best_fitness
