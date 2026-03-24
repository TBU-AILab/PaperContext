#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 8.627763194400856
#   algorithm code is: [DE with Adaptive Restarts]
#2. output value is: 11.61933730018302
#   algorithm code is: [L-SHADE with jDE]
#3. output value is: 15.230174087252792
#   algorithm code is: [SHADE with Restarts]
#
#The best performance so far (~8.63) was achieved by a **Differential Evolution** variant using the `current-to-pbest` strategy with restarts, but it lacked the external archive and memory-based adaptation that are standard in state-of-the-art optimizers (like SHADE). The previous SHADE implementation likely suffered from overhead or parameter tuning issues given the time constraints.
#
#The following algorithm improves upon these by implementing a **High-Efficiency SHADE (Success-History Adaptive Differential Evolution)** optimized for speed. Key improvements include:
#1.  **Optimized Memory & Archive**: Pre-allocated Numpy arrays replace Python lists for the archive and memory, significantly reducing overhead during the mutation step.
#2.  **SHADE Strategy**: It uses the `current-to-pbest/1` mutation strategy with an external archive to maintain diversity (preventing premature convergence) and history-based parameter adaptation ($F, CR$) to "learn" the fitness landscape.
#3.  **Elitist Restarts**: When the population stagnates (variance drops or fitness plateaus), it triggers a restart but injects the global best solution to ensure the search only improves.
#4.  **Vectorized Operations**: Almost all operations except the function evaluation itself are fully vectorized for maximum throughput within the time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using an optimized Success-History Adaptive 
    Differential Evolution (SHADE) with Restarts.
    """
    start_time = time.time()
    
    # --- Exception for Instant Timeout ---
    class TimeLimitExceeded(Exception):
        pass

    def check_time():
        if (time.time() - start_time) >= max_time:
            raise TimeLimitExceeded

    # --- Pre-computation ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global State
    best_fitness = float('inf')
    best_solution = None

    # --- Configuration ---
    # Population size: 18 * dim is a standard heuristic for SHADE.
    # We clamp it to ensure generations run fast enough within the time limit.
    pop_size = int(18 * dim)
    pop_size = max(30, min(150, pop_size))
    
    # Archive Size: typically 2.0 to 2.6 times the population size
    archive_size = int(2.0 * pop_size)
    
    # SHADE Memory Size
    H = 6 

    try:
        # --- Main Restart Loop ---
        while True:
            check_time()
            
            # 1. Initialize Population
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            fitness = np.full(pop_size, float('inf'))
            
            # Elitism: Inject global best from previous runs to maintain monotonicity
            start_eval_idx = 0
            if best_solution is not None:
                population[0] = best_solution
                fitness[0] = best_fitness
                start_eval_idx = 1
            
            # Evaluate Initial Population
            for i in range(start_eval_idx, pop_size):
                check_time()
                val = func(population[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_solution = population[i].copy()

            # Initialize SHADE Memory (F=0.5, CR=0.5 initially)
            mem_F = np.full(H, 0.5)
            mem_CR = np.full(H, 0.5)
            k_mem = 0
            
            # Initialize Archive (Pre-allocated for speed)
            archive = np.zeros((archive_size, dim))
            n_arc = 0 # Current number of items in archive
            
            # Stagnation Counters
            stag_counter = 0
            last_best_val = np.min(fitness)
            
            # --- Evolutionary Generation Loop ---
            while True:
                check_time()
                
                # Sort population based on fitness (required for current-to-pbest)
                # Best individuals move to the top (index 0)
                sorted_idx = np.argsort(fitness)
                population = population[sorted_idx]
                fitness = fitness[sorted_idx]
                
                # Check Stagnation / Convergence
                current_best_val = fitness[0]
                if abs(current_best_val - last_best_val) < 1e-12:
                    stag_counter += 1
                else:
                    stag_counter = 0
                    last_best_val = current_best_val
                
                # Restart if stuck (adjust 35 based on allowed time overhead)
                if stag_counter > 35 or np.std(fitness) < 1e-10:
                    break
                
                # --- Parameter Generation (SHADE) ---
                # Select random memory index for each individual
                r_idx = np.random.randint(0, H, pop_size)
                mu_f = mem_F[r_idx]
                mu_cr = mem_CR[r_idx]
                
                # Generate CR ~ Normal(mu_cr, 0.1)
                cr_g = np.random.normal(mu_cr, 0.1)
                cr_g = np.clip(cr_g, 0.0, 1.0)
                
                # Generate F ~ Cauchy(mu_f, 0.1)
                # Cauchy = loc + scale * tan(pi * (rand - 0.5))
                f_g = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
                f_g = np.where(f_g <= 0, 0.5, f_g) # Handle F <= 0
                f_g = np.clip(f_g, 0.0, 1.0)       # Clip F > 1
                
                # --- Mutation: current-to-pbest/1/bin ---
                # p-best selection: random individual from top p%
                # Randomized p_rate [2/pop_size, 0.2] adds robustness
                p_min = 2.0 / pop_size
                p_rates = np.random.uniform(p_min, 0.2, pop_size)
                p_indices = (p_rates * pop_size).astype(int)
                p_indices = np.maximum(p_indices, 1)
                
                # Generate indices for mutation vectors
                # pbest_idx: random int in [0, p_indices[i]]
                pbest_idxs = np.array([np.random.randint(0, p) for p in p_indices])
                r1_idxs = np.random.randint(0, pop_size, pop_size)
                
                # r2 selection: Union(Population, Archive)
                if n_arc > 0:
                    # Create view of active archive
                    active_archive = archive[:n_arc]
                    # Efficient vertical stack
                    union_pop = np.vstack((population, active_archive))
                else:
                    union_pop = population
                    
                r2_idxs = np.random.randint(0, len(union_pop), pop_size)
                
                # Pointers to vectors
                x = population
                x_pbest = population[pbest_idxs]
                x_r1 = population[r1_idxs]
                x_r2 = union_pop[r2_idxs]
                
                # Compute Mutant: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
                f_col = f_g[:, None]
                mutant = x + f_col * (x_pbest - x) + f_col * (x_r1 - x_r2)
                mutant = np.clip(mutant, min_b, max_b)
                
                # --- Crossover ---
                rand_cross = np.random.rand(pop_size, dim)
                mask = rand_cross < cr_g[:, None]
                # Ensure at least one dimension is taken from mutant
                j_rand = np.random.randint(0, dim, pop_size)
                mask[np.arange(pop_size), j_rand] = True
                
                trial_pop = np.where(mask, mutant, population)
                
                # --- Evaluation and Selection ---
                succ_F = []
                succ_CR = []
                diff_fitness = []
                
                # We update population in place, but need a copy logic for archive
                for i in range(pop_size):
                    check_time()
                    
                    f_trial = func(trial_pop[i])
                    f_old = fitness[i]
                    
                    if f_trial <= f_old:
                        # Successful update
                        
                        # 1. Update Archive (if strictly better)
                        if f_trial < f_old:
                            if n_arc < archive_size:
                                archive[n_arc] = population[i].copy()
                                n_arc += 1
                            else:
                                # Replace random member
                                replace_idx = np.random.randint(0, archive_size)
                                archive[replace_idx] = population[i].copy()
                                
                            # Record Success for Memory Update
                            succ_F.append(f_g[i])
                            succ_CR.append(cr_g[i])
                            diff_fitness.append(f_old - f_trial)
                        
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
                    df = np.array(diff_fitness)
                    
                    # Weights based on fitness improvement
                    total_df = np.sum(df)
                    if total_df > 0:
                        weights = df / total_df
                        
                        # Weighted Lehmer Mean for F
                        mean_f = np.sum(weights * (succ_F**2)) / (np.sum(weights * succ_F) + 1e-15)
                        
                        # Weighted Mean for CR
                        mean_cr = np.sum(weights * succ_CR)
                        
                        # Update Memory (moving average)
                        mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * mean_f
                        mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * mean_cr
                        
                        # Clamp to safe ranges
                        mem_F[k_mem] = np.clip(mem_F[k_mem], 0.05, 1.0)
                        mem_CR[k_mem] = np.clip(mem_CR[k_mem], 0.0, 1.0)
                        
                        # Advance pointer
                        k_mem = (k_mem + 1) % H

    except TimeLimitExceeded:
        return best_fitness

    return best_fitness
