#Here is a fully operational, self-contained implementation using the **JADE (Joint Adaptive Differential Evolution)** algorithm with a **Restart Mechanism**.
#
#This approach improves upon the previous standard Differential Evolution by:
#1.  **Adapting Parameters:** It automatically tunes the mutation factor ($F$) and crossover rate ($CR$) based on successful evolutionary paths, eliminating the need to guess fixed hyperparameters.
#2.  **Current-to-pBest Mutation:** It directs the search towards the top $p\%$ of best solutions while maintaining diversity, converging significantly faster.
#3.  **Archive:** It maintains a history of recently discarded solutions to provide extra direction information, reducing the risk of stagnation.
#4.  **Restart Mechanism:** It detects population stagnation (when variance drops too low) and restarts the population (keeping the best solution) to escape local minima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using JADE (Joint Adaptive Differential Evolution) with 
    an Archive and Restart mechanism.
    
    This algorithm adapts 'F' and 'CR' parameters during execution and uses a 
    'current-to-pbest' mutation strategy to balance exploration and exploitation.
    It restarts the population if convergence is detected before time runs out.
    """
    
    # --- Helper Functions ---
    def trim_params(params, low, high):
        return np.clip(params, low, high)

    # --- Initialization ---
    start_time = time.time()
    
    # JADE Hyperparameters
    pop_size = max(10, 10 * dim)  # Population size
    archive_size = pop_size       # Archive size matches population
    p_best_rate = 0.05            # Top 5% used for mutation (greedy factor)
    c = 0.1                       # Learning rate for parameter adaptation
    
    # Adaptive Memory for F and CR
    mu_cr = 0.5
    mu_f = 0.5
    
    # Bounds processing
    bounds_np = np.array(bounds)
    lower_bound = bounds_np[:, 0]
    upper_bound = bounds_np[:, 1]
    bound_diff = upper_bound - lower_bound

    # Global Best Tracking
    global_best_fitness = float('inf')
    global_best_params = None

    # --- Main Optimization Loop (with Restarts) ---
    while True:
        # Check overall time before starting a new run/restart
        if (time.time() - start_time) >= max_time:
            return global_best_fitness

        # 1. Initialize Population
        population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
        fitness = np.full(pop_size, float('inf'))
        
        # Archive for inferior solutions
        archive = [] 
        
        # Evaluate initial population
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_params = population[i].copy()

        # 2. Evolutionary Cycle
        # We run generations until time runs out or convergence is detected
        while True:
            # Time Check
            if (time.time() - start_time) >= max_time:
                return global_best_fitness

            # Sort population by fitness to find p-best
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # --- Convergence Detection (Restart Logic) ---
            # If the population diversity is extremely low, break to outer loop to restart
            # (Calculated on the top 50% to ignore outliers)
            top_half = population[:pop_size//2]
            std_dev = np.mean(np.std(top_half, axis=0))
            if std_dev < 1e-8 and (time.time() - start_time) < (max_time * 0.9):
                # Apply a slight "shake" to the global best before restarting to avoid identical path
                # But mostly, we just break to re-init the population in the outer loop
                break 

            # --- Parameter Adaptation ---
            # Generate CR (Normal Distribution)
            crs = np.random.normal(mu_cr, 0.1, pop_size)
            crs = np.clip(crs, 0, 1)
            
            # Generate F (Cauchy Distribution)
            # numpy doesn't have a direct cauchy(loc, scale), we use standard_cauchy * scale + loc
            fs = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
            fs = np.clip(fs, 0.1, 1.0) # Truncate to valid range

            # Lists to store successful parameters
            succ_crs = []
            succ_fs = []
            
            # Create next generation
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)
            
            # Construct Union of Population and Archive for mutation
            if len(archive) > 0:
                archive_np = np.array(archive)
                pop_archive = np.vstack((population, archive_np))
            else:
                pop_archive = population

            # Determine p-best count
            p_limit = max(1, int(p_best_rate * pop_size))
            
            # --- Evolution Loop over individuals ---
            for i in range(pop_size):
                if (time.time() - start_time) >= max_time:
                    return global_best_fitness

                # Select r1 (distinct from i)
                idxs = [idx for idx in range(pop_size) if idx != i]
                r1 = population[np.random.choice(idxs)]
                
                # Select r2 (distinct from i and r1, from union of Pop + Archive)
                # Note: Logic simplified for speed; low prob of collision is acceptable
                r2_idx = np.random.randint(0, len(pop_archive))
                r2 = pop_archive[r2_idx]
                
                # Select p-best (random from top p%)
                p_best = population[np.random.randint(0, p_limit)]
                
                # Mutation: Current-to-pbest/1
                # V = Xi + F * (Xpbest - Xi) + F * (Xr1 - Xr2)
                mutant = population[i] + fs[i] * (p_best - population[i]) + fs[i] * (r1 - r2)
                
                # Crossover: Binomial
                cross_points = np.random.rand(dim) < crs[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                trial = trim_params(trial, lower_bound, upper_bound)
                
                # Selection
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    
                    # Store successful parameters
                    succ_crs.append(crs[i])
                    succ_fs.append(fs[i])
                    
                    # Add replaced individual to archive
                    archive.append(population[i].copy())
                    
                    # Update global best
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial
                        global_best_params = trial.copy()
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

            # Update population
            population = new_population
            fitness = new_fitness
            
            # Maintain Archive Size
            while len(archive) > archive_size:
                # Randomly remove elements to maintain size
                del archive[np.random.randint(0, len(archive))]

            # Update Adaptive Parameters (Lehmer Mean)
            if len(succ_crs) > 0:
                mean_pow = lambda x, p: np.sum(np.array(x)**p) / np.sum(np.array(x)**(p-1)) if np.sum(np.array(x)**(p-1)) != 0 else 0
                
                mu_cr = (1 - c) * mu_cr + c * np.mean(succ_crs)
                
                # Lehmer mean for F
                l_mean_f = mean_pow(succ_fs, 2)
                mu_f = (1 - c) * mu_f + c * l_mean_f

    return global_best_fitness
