#Here is a fully operational, self-contained Python implementation. 
#
#### Improvements over the previous version:
#1.  **Adaptive Strategy (JADE-like)**: Instead of fixed `F` and `CR` parameters, this algorithm adapts them based on the success history of the population. This prevents the algorithm from being too aggressive (getting stuck) or too slow (wasting time).
#2.  **Mutation Strategy "Current-to-pBest"**: It uses a mutation strategy that pulls individuals towards the top percentage of the best solutions found so far ($p$-best), balancing exploitation (convergence) with exploration better than the standard random strategy.
#3.  **Restart Mechanism**: If the population converges (all individuals become very similar) before `max_time` is up, the algorithm saves the best solution and performs a "soft restart" of the population to search different areas of the solution space.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using an Adaptive Differential Evolution (JADE variant)
    with a 'current-to-pbest' mutation strategy and restart mechanism.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: usually 10x-20x dimension is good.
    pop_size = int(max(20, 15 * dim))
    
    # Adaptive Parameter Memory (Mean values for Cauchy/Normal distributions)
    mu_cr = 0.5
    mu_f = 0.5
    
    # Archive to maintain diversity (simplified for this implementation)
    # p-value for 'current-to-pbest' strategy (top 5% to 20%)
    p_val = 0.05 

    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Global best tracker
    global_best_val = float('inf')
    global_best_vec = None

    # --- Restart Loop ---
    # We loop until time runs out. If population converges, we restart.
    while True:
        
        # 1. Initialization
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val if global_best_vec is not None else float('inf')
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = population[i].copy()

        # 2. Main Generation Loop
        while True:
            # Time check
            if (time.time() - start_time) >= max_time:
                return global_best_val

            # Sort population by fitness to find 'p-best'
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # --- Convergence Check (Restart Trigger) ---
            # If the spread of fitness values is tiny, we are stuck in a local minima
            fit_std = np.std(fitness)
            if fit_std < 1e-8:
                # Break inner loop to trigger restart
                break

            # Lists to store successful parameters for adaptation
            successful_cr = []
            successful_f = []

            # --- Generate Parameters (F and CR) for this generation ---
            # F ~ Cauchy(mu_f, 0.1), CR ~ Normal(mu_cr, 0.1)
            # We generate slightly more to filter out invalid ones
            cr_g = np.random.normal(mu_cr, 0.1, pop_size)
            cr_g = np.clip(cr_g, 0, 1)
            
            # Cauchy for F: standard_cauchy + location
            f_g = 0.1 * np.random.standard_cauchy(pop_size) + mu_f
            # Clip F to (0, 1] - simpler handling than regenerating
            f_g = np.clip(f_g, 0.1, 1.0) 

            # Create Trial Population
            trials = np.zeros_like(population)
            
            # Determine top p% individuals for current-to-pbest
            top_p_count = max(1, int(pop_size * p_val))
            
            # Vectorized Indices for mutation
            # r1 != i, r2 != i, r1 != r2
            # We generate a pool and select. For speed in Python, we use random choice approx
            # tailored for current-to-pbest/1
            
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            idx_r2 = np.random.randint(0, pop_size, pop_size)
            
            # Select p-best indices (randomly from top p%)
            idx_pbest = np.random.randint(0, top_p_count, pop_size)

            for i in range(pop_size):
                if (time.time() - start_time) >= max_time:
                    return global_best_val

                # Ensure distinct indices (simple retry logic for robustness)
                r1 = idx_r1[i]
                r2 = idx_r2[i]
                pbest = idx_pbest[i]
                
                while r1 == i: r1 = np.random.randint(0, pop_size)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, pop_size)
                
                # Mutation: current-to-pbest/1
                # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
                # This drives convergence faster than rand/1
                
                x_i = population[i]
                x_pbest = population[pbest]
                x_r1 = population[r1]
                x_r2 = population[r2]
                
                F = f_g[i]
                mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
                
                # Boundary handling (Bounce back or Clip) -> Clipping is safer for general funcs
                mutant = np.clip(mutant, min_b, max_b)
                
                # Crossover (Binomial)
                CR = cr_g[i]
                j_rand = np.random.randint(0, dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True # Ensure at least one parameter changes
                
                trial = np.where(mask, mutant, x_i)
                
                # Selection
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    successful_cr.append(CR)
                    successful_f.append(F)
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_vec = trial.copy()
            
            # --- Parameter Adaptation (Lehman Mean) ---
            if len(successful_cr) > 0:
                # Update mu_cr (Arithmetic mean)
                mu_cr = (1 - 0.1) * mu_cr + 0.1 * np.mean(successful_cr)
                
                # Update mu_f (Lehmer mean: sum(x^2) / sum(x))
                sf = np.array(successful_f)
                mean_lehmer = np.sum(sf**2) / (np.sum(sf) + 1e-10)
                mu_f = (1 - 0.1) * mu_f + 0.1 * mean_lehmer

    return global_best_val
