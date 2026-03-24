#The following algorithm implements a **Restart Differential Evolution (DE) with JADE-like adaptive mutation**.
#
#Key improvements over standard implementations:
#1.  **Restart Mechanism**: If the population converges to a local minimum or stagnates, the algorithm automatically restarts with a fresh population (while preserving the global best), allowing it to escape local optima and explore the search space more thoroughly within the time limit.
#2.  **Current-to-pBest Mutation**: Instead of random mutation, it uses a `current-to-pbest/1` strategy. This guides the search towards the top $10\%$ of superior solutions found so far, significantly speeding up convergence.
#3.  **Adaptive Parameters**: Mutation factor ($F$) and Crossover probability ($CR$) are randomized per individual for each generation, providing a robust mix of exploration and exploitation without needing manual tuning.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restart Differential Evolution with current-to-pbest mutation.
    This algorithm effectively balances fast local convergence with global search capabilities 
    via restarts.
    """
    start_time = time.time()
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    
    # Configuration
    # Population size: 10*dim is standard, bounded to [5, inf] to handle low dims
    pop_size = max(5, int(10 * dim))
    
    # 'p-best' parameter: mutation targets the top 10% of the population
    p_best_rate = 0.1
    
    # --- Outer Loop: Restarts ---
    while True:
        # Check if we have enough time to initialize a new population (heuristic buffer)
        if (time.time() - start_time) > max_time - 0.05:
            return best_fitness

        # 1. Initialize Population
        # Random distribution within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            # Strict time check inside evaluation loops
            if (time.time() - start_time) >= max_time:
                return best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val

        # Track local convergence for this restart
        local_min = np.min(fitness)
        stagnation_count = 0
        
        # --- Inner Loop: Evolution ---
        while True:
            # Time check
            if (time.time() - start_time) >= max_time:
                return best_fitness
            
            # 2. Adaptive Parameters (JADE-style)
            # F (Mutation Factor): Normal(0.5, 0.3), clipped [0.1, 1.0]
            # CR (Crossover Prob): Normal(0.9, 0.1), clipped [0.0, 1.0]
            # Randomized per individual to maintain diversity
            F = np.random.normal(0.5, 0.3, size=pop_size)
            F = np.clip(F, 0.1, 1.0)
            
            CR = np.random.normal(0.9, 0.1, size=pop_size)
            CR = np.clip(CR, 0.0, 1.0)
            
            # 3. Mutation: current-to-pbest/1
            # Equation: V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # Identify p-best individuals (top % of current population)
            sorted_idx = np.argsort(fitness)
            num_pbest = max(1, int(pop_size * p_best_rate))
            pbest_pool = sorted_idx[:num_pbest]
            
            # Select target vectors
            # pbest: random choice from top pool
            pbest_indices = np.random.choice(pbest_pool, size=pop_size)
            # r1, r2: random choice from population
            r1 = np.random.randint(0, pop_size, size=pop_size)
            r2 = np.random.randint(0, pop_size, size=pop_size)
            
            x = population
            x_pbest = population[pbest_indices]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Compute Mutant Vector
            # Reshape F for broadcasting
            F_col = F[:, np.newaxis]
            mutant = x + F_col * (x_pbest - x) + F_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            CR_col = CR[:, np.newaxis]
            mask = rand_vals < CR_col
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, size=pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trials = np.where(mask, mutant, x)
            
            # 5. Selection
            improved_this_gen = False
            for i in range(pop_size):
                if (time.time() - start_time) >= max_time:
                    return best_fitness
                
                f_trial = func(trials[i])
                
                # Greedy selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trials[i]
                    
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                    
                    if f_trial < local_min:
                        local_min = f_trial
                        improved_this_gen = True

            # 6. Convergence Check / Restart Trigger
            if not improved_this_gen:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            # Calculate population diversity (std dev of parameter space)
            pop_std = np.mean(np.std(population, axis=0))
            
            # Restart if:
            # - Stagnated (no improvement) for 20 generations
            # - Converged (variance is negligible)
            if stagnation_count > 20 or pop_std < 1e-6:
                break

    return best_fitness
