#The following is a fully operational and self-contained algorithm using an improved **Differential Evolution (DE)** strategy optimized for limited-time execution.
#
##### Algorithm Description
#To achieve a lower minimum value (better fitness) than previous iterations, this algorithm implements **DE/current-to-pbest/1/bin with Adaptive Restarts**. 
#1.  **Strategy (DE/current-to-pbest/1/bin)**: Unlike `current-to-best`, which can converge prematurely to local optima, `current-to-pbest` guides the population towards a random individual selected from the top $p\%$ best candidates. This balances convergence speed with population diversity.
#2.  **Adaptive Population Sizing**: The population size is heuristically derived from the dimension ($20\sqrt{D}$) to ensure sufficient search density for higher dimensions while remaining fast for lower ones.
#3.  **Randomized Control Parameters**: Instead of fixed values, mutation factor $F$ and crossover rate $CR$ are randomized per individual for every generation. This mimics self-adaptive behaviors (like SHADE), allowing the algorithm to dynamically handle different fitness landscape features.
#4.  **Stagnation Detection & Restarts**: The algorithm monitors fitness improvements. If the population standard deviation drops (convergence) or the best fitness doesn't improve for a set number of generations (stagnation), it restarts.
#5.  **Elitism**: On restart, the global best solution is carried over to the new population to prevent regression.
#
##### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Differential Evolution with the 
    DE/current-to-pbest/1/bin strategy and adaptive restarts.
    """
    # Initialize timing
    start_time = time.time()
    
    # Precompute bounds for vectorization efficiency
    bounds_arr = np.array(bounds)
    min_bound = bounds_arr[:, 0]
    max_bound = bounds_arr[:, 1]
    diff_bound = max_bound - min_bound
    
    # --- Adaptive Configuration ---
    # Population size scaling: slightly larger than standard to support p-best diversity
    # Heuristic: pop_size = 20 * sqrt(dim), clamped between 20 and 100 for efficiency
    pop_size = int(20 * np.sqrt(dim))
    pop_size = max(20, min(100, pop_size))
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    best_solution = None
    
    # Helper to enforce time limit
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    while not is_time_up():
        
        # 1. Initialize Population
        # Uniform random initialization within bounds
        population = min_bound + np.random.rand(pop_size, dim) * diff_bound
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        start_idx = 0
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
            start_idx = 1 # Skip re-evaluating the injected best
        
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if is_time_up(): return best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
                
        # 2. Differential Evolution Loop
        stagnation_count = 0
        prev_min_fit = np.min(fitness)
        
        while not is_time_up():
            # Sort population by fitness (required for current-to-pbest strategy)
            # This moves the best individuals to the top of the arrays
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            current_min_fit = fitness[0]
            
            # --- Convergence & Stagnation Checks ---
            # If population variance is negligible, we have converged -> Restart
            if np.std(fitness) < 1e-8:
                break
            
            # If best fitness hasn't improved significantly, increment stagnation counter
            if abs(current_min_fit - prev_min_fit) < 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_min_fit = current_min_fit
            
            # Trigger restart if stuck in a local optimum for too long
            if stagnation_count > 30:
                break
            
            # --- Strategy: DE/current-to-pbest/1/bin ---
            # 'p-best': Select target from the top p% (p_val) individuals
            # This maintains diversity better than targeting solely the single best
            p_val = max(2, int(0.15 * pop_size))
            
            # Vectorized Indices Generation
            # 1. p-best indices: random integer from [0, p_val) for each individual
            idxs_pbest = np.random.randint(0, p_val, pop_size)
            x_pbest = population[idxs_pbest]
            
            # 2. r1, r2 indices: random integers from whole population
            idxs_r1 = np.random.randint(0, pop_size, pop_size)
            idxs_r2 = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[idxs_r1]
            x_r2 = population[idxs_r2]
            
            # Randomized Control Parameters (Dithering)
            # F ~ Uniform(0.5, 0.95), CR ~ Uniform(0.8, 1.0)
            # Randomizing per individual helps escape local basins
            F = np.random.uniform(0.5, 0.95, (pop_size, 1))
            CR = np.random.uniform(0.8, 1.0, (pop_size, 1))
            
            # Mutation Equation: v = x + F(x_pbest - x) + F(x_r1 - x_r2)
            # Note: 'population' is the current x because we sorted it
            mutant = population + F * (x_pbest - population) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(pop_size, dim) < CR
            # Constraint: Ensure at least one dimension is inherited from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            # Construct Trial Population
            trial = np.where(cross_mask, mutant, population)
            
            # Bounds Constraint (Clipping)
            trial = np.clip(trial, min_bound, max_bound)
            
            # --- Selection (Evaluation) ---
            # Evaluate trial vectors and apply greedy selection
            for i in range(pop_size):
                if is_time_up(): return best_fitness
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial[i]
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial[i].copy()
                        
    return best_fitness
