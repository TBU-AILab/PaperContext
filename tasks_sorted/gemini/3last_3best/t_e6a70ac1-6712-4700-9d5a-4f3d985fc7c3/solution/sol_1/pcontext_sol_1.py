#The following is a fully operational and self-contained algorithm using an improved **Differential Evolution (DE)** strategy.
#
#### Algorithm Description
#To minimize the output value within the limited time, this algorithm implements **Differential Evolution with Restarts and Dithering**.
#1.  **Strategy (DE/current-to-best/1/bin)**: This variant balances exploration and exploitation better than the standard `rand/1` strategy by guiding individuals towards the best solution found so far while maintaining diversity using difference vectors.
#2.  **Restarts**: If the population converges (standard deviation of fitness becomes negligible) before the time limit, the algorithm restarts with a fresh random population. This prevents the algorithm from getting stuck in local optima.
#3.  **Elitism**: During restarts, the global best solution found is injected into the new population to ensure the search never regresses.
#4.  **Dithering**: The mutation factor $F$ is randomized for each generation to handle various fitness landscape scales dynamically.
#5.  **Vectorization**: NumPy operations are used for mutation and crossover to maximize computational efficiency, allowing more function evaluations within `max_time`.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Differential Evolution with Restarts 
    and the DE/current-to-best/1/bin strategy.
    """
    start_time = time.time()
    
    # --- Algorithm Configuration ---
    # Population size: Adaptive based on dimension.
    # Scaled to ensure enough diversity for high dims, but fast enough for low dims.
    pop_size = int(10 * np.sqrt(dim))
    pop_size = max(10, pop_size)
    pop_size = min(50, pop_size)
    
    # Crossover Probability
    CR = 0.9

    # Pre-process bounds for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    best_solution = None

    # Helper function to check if time budget is exhausted
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    # Loop continues restarting the population until time runs out
    while not is_time_up():
        
        # 1. Initialize Population
        # Uniform random distribution within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        # This ensures we don't lose the best candidate during a restart.
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if is_time_up(): return best_fitness
            
            # Skip re-evaluation of the injected best solution
            if best_solution is not None and i == 0:
                continue

            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
        
        # 2. Evolution Generations Loop
        # Run until the population converges or time runs out
        while not is_time_up():
            
            # Check for Convergence
            # If the population variance is extremely low, we are likely in a local optimum.
            # Break the inner loop to trigger a restart.
            if np.std(fitness) < 1e-8:
                break
            
            # Dithering: Randomize Mutation Factor F [0.5, 1.0] per generation
            F = 0.5 + 0.5 * np.random.rand()
            
            # Find the best individual in the current population for the mutation strategy
            current_best_idx = np.argmin(fitness)
            x_best = population[current_best_idx]
            
            # --- Vectorized Mutation & Crossover ---
            
            # Generate random indices for difference vectors
            # Note: We allow minor collisions (r1==r2 or r1==i) for computational speed.
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Strategy: DE/current-to-best/1/bin
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This moves every individual towards the best, plus a random difference for diversity.
            mutant = population + F * (x_best - population) + F * (population[r1] - population[r2])
            
            # Binary Crossover
            cross_mask = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one parameter is taken from the mutant (DE constraint)
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            # Create trial population
            trial_pop = np.where(cross_mask, mutant, population)
            
            # Apply Bounds Constraints (Clipping)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection (Evaluation) ---
            for i in range(pop_size):
                if is_time_up(): return best_fitness
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection: Accept if better or equal
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_pop[i].copy()

    return best_fitness
