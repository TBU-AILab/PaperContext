#Here is the implementation of a Differential Evolution (DE) algorithm. DE is a robust, gradient-free global optimization method well-suited for continuous parameter spaces and works efficiently within fixed time constraints.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    # Capture start time to manage the time budget
    start_time = time.time()
    
    # Define a safety buffer (in seconds) to ensure the function returns before the hard cutoff
    time_buffer = 0.05 
    timeout = max_time - time_buffer

    # Differential Evolution Hyperparameters
    # F: Mutation factor (typically 0.5-1.0). Controls amplification of differential variation.
    F = 0.8 
    # CR: Crossover probability (0.0-1.0). Controls diversity.
    CR = 0.9 
    
    # Population size. Standard is 10*dim, but constrained by time we keep it manageable.
    # We ensure a minimum of 5 to support the mutation logic (requires 3 others + target).
    pop_size = max(5, int(10 * dim))
    # Cap population for very high dimensions to ensure we get at least some generations
    if pop_size > 50:
        pop_size = 50

    # Parse bounds into numpy arrays for vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize Population: Random uniform distribution within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    best_vec = None

    # Initial evaluation loop
    for i in range(pop_size):
        # Check time constraint
        if (time.time() - start_time) > timeout:
            # If time runs out during initialization, return best found so far
            return best_val if best_val != float('inf') else func(population[0])

        val = func(population[i])
        fitness[i] = val
        
        # Update global best
        if val < best_val:
            best_val = val
            best_vec = population[i]

    # Main Evolutionary Loop
    while True:
        for i in range(pop_size):
            # Check time constraint strictly before expensive operations
            if (time.time() - start_time) > timeout:
                return best_val

            # --- Mutation ---
            # Select 3 distinct random indices distinct from current index i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]

            # Generate mutant vector: V = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Clip mutant vector to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover ---
            # Binomial crossover
            rand_vals = np.random.rand(dim)
            # Ensure at least one parameter is taken from mutant (fixed j_rand)
            j_rand = np.random.randint(dim)
            cross_mask = (rand_vals < CR)
            cross_mask[j_rand] = True
            
            # Trial vector u is mutant where mask is True, else target (population[i])
            trial = np.where(cross_mask, mutant, population[i])

            # --- Selection ---
            trial_fitness = func(trial)

            # Greedy selection: if trial is better or equal, replace target
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best found so far
                if trial_fitness < best_val:
                    best_val = trial_fitness
                    best_vec = trial

    # return fitness of the best found solution
    return best_val
