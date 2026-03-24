#Here is a fully operational, self-contained implementation using the **Differential Evolution** algorithm.
#
#Differential Evolution is a robust, stochastic, derivative-free optimization method that is generally much more effective than random search for finding global minima in continuous spaces within a limited timeframe.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a specified time limit.
    
    Algorithm: DE/rand/1/bin
    - Mutation strategy: Target vector + Difference of two random vectors
    - Crossover: Binomial
    """
    
    # --- 1. Initialization & Hyperparameters ---
    start_time = time.time()
    
    # DE Hyperparameters
    # Population size: usually 10x to 15x the dimension is a good rule of thumb.
    # We ensure a minimum of 4 to allow for mutation (requires 3 distinct neighbors).
    pop_size = max(5, 10 * dim) 
    mutation_factor = 0.8  # F: Controls the amplification of the differential variation
    crossover_prob = 0.7   # CR: Probability of crossover
    
    # Convert bounds to numpy arrays for efficient calculation
    bounds_array = np.array(bounds)
    lower_bound = bounds_array[:, 0]
    upper_bound = bounds_array[:, 1]
    diff_bound = upper_bound - lower_bound

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = lower_bound + np.random.rand(pop_size, dim) * diff_bound
    
    # Evaluate initial population fitness
    # We initialize 'best' with infinity
    best_fitness = float('inf')
    best_vector = None
    
    # Store fitness of the population
    fitness_scores = np.zeros(pop_size)

    # Evaluate initial population safely respecting time
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            # If time runs out during initialization, return whatever best we found (or inf)
            return best_fitness if best_fitness != float('inf') else func(population[0])
            
        score = func(population[i])
        fitness_scores[i] = score
        
        if score < best_fitness:
            best_fitness = score
            best_vector = population[i].copy()

    # --- 2. Main Optimization Loop ---
    while True:
        # Check time constraint at the start of every generation
        if (time.time() - start_time) >= max_time:
            return best_fitness

        # Iterate through the population
        for i in range(pop_size):
            # Check time constraint frequently (inside the loop) to avoid overrunning
            # if the objective function is slow.
            if (time.time() - start_time) >= max_time:
                return best_fitness

            # --- Mutation ---
            # Select 3 distinct random indices distinct from current index i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]

            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # Bound constraints: Clip values to stay within bounds
            mutant = np.clip(mutant, lower_bound, upper_bound)

            # --- Crossover ---
            # Binomial crossover
            rand_vals = np.random.rand(dim)
            # Ensure at least one parameter is changed from the mutant (random index)
            j_rand = np.random.randint(dim)
            
            # Construct trial vector
            mask = (rand_vals < crossover_prob)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, population[i])

            # --- Selection ---
            trial_fitness = func(trial)

            # Greedy selection: if trial is better or equal, replace current
            if trial_fitness <= fitness_scores[i]:
                population[i] = trial
                fitness_scores[i] = trial_fitness
                
                # Update global best if found
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_vector = trial.copy()

    # return fitness of the best found solution
    return best_fitness
