#Here is a fully operational, self-contained Python algorithm based on **Differential Evolution (DE)**.
#
#Differential Evolution is a robust, stochastic, derivative-free optimization method that generally outperforms random search and is well-suited for finding global minima in continuous search spaces within a limited time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution algorithm.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Hyperparameters ---
    # Population size: adaptive based on dimension, but clamped to ensure speed
    pop_size = int(np.clip(10 * dim, 20, 100))
    mutation_factor = 0.8    # F: Weighting factor for mutation (0.5 to 1.0)
    crossover_prob = 0.7     # CR: Crossover probability (0.0 to 1.0)

    # Convert bounds to numpy array for easier handling
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- Initialization ---
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.array([float('inf')] * pop_size)
    best_fitness = float('inf')
    
    # We perform an initial pass to fill fitness values and find the best so far
    for i in range(pop_size):
        # Time check
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness

        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Evolution Loop ---
    while True:
        # Check time at the start of every generation
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness
            
        for i in range(pop_size):
            # 1. Mutation: Select 3 distinct random individuals (r1, r2, r3) != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            x_1 = population[r1]
            x_2 = population[r2]
            x_3 = population[r3]
            
            # Create mutant vector: v = x_1 + F * (x_2 - x_3)
            mutant_vector = x_1 + mutation_factor * (x_2 - x_3)
            
            # 2. Crossover: Create trial vector
            # Select random index to ensure at least one parameter comes from mutant
            rand_j = np.random.randint(dim)
            crossover_mask = np.random.rand(dim) < crossover_prob
            crossover_mask[rand_j] = True
            
            trial_vector = np.where(crossover_mask, mutant_vector, population[i])
            
            # 3. Boundary Handling: Clip values to remain within bounds
            trial_vector = np.clip(trial_vector, min_b, max_b)
            
            # 4. Selection: Evaluate trial vector
            # Time check before expensive function call
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness

            trial_fitness = func(trial_vector)
            
            # Greedy selection
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best immediately if found
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
