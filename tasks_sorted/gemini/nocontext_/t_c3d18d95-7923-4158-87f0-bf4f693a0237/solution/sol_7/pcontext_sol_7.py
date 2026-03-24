#Here is the operational Python code implementing the **Differential Evolution (DE)** algorithm.
#
#DE is a stochastic, population-based optimization algorithm specifically designed for real-valued parameters. It is highly effective for global optimization of black-box functions because it does not require gradients, is robust against local minima, and adapts well to the search landscape.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the input parameters that minimize the output of 'func' using
    Differential Evolution within a specified time limit.
    """
    
    # --- Configuration ---
    # Differential Evolution Hyperparameters
    # Population size: Standard practice is 10x dimension, but we cap it 
    # to ensure efficiency within short time constraints.
    pop_size = max(10, min(10 * dim, 50)) 
    mutation_factor = 0.8   # F: Scaling factor for mutation [0, 2]
    crossover_prob = 0.7    # CR: Probability of crossover [0, 1]

    # Time Management
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    best_idx = -1

    for i in range(pop_size):
        # Check time before every function call
        if datetime.now() - start_time >= limit:
            return best_val

        val = func(population[i])
        fitness[i] = val

        if val < best_val:
            best_val = val
            best_idx = i

    # --- Evolution Loop ---
    while True:
        # Check time at the start of each generation
        if datetime.now() - start_time >= limit:
            return best_val

        for i in range(pop_size):
            # 1. Mutation (Strategy: DE/rand/1)
            # Pick 3 distinct random indices excluding the current one (i)
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Create mutant vector: V = X_a + F * (X_b - X_c)
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            
            # Clip mutant to ensure it stays within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover (Binomial)
            # Create trial vector by mixing target (population[i]) and mutant
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Guarantee at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection
            # Check time before expensive function call
            if datetime.now() - start_time >= limit:
                return best_val

            trial_fitness = func(trial_vector)

            # Greedy selection: if trial is better or equal, replace target
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best
                if trial_fitness < best_val:
                    best_val = trial_fitness
                    best_idx = i

    return best_val
