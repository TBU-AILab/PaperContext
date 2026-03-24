#Here is a fully operational, self-contained Python implementation using the **Differential Evolution** algorithm. This method is a robust, gradient-free global optimization technique well-suited for finding minima in continuous search spaces within a limited time frame.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the set of input parameter values that lead to the minimum output value
    using the Differential Evolution algorithm within a limited time.
    """
    
    # --- Algorithm Hyperparameters ---
    # Population size: Typically 10x-15x dimension, but capped here to ensure 
    # rapid iterations within strict time limits.
    pop_size = max(5, 10 * dim)
    if pop_size > 50: 
        pop_size = 50
    
    mutation_factor (F) = 0.8      # Differential weight (0.5 to 1.0)
    crossover_prob (CR) = 0.7      # Crossover probability (0.0 to 1.0)

    # --- Initialization ---
    start = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population: shape (pop_size, dim)
    # Random values scaled to the bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population fitness
    fitness = np.full(pop_size, float('inf'))
    best_idx = 0
    best_val = float('inf')

    # We evaluate the initial population. We must check time even here.
    for i in range(pop_size):
        if (datetime.now() - start) >= time_limit:
            # If time runs out during init, return best found so far
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Differential Evolution Main Loop ---
    while True:
        # Check time at the start of every generation
        if (datetime.now() - start) >= time_limit:
            return best_val

        # Iterate over the population
        for i in range(pop_size):
            # Check time frequently (per individual) to adhere strictly to max_time
            if (datetime.now() - start) >= time_limit:
                return best_val

            # 1. Mutation: Target = a + F * (b - c)
            # Select 3 random distinct indices excluding current index i
            candidates = list(range(pop_size))
            candidates.remove(i)
            idxs = np.random.choice(candidates, 3, replace=False)
            
            a = population[idxs[0]]
            b = population[idxs[1]]
            c = population[idxs[2]]

            mutant = a + F * (b - c)

            # Clip mutant vector to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Binomial crossover
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter changes (forced dimension)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Create trial vector: choose from mutant if crossover True, else from population
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            trial_fitness = func(trial)

            # If trial is better or equal, replace existing individual
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best
                if trial_fitness < best_val:
                    best_val = trial_fitness

    return best_val
