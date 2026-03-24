#Here is a fully operational, self-contained Python algorithm based on **Differential Evolution (DE)**.
#
#DE is a stochastic, population-based optimization algorithm that is highly effective for finding global minima in multi-dimensional spaces without requiring gradient information. It is robust against local minima and fits well within the "black box" constraints of the problem.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the minimum output value of func within max_time using Differential Evolution.
    """
    
    # --- Configuration ---
    # Setup time tracking
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)

    # Algorithm Hyperparameters
    # Population size: Scaled by dimension. Clamped between 10 and 50 to ensure
    # we get iterations done even if func is slow or dim is high.
    pop_size = max(10, min(50, 10 * dim)) 
    F = 0.8   # Differential weight (Mutation factor) [0, 2]
    CR = 0.9  # Crossover probability [0, 1]

    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population: shape (pop_size, dim)
    # Randomly distributed within the bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b

    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    
    # We iterate manually to check time after every function call
    for i in range(pop_size):
        if datetime.now() - start_time >= limit:
            # If we timeout during initialization, return best found so far.
            # If no eval done yet, force one eval to return valid float.
            if i == 0: 
                return func(population[0])
            return best_fitness

        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Optimization Loop ---
    # Standard Differential Evolution: DE/rand/1/bin
    while True:
        # Iterate through each individual in the population
        for i in range(pop_size):
            
            # Critical time check
            if datetime.now() - start_time >= limit:
                return best_fitness

            # 1. Mutation
            # Select 3 distinct random indices (a, b, c) excluding current index i
            candidates = np.delete(np.arange(pop_size), i)
            # Use choice to pick 3 distinct agents
            idxs = np.random.choice(candidates, 3, replace=False)
            a = population[idxs[0]]
            b = population[idxs[1]]
            c = population[idxs[2]]

            # Generate mutant vector: V = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Boundary constraint handling: Clip to bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover
            # Create trial vector by mixing target (population[i]) and mutant
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter comes from the mutant (guarantee mutation)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            # Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better, replace parent
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best
                if f_trial < best_fitness:
                    best_fitness = f_trial

    # return fitness of the best found solution
    return best_fitness
