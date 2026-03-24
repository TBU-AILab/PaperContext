import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm.
    DE is a robust stochastic evolutionary algorithm suitable for continuous 
    optimization problems and limited time constraints.
    """
    
    # --- Configuration ---
    # Population size: usually between 5*dim and 15*dim. 
    # We use a lower bound of 4 because DE requires at least 4 individuals (target + 3 randoms).
    pop_size = max(4, 10 * dim)
    
    # DE Parameters
    # F (Mutation factor): typically [0.5, 1.0]. Controls amplification of differential variation.
    # CR (Crossover probability): typically [0.0, 1.0]. Controls recombination.
    F = 0.8
    CR = 0.7

    # Time management
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    best_idx = -1

    # We evaluate individuals one by one to ensure we have a valid result 
    # even if max_time is extremely short.
    for i in range(pop_size):
        if datetime.now() >= end_time:
            # If we timed out during initialization, return the best found so far
            return best_val
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Optimization Loop ---
    # We use the 'rand/1/bin' strategy:
    # v = x_r1 + F * (x_r2 - x_r3)
    
    while True:
        # Iterate through the population
        for i in range(pop_size):
            # Check time constraint at every step to ensure strict adherence
            if datetime.now() >= end_time:
                return best_val

            # 1. Mutation
            # Select 3 random distinct indices excluding current index i
            candidates = list(range(pop_size))
            candidates.remove(i)
            # Efficiently sample 3 indices
            idxs = np.random.choice(candidates, 3, replace=False)
            
            x_1 = population[idxs[0]]
            x_2 = population[idxs[1]]
            x_3 = population[idxs[2]]
            
            # Create mutant vector
            mutant_vector = x_1 + F * (x_2 - x_3)
            
            # Clip mutant vector to bounds to ensure validity
            mutant_vector = np.clip(mutant_vector, min_b, max_b)

            # 2. Crossover (Binomial)
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter is changed (taken from mutant)
            # to avoid evaluating the exact same point
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial_vector = np.where(cross_points, mutant_vector, population[i])

            # 3. Selection
            trial_fitness = func(trial_vector)

            # Greedy selection: if trial is better, replace current
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best
                if trial_fitness < best_val:
                    best_val = trial_fitness
    
    return best_val
