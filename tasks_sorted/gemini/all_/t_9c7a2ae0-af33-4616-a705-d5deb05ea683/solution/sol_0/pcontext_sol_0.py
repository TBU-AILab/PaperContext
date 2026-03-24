#Here is a fully operational, self-contained Python implementation using the **Differential Evolution** algorithm. This method is generally far more efficient than random search for finding global minima in continuous space, especially within limited time constraints.
#
#It uses `numpy` for vector operations as implied by the problem glossary.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the minimum output value of func using Differential Evolution
    within a specified time limit.
    """
    
    # --- Configuration ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # DE Hyperparameters
    # Population size: Balance between diversity (high) and speed (low).
    # We use a dynamic size, capped to ensure iterations occur even in short time windows.
    pop_size = max(10, min(50, 10 * dim))
    F = 0.8  # Mutation factor (typically 0.5-1.0)
    CR = 0.9 # Crossover probability (typically 0.7-1.0)

    # Pre-process bounds for efficient numpy operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = np.fabs(max_b - min_b)

    # --- Initialization ---
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Array to store fitness values
    fitness = np.full(pop_size, float('inf'))
    
    # Track the global best
    best_val = float('inf')

    # Evaluate initial population
    # We check time inside this loop in case function evaluation is very slow
    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_val

        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # --- Main Optimization Loop ---
    while True:
        # Iterate through the population
        for i in range(pop_size):
            # strict time check before every new calculation
            if datetime.now() >= end_time:
                return best_val

            # 1. Mutation
            # Select 3 distinct random agents (a, b, c) excluding current agent i
            candidates = [idx for idx in range(pop_size) if idx != i]
            idxs = np.random.choice(candidates, 3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
            
            # Generate mutant vector: V = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Bound constraints: Clip mutant to stay within valid input range
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover
            # Create trial vector by mixing target(i) and mutant
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter is taken from mutant (standard DE rule)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            # Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better, replace current individual
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best found so far
                if f_trial < best_val:
                    best_val = f_trial

    # return fitness of the best found solution
    return best_val
