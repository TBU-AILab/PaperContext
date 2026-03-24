#Here is the fully operational Python code implementing a Differential Evolution algorithm to minimize the output value within the time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the output of func using Differential Evolution within max_time.
    """
    # Start the timer
    start_time = time.time()
    
    # --- Differential Evolution Parameters ---
    # Population size: Adapted to dimension, but kept moderate to ensure 
    # generations can run within potentially short time limits.
    pop_size = max(10, int(10 + 1.5 * dim))
    
    # Mutation factor (F) and Crossover probability (CR)
    F = 0.6
    CR = 0.8

    # --- Initialization ---
    # Convert bounds to numpy arrays for efficient vector operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    bounds_diff = max_b - min_b

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * bounds_diff
    
    # Array to store fitness values
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize best found solution
    best = float('inf')

    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Check time budget
        if time.time() - start_time >= max_time:
            return best

        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Evolution Loop ---
    while True:
        for i in range(pop_size):
            # Check time budget strictly before every expensive operation
            if time.time() - start_time >= max_time:
                return best

            # Mutation: DE/rand/1
            # Select 3 distinct random individuals excluding the current one (i)
            candidates = [idx for idx in range(pop_size) if idx != i]
            idxs = np.random.choice(candidates, 3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
            
            # Create mutant vector
            mutant = a + F * (b - c)
            
            # Crossover: Binomial
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter is taken from the mutant
            cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])
            
            # Boundary Constraint Handling: Clip values to remain within bounds
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            # Evaluate the trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better, replace current individual
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if necessary
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
