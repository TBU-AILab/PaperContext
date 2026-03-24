#Here is the fully operational Python code implementing a Differential Evolution algorithm. This approach is generally much more efficient than random search for continuous optimization problems, as it intelligently explores the search space using the differences between existing solutions.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution algorithm within a time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Differential Evolution Hyperparameters ---
    # F: Mutation factor (typically 0.5 to 1.0) controls differential weight
    F = 0.8
    # CR: Crossover probability (0.0 to 1.0) controls diversity
    CR = 0.9
    # Population size: adaptive to dimension, capped for performance
    pop_size = max(10, int(5 * dim))
    if pop_size > 50:
        pop_size = 50

    # --- Initialization ---
    # Prepare bounds as numpy arrays for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Array to store fitness of current population
    fitness_scores = np.full(pop_size, float('inf'))
    
    # Track global best
    best = float('inf')

    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Check time constraint
        if datetime.now() - start_time >= time_limit:
            return best
            
        val = func(population[i])
        fitness_scores[i] = val
        
        if val < best:
            best = val

    # --- Main Optimization Loop ---
    while True:
        for i in range(pop_size):
            # Check time constraint strictly before every evaluation
            if datetime.now() - start_time >= time_limit:
                return best

            # 1. Mutation (DE/rand/1 strategy)
            # Select 3 distinct random individuals different from current 'i'
            idxs = list(range(pop_size))
            idxs.remove(i)
            # Using random.sample is efficient for small counts
            a_idx, b_idx, c_idx = random.sample(idxs, 3)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Create mutant vector
            mutant = a + F * (b - c)
            
            # Enforce bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover (Binomial)
            # Generate mask for crossover
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one dimension is changed to avoid duplicating parent
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            # Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better or equal, it replaces the parent
            if f_trial <= fitness_scores[i]:
                fitness_scores[i] = f_trial
                population[i] = trial
                
                # Update global best if found
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
