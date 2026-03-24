#Here is a fully operational, self-contained Python algorithm using **Differential Evolution**. This approach is generally more robust and converges faster than simple random search for continuous optimization problems, making it suitable for finding minimums within a time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the output of 'func' using Differential Evolution within 'max_time'.
    """
    # Initialize timing
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: adaptive based on dimension, but capped to ensure speed
    pop_size = max(10, min(100, 10 * dim)) 
    F = 0.8   # Mutation factor (typically 0.5-0.9)
    CR = 0.9  # Crossover probability (typically 0.7-0.9)

    # Parse bounds into numpy arrays for efficient vectorization
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # 1. Initialize Population
    # Create random vectors within the bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best = float('inf')

    # Evaluate initial population
    for i in range(pop_size):
        # Check time budget before expensive function call
        if (datetime.now() - start_time) >= limit:
            return best
        
        score = func(population[i])
        fitness[i] = score
        
        if score < best:
            best = score

    # 2. Main Evolution Loop
    while True:
        for i in range(pop_size):
            # Check time budget
            if (datetime.now() - start_time) >= limit:
                return best

            # Mutation: Select 3 distinct random vectors (a, b, c) excluding current 'i'
            candidates = [idx for idx in range(pop_size) if idx != i]
            idxs = np.random.choice(candidates, 3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]

            # Create mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Constrain mutant to bounds (clip)
            mutant = np.clip(mutant, min_b, max_b)

            # Crossover: Create trial vector
            # Select dimensions to crossover based on probability CR
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one dimension is changed to maintain diversity
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])

            # Selection: Evaluate trial vector
            trial_score = func(trial)

            # Greedy selection: if trial is better, replace parent
            if trial_score < fitness[i]:
                fitness[i] = trial_score
                population[i] = trial
                
                # Update global best
                if trial_score < best:
                    best = trial_score

    return best
