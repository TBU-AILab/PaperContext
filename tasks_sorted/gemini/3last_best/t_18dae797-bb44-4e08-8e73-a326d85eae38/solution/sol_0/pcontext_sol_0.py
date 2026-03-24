#Here is a fully operational, self-contained Python algorithm based on **Differential Evolution (DE)**. DE is a robust global optimization algorithm well-suited for finding the minimum of a function within a continuous search space and works effectively without requiring gradient information.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using Differential Evolution (DE) within a specific time limit.
    Strategy: DE/rand/1/bin
    """
    
    # --- Initialization & Time Management ---
    start_time = datetime.now()
    # Subtract a small buffer to ensure we return cleanly before the strict limit
    time_limit = timedelta(seconds=max_time) 
    
    # Initialize best result variable
    best = float('inf')

    # --- Hyperparameters ---
    # Population size: Heuristic to balance diversity vs. speed.
    # We cap it at 100 to ensure generations complete within short max_time limits.
    # Minimum 10 to allow sufficient mutation candidates.
    pop_size = int(max(10, min(100, 10 * dim)))
    
    # F: Mutation factor (typically 0.5-0.9). Controls amplification of differential variation.
    F = 0.8 
    # CR: Crossover probability (typically 0.5-0.9). Controls recombination.
    CR = 0.9 

    # --- Pre-process Bounds ---
    # Convert bounds to numpy arrays for efficient vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- Create Initial Population ---
    # Shape: (pop_size, dim)
    # Start with random values distributed uniformly within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))

    # --- Evaluate Initial Population ---
    # We check time during initialization in case func is very slow
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best if best != float('inf') else float('inf')

        score = func(population[i])
        fitness[i] = score
        
        if score < best:
            best = score

    # --- Main Differential Evolution Loop ---
    while True:
        # Iterate through each individual in the population
        for i in range(pop_size):
            
            # 1. Check Time Constraint
            if (datetime.now() - start_time) >= time_limit:
                return best

            # 2. Mutation (DE/rand/1)
            # Select 3 distinct random individuals (a, b, c) different from current (i)
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = random.sample(idxs, 3)
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            # Create mutant vector: V = X_a + F * (X_b - X_c)
            mutant = x_a + F * (x_b - x_c)
            
            # 3. Crossover (Binomial)
            # Generate mask for crossover
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter comes from the mutant (standard DE guarantee)
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            
            # Create trial vector U based on crossover mask
            trial = np.where(cross_points, mutant, population[i])
            
            # 4. Bound Handling
            # Clip values to ensure they stay within search space
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection
            trial_score = func(trial)
            
            # If trial is better or equal, replace the parent
            if trial_score <= fitness[i]:
                fitness[i] = trial_score
                population[i] = trial
                
                # Update global best if this is the best seen so far
                if trial_score < best:
                    best = trial_score

    # return fitness of the best found solution
    return best
