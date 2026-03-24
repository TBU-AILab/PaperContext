#Here is the fully operational and self-contained Python code implementing a Differential Evolution algorithm. It is designed to minimize a black-box function within a specified time limit using only standard libraries and NumPy.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    best = float('inf')

    # Differential Evolution Hyperparameters
    # pop_size: Set to balance between exploration capability and iteration speed.
    # We use dynamic sizing based on dimension, clamped to a safe range [10, 50].
    pop_size = max(10, min(50, 10 * dim))
    F = 0.8      # Mutation factor (typically 0.5-1.0)
    CR = 0.9     # Crossover probability (typically 0.8-1.0)

    # Convert bounds to numpy arrays for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # Initialize Population: random values within bounds
    # Shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))

    # Evaluate initial population
    for i in range(pop_size):
        # Time check
        if time.time() - start_time >= max_time:
            return best
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # Main Optimization Loop
    while True:
        # Iterate through each individual in the population
        for i in range(pop_size):
            # Strict time check before every new operation
            if time.time() - start_time >= max_time:
                return best

            # 1. Mutation (DE/rand/1 strategy)
            # Select 3 distinct random indices excluding current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Generate mutant vector
            mutant = pop[a] + F * (pop[b] - pop[c])

            # 2. Crossover (Binomial)
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter comes from the mutant to promote diversity
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            
            # Create trial vector
            trial = np.where(cross_points, mutant, pop[i])
            
            # 3. Boundary Handling
            # Clip values to ensure they stay within input bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection
            # Evaluate the trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better, replace current individual
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best if found
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
