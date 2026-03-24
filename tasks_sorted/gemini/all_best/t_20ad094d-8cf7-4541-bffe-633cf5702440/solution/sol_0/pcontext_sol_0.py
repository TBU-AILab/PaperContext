#Here is the fully operational and self-contained Python code implementing a Differential Evolution algorithm. This approach is generally more efficient than random search for finding global minima in continuous search spaces within a limited time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: adaptive to dimension but capped to ensure iterations in limited time
    pop_size = max(10, min(50, int(5 * dim)))
    F = 0.8      # Mutation factor (differential weight)
    CR = 0.9     # Crossover probability

    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize Population (Uniform distribution within bounds)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best = float('inf')

    # Evaluate initial population
    for i in range(pop_size):
        # Check time constraint
        if datetime.now() - start >= limit:
            return best

        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # Main Evolution Loop
    while True:
        # Check time constraint at start of generation
        if datetime.now() - start >= limit:
            return best
            
        for i in range(pop_size):
            # Check time constraint before every individual evaluation
            # to maximize usage of available time without exceeding it significantly
            if datetime.now() - start >= limit:
                return best
            
            # 1. Mutation (DE/rand/1 strategy)
            # Select 3 distinct random individuals excluding current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            mutant = a + F * (b - c)
            
            # 2. Crossover (Binomial)
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter is updated from the mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Boundary Handling (Clip to bounds)
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            f_trial = func(trial)
            
            # Greedy selection: if trial is better or equal, it replaces the target
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if found
                if f_trial < best:
                    best = f_trial

    return best
