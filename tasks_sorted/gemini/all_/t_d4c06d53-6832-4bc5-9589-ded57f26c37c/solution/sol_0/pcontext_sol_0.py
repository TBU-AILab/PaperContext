import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE).
    DE is a robust, gradient-free evolutionary algorithm suitable for 
    continuous optimization problems within a limited timeframe.
    """
    start = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- DE Hyperparameters ---
    # Population size: Standard is 10*dim, but we keep it adaptive and capped 
    # to ensure the algorithm performs enough generations within max_time.
    pop_size = max(10, 2 * dim)
    pop_size = min(pop_size, 50)  # Hard cap to prevent slow start on high dim
    
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_rate = 0.7   # Crossover probability (CR)

    # --- Pre-processing ---
    # Convert bounds to numpy arrays for efficient vector operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize storage
    population = np.zeros((pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    
    # --- Initialization Phase ---
    # We initialize individuals one by one to ensure we have a valid 'best'
    # result even if max_time is extremely short.
    for i in range(pop_size):
        if (datetime.now() - start) >= time_limit:
            return best
        
        # Generate random solution within bounds
        population[i] = min_b + np.random.rand(dim) * diff_b
        
        # Evaluate
        val = func(population[i])
        fitness[i] = val
        
        # Update global best
        if val < best:
            best = val

    # --- Evolution Loop ---
    # Strategy: DE/rand/1/bin
    while True:
        for i in range(pop_size):
            # Strict time check before every expensive function evaluation
            if (datetime.now() - start) >= time_limit:
                return best

            # 1. Mutation: Select 3 distinct random agents a, b, c from population where != i
            # Using random.sample on indices is generally faster than numpy choice for small sets
            idxs = list(range(pop_size))
            idxs.remove(i)
            a_idx, b_idx, c_idx = random.sample(idxs, 3)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Compute mutant vector: V = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Bound Constraint Handling: Clip mutant to stay within valid search space
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector U by mixing target X(i) and mutant V
            cross_points = np.random.rand(dim) < crossover_rate
            
            # Ensure at least one parameter comes from mutant (standard DE requirement)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection: Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better, replace parent
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if found
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
