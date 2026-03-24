import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size is adapted to dimension but capped to ensure responsiveness
    pop_size = 15 * dim
    if pop_size > 50:
        pop_size = 50
    if pop_size < 10:
        pop_size = 10
        
    F = 0.8   # Mutation factor
    CR = 0.9  # Crossover probability
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population: shape (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Check time constraint strictly
        if datetime.now() - start_time >= limit:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            
    # Main Evolution Loop
    while True:
        # Check time at start of generation
        if datetime.now() - start_time >= limit:
            return best_val
            
        for i in range(pop_size):
            # Check time periodically within generation
            if (datetime.now() - start_time >= limit):
                return best_val

            # Mutation Strategy: DE/rand/1/bin
            # Select 3 distinct random indices excluding current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            mutant = a + F * (b - c)
            
            # Enforce bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # Crossover
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter is changed
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])
            
            # Evaluation
            trial_fitness = func(trial_vector)
            
            # Selection
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best
                if trial_fitness < best_val:
                    best_val = trial_fitness
                    
    return best_val
