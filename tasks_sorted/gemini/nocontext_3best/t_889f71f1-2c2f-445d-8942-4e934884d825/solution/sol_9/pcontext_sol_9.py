import numpy as np
import time

def run(func, dim, bounds, max_time):
    # Differential Evolution (DE) Algorithm
    # Parameters suitable for a wide range of problems
    pop_size = 15 * dim  # Population size proportional to dimension
    if pop_size < 10: pop_size = 10
    if pop_size > 50: pop_size = 50  # Cap population to ensure performance within time limit
    
    F = 0.6  # Differential weight (Mutation factor)
    CR = 0.8 # Crossover probability

    # Initialize timer
    start_time = time.time()
    
    # Pre-process bounds
    # bounds is a list of (min, max) pairs
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff = max_b - min_b

    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')

    # Evaluate initial population
    for i in range(pop_size):
        # Time check
        if time.time() - start_time >= max_time:
            return best_fitness

        val = func(population[i])
        fitness[i] = val

        if val < best_fitness:
            best_fitness = val

    # Main Evolution Loop
    while True:
        for i in range(pop_size):
            # Time check
            if time.time() - start_time >= max_time:
                return best_fitness

            # Mutation: DE/rand/1 strategy
            # Select 3 distinct random indices other than current i
            candidates = list(range(pop_size))
            candidates.remove(i)
            idxs = np.random.choice(candidates, 3, replace=False)
            
            x_1 = population[idxs[0]]
            x_2 = population[idxs[1]]
            x_3 = population[idxs[2]]
            
            mutant_vector = x_1 + F * (x_2 - x_3)
            
            # Crossover: Binomial
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one dimension is changed to maintain diversity
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial_vector = np.where(cross_points, mutant_vector, population[i])
            
            # Boundary handling: Clip to defined bounds
            trial_vector = np.clip(trial_vector, min_b, max_b)
            
            # Selection (Greedy)
            try:
                trial_fitness = func(trial_vector)
            except Exception:
                # Handle cases where func might fail on specific inputs
                trial_fitness = float('inf')
            
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
