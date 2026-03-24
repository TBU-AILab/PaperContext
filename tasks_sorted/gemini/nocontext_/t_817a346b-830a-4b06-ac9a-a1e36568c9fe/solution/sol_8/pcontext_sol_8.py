#Here is a fully operational, self-contained Python algorithm using **Differential Evolution**. This method is generally much more efficient than random search for finding global minima in continuous spaces, as it uses the differences between existing solutions to navigate the search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a time limit.
    """
    # --- Configuration ---
    # F: Differential weight [0, 2], typically 0.8
    # CR: Crossover probability [0, 1], typically 0.9
    F = 0.8
    CR = 0.9
    
    # Population size. 10x to 15x dimension is standard, but we clamp it 
    # to ensure we get some iterations done if dim is very high.
    pop_size = max(5, min(10 * dim, 100))

    # --- Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # Helper to check if we have time left
    def has_time():
        return (datetime.now() - start_time) < time_limit

    # Prepare bounds as numpy arrays for vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = np.fabs(max_b - min_b)

    # Initialize population: Random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    best_idx = -1

    # We evaluate initial population one by one to check time constraints safely
    for i in range(pop_size):
        if not has_time():
            # If we run out of time during initialization, return best found so far
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Optimization Loop ---
    # Strategy: DE/rand/1/bin
    while has_time():
        for i in range(pop_size):
            if not has_time():
                return best_val

            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != i
            # We create a list of available indices excluding the current index 'i'
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)

            # Create mutant vector: v = x_a + F * (x_b - x_c)
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            mutant = x_a + F * (x_b - x_c)

            # Clip mutant to bounds to ensure valid input
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR
            
            # Force at least one parameter to change to ensure diversity
            # (pick a random dimension index)
            j_rand = np.random.randint(0, dim)
            cross_points[j_rand] = True

            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            trial_fitness = func(trial)

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # Update global best if this new individual is the best seen so far
                if trial_fitness < best_val:
                    best_val = trial_fitness

    return best_val
