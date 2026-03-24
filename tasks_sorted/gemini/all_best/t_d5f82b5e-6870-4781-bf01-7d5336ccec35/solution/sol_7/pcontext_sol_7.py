import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    
    # Differential Evolution Parameters
    # Adjust population size based on dimension, capped to balance exploration and speed
    pop_size = 15 * dim
    if pop_size > 50: pop_size = 50
    if pop_size < 10: pop_size = 10
    
    F = 0.8   # Mutation factor
    CR = 0.7  # Crossover probability

    # Convert bounds to numpy arrays for efficient calculation
    bounds_array = np.array(bounds)
    lower_bound = bounds_array[:, 0]
    upper_bound = bounds_array[:, 1]
    bound_diff = upper_bound - lower_bound

    # Initialize Population
    population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
    fitness = np.full(pop_size, np.inf)
    
    best_val = np.inf

    # Initial evaluation loop with strict time checking
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
            
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val

    # Main Optimization Loop
    while (time.time() - start_time) < max_time:
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val

            # Mutation strategy: DE/rand/1/bin
            # Select 3 distinct random indices other than current i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            x_r1 = population[r1]
            x_r2 = population[r2]
            x_r3 = population[r3]

            mutant_vector = x_r1 + F * (x_r2 - x_r3)
            
            # Handle boundary constraints by clipping
            mutant_vector = np.clip(mutant_vector, lower_bound, upper_bound)

            # Crossover
            cross_mask = np.random.rand(dim) < CR
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial_vector = np.where(cross_mask, mutant_vector, population[i])

            # Selection
            trial_fitness = func(trial_vector)

            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                if trial_fitness < best_val:
                    best_val = trial_fitness

    return best_val
