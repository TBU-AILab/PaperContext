#Here is the fully operational, self-contained Python code implementing the Differential Evolution algorithm within the provided template.
#
#Differential Evolution is a robust, derivative-free global optimization method well-suited for finding the minimum of a function within a continuous search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the input parameters that minimize the output of func using 
    Differential Evolution within a specified time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: usually 10 to 15 times the dimension is effective.
    # We clip it between 10 and 50 to balance diversity and speed for time-constrained scenarios.
    pop_size = max(10, min(50, 15 * dim))
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_prob = 0.7   # Crossover probability (CR)

    # Pre-process bounds into numpy arrays for vectorization
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    bounds_diff = upper_bounds - lower_bounds

    # Initialize Population
    # Create random vectors uniformly distributed within the bounds
    population = lower_bounds + np.random.rand(pop_size, dim) * bounds_diff
    fitness = np.full(pop_size, float('inf'))
    
    best = float('inf')

    # Initial evaluation of the population
    for i in range(pop_size):
        # Check time constraint before every expensive function call
        if (datetime.now() - start_time) >= time_limit:
            return best

        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # Main Evolution Loop
    while True:
        # Check time at the start of each generation
        if (datetime.now() - start_time) >= time_limit:
            return best

        for i in range(pop_size):
            # Check time frequently (inside the loop) ensures we don't overrun 
            # if the population is large or func is slow
            if (datetime.now() - start_time) >= time_limit:
                return best

            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != i
            # We generate a list of candidates excluding current index i
            candidates = [idx for idx in range(pop_size) if idx != i]
            # Select 3 distinct indices
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Constrain mutant to bounds (clipping)
            mutant = np.clip(mutant, lower_bounds, upper_bounds)

            # 2. Crossover: Create trial vector
            # Select random dimensions to inherit from mutant based on crossover probability
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one dimension is changed to avoid duplicating the parent
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            trial_fitness = func(trial)

            # If trial is better or equal, replace the parent
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
