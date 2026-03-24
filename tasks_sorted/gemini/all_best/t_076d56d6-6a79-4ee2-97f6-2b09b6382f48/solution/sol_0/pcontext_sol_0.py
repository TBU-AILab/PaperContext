#Here is a fully operational and self-contained Python algorithm using Differential Evolution (DE). DE is a robust global optimization method that generally outperforms random search by using the differences between randomly selected solutions to guide the search direction, making it efficient for finding minima in continuous search spaces within a limited time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function `func` within `max_time` seconds using 
    Differential Evolution.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: Standard is 10-20 * dim. We cap it to ensure 
    # many generations can run if max_time is short or dim is high.
    pop_size = min(max(10, 10 * dim), 50)
    mutation_factor = 0.8  # F: Weighting factor for mutation
    crossover_prob = 0.9   # CR: Crossover probability

    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff = max_b - min_b
    
    # Initialize Population uniformly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff
    fitness = np.full(pop_size, float('inf'))
    
    best = float('inf')

    # Evaluate Initial Population
    # We evaluate sequentially to respect the time limit strictly from the start
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # Main Evolution Loop
    while True:
        for i in range(pop_size):
            # Strict time check inside the loop
            if datetime.now() - start_time >= time_limit:
                return best
            
            # --- Mutation ---
            # Select 3 distinct random indices distinct from current 'i'
            # We create a pool and sample from it
            candidates = list(range(pop_size))
            candidates.remove(i)
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # DE/rand/1 mutation strategy
            mutant = a + mutation_factor * (b - c)
            
            # --- Crossover ---
            # Binomial crossover
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one dimension is changed to avoid stagnation
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # --- Bound Handling ---
            # Clip values to stay within the defined search space
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            trial_fitness = func(trial)
            
            # Greedy selection: if trial is better, replace the individual
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
