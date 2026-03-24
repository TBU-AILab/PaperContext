import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution algorithm within a specified time limit.
    """
    start = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: adaptive but capped to ensure iterations occur even if func is slow
    pop_size = max(10, int(dim * 2))
    if pop_size > 40:
        pop_size = 40
        
    mutation_factor = 0.8
    crossover_prob = 0.9

    # Prepare bounds as numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize Population
    # pop is a matrix of shape (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_idx = -1

    # Evaluate initial population
    # We loop explicitly to check time after every function call
    for i in range(pop_size):
        if (datetime.now() - start) >= time_limit:
            return best_fitness

        val = func(pop[i])
        fitness[i] = val

        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # Main Optimization Loop
    while True:
        for i in range(pop_size):
            # Check time constraint strictly before every evaluation
            if (datetime.now() - start) >= time_limit:
                return best_fitness

            # 1. Mutation: DE/rand/1 strategy
            # Select 3 random distinct indices not equal to i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)

            mutant = pop[a] + mutation_factor * (pop[b] - pop[c])

            # Boundary Constraint Handling (Clamping)
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one dimension is changed
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            # Construct trial vector
            trial = np.where(cross_points, mutant, pop[i])

            # 3. Selection
            f_trial = func(trial)

            # Greedy selection: if trial is better or equal, replace current
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best found so far
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
