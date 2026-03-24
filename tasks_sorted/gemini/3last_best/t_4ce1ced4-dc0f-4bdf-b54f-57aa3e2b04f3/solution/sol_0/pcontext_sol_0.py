#Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is generally more robust and efficient than random search for finding global minima in continuous search spaces within a limited time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using Differential Evolution within a time limit.
    """
    start_time = datetime.now()
    # We subtract a small buffer to ensure we return before the external timeout logic triggers
    time_limit = timedelta(seconds=max_time * 0.99)
    
    # Differential Evolution Hyperparameters
    # Population size: 15 * dim is a standard heuristic, clipped to reasonable limits
    # to ensure we get at least a few generations in if func is slow.
    pop_size = max(10, min(15 * dim, 100)) 
    mutation_factor = 0.8  # F: controls the amplification of the differential variation
    crossover_prob = 0.7   # CR: probability of crossover
    
    # Convert bounds to numpy arrays for vectorized operations
    bounds_array = np.array(bounds)
    min_b = bounds_array[:, 0]
    max_b = bounds_array[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate Initial Population
    fitness = np.full(pop_size, float('inf'))
    best_idx = 0
    best_fitness = float('inf')
    
    for i in range(pop_size):
        # Time check before expensive function call
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # Main DE Loop
    while True:
        for i in range(pop_size):
            # strict time check inside the loop
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant to bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover: Create trial vector u
            # Binomial crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter comes from the mutant to avoid duplicating the parent
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            # Calculate fitness of trial vector
            f_trial = func(trial)
            
            # If trial is better or equal, replace the target
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_idx = i

    return best_fitness
