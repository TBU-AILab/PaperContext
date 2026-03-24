import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function 'func' using Differential Evolution (DE).
    DE is a stochastic, population-based optimization algorithm suitable for 
    continuous global optimization problems.
    """
    
    # Initialize timing
    start = datetime.now()
    limit = timedelta(seconds=max_time)

    # -------------------------------------------------------------------------
    # Algorithm Configuration (DE/rand/1/bin strategy)
    # -------------------------------------------------------------------------
    # Population size: Higher allows more exploration but is slower per generation.
    # 10 * dim is a standard rule of thumb.
    pop_size = max(10, int(10 * dim))
    
    # Differential weight (F): Scaling factor for mutation (typically [0.5, 1.0])
    mutation_factor = 0.8
    
    # Crossover probability (CR): Fraction of parameters to shuffle (typically [0.8, 1.0])
    crossover_prob = 0.9

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    
    # We define a helper to safely check time usage
    def check_timeout():
        return (datetime.now() - start) >= limit

    # Evaluate initial population
    for i in range(pop_size):
        if check_timeout():
            # If we timeout during initialization, return best found so far
            return best if best != float('inf') else fitness[0]
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------
    while not check_timeout():
        for i in range(pop_size):
            # Strict time check inside the loop to ensure we don't overrun
            if check_timeout():
                return best

            # 1. Mutation
            # Select 3 distinct indices excluding the current index i
            candidates = list(range(pop_size))
            candidates.remove(i)
            # Pick a, b, c random vectors
            idxs = np.random.choice(candidates, 3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Constrain mutant to bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover (Binomial)
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one dimension is changed to maintain diversity
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            trial_fitness = func(trial)

            # Greedy selection: if trial is better or equal, replace current
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
