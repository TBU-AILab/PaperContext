#Here is a fully operational, self-contained Python algorithm implementing **Differential Evolution**. This method is robust for global optimization problems where derivatives are unknown, handles non-linear and multi-modal functions well, and efficiently uses the allotted time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Differential Evolution within max_time.
    """
    # Initialize timing
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: Scale with dimension, but cap to ensure iterations run within time limits
    pop_size = max(5, int(10 * dim))
    if pop_size > 50: 
        pop_size = 50
        
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_rate = 0.9   # Crossover probability (CR)

    # Pre-process bounds for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    bound_diff = np.fabs(max_b - min_b)

    # Initialize Population
    # Create random vectors within the bounds
    population = min_b + np.random.rand(pop_size, dim) * bound_diff
    
    # Initialize fitness array and global best
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Time check
        if datetime.now() - start_time >= limit:
            return best
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # Main Evolution Loop
    while True:
        # Check time at start of generation
        if datetime.now() - start_time >= limit:
            return best
            
        # Iterate through population
        for i in range(pop_size):
            # Check time frequently inside the loop (crucial if func is slow)
            if datetime.now() - start_time >= limit:
                return best

            # 1. Mutation: Select 3 distinct random vectors a, b, c != i
            # We create a list of available indices excluding current 'i'
            candidates = list(range(pop_size))
            candidates.remove(i)
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Constrain mutant to bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector
            # Select random crossover points
            cross_points = np.random.rand(dim) < crossover_rate
            
            # Ensure at least one parameter comes from the mutant to ensure evolution
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            trial_fitness = func(trial)

            # Greedy selection: if trial is better or equal, replace parent
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness

    return best
