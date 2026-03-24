#Here is a fully operational, self-contained implementation of the Differential Evolution algorithm. This algorithm is robust for finding global minima in continuous search spaces without requiring gradient information, making it suitable for black-box optimization within a time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Differential Evolution (DE) algorithm to minimize func within max_time.
    """
    start_time = time.time()
    
    # --- Hyperparameters ---
    # Population size: Higher is more robust but slower per generation.
    # We use a dynamic size capped at 50 to ensure iterations occur 
    # quickly enough even for high dimensions or short time limits.
    pop_size = max(10, min(50, 5 * dim))
    
    mutation_factor = 0.8    # Scaling factor (F)
    crossover_prob = 0.7     # Crossover probability (CR)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Generate initial population uniformly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track fitness and global best
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    
    # Evaluate initial population
    # We check time within this loop to handle very expensive functions or short limits
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            # If time is up, return best found so far. 
            # If no evaluations completed, evaluate one random solution.
            if best == float('inf'):
                return func(population[0])
            return best

        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Optimization Loop ---
    # Strategy: DE/rand/1/bin
    while True:
        for i in range(pop_size):
            # Check time constraint strictly before every evaluation
            if time.time() - start_time >= max_time:
                return best

            # 1. Mutation: Select 3 distinct random individuals (r1, r2, r3) != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Generate mutant vector: a + F * (b - c)
            mutant = population[r1] + mutation_factor * (population[r2] - population[r3])
            
            # Constraint handling: Clip values to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Binomial
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one parameter is taken from mutant to avoid stagnation
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection
            trial_fitness = func(trial_vector)

            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best immediately
                if trial_fitness < best:
                    best = trial_fitness

    return best
