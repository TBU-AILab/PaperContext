#Here is a fully operational implementation using the **Differential Evolution** algorithm. 
#
#This algorithm is superior to random search for continuous optimization problems. It uses a population of candidate solutions and iteratively improves them by mixing vector differences (mutation) and performing crossover, effectively navigating complex fitness landscapes without requiring gradients.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a time limit.
    """
    start_time = time.time()
    
    # --- DE Hyperparameters ---
    # Population size: larger allows more exploration but is slower per generation.
    # We adapt it slightly based on dimension but cap it to ensure agility.
    pop_size = max(10, int(10 * dim))
    if pop_size > 50:
        pop_size = 50
        
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_rate = 0.9   # Crossover probability (CR)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population uniformly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track fitness of the population
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Optimization Loop ---
    while True:
        # Check time overhead before performing matrix operations
        if (time.time() - start_time) >= max_time:
            return best

        # 1. Mutation: Mutant = a + F * (b - c)
        # We select random parents using vectorized indices for speed.
        # While DE usually requests distinct parents (a!=b!=c!=i), random selection
        # is a standard high-performance approximation in Python.
        idx_a = np.random.randint(0, pop_size, pop_size)
        idx_b = np.random.randint(0, pop_size, pop_size)
        idx_c = np.random.randint(0, pop_size, pop_size)

        a = population[idx_a]
        b = population[idx_b]
        c = population[idx_c]

        mutants = a + mutation_factor * (b - c)
        
        # Clip mutants to stay within search space bounds
        mutants = np.clip(mutants, min_b, max_b)

        # 2. Crossover
        # Generate mask: True means take from mutant, False from original
        cross_points = np.random.rand(pop_size, dim) < crossover_rate
        
        # Construct trial vectors
        trials = np.where(cross_points, mutants, population)

        # 3. Selection
        for i in range(pop_size):
            # Critical time check inside the evaluation loop
            if (time.time() - start_time) >= max_time:
                return best
            
            # Evaluate trial vector
            trial_fitness = func(trials[i])

            # Greedy selection: if trial is better, replace existing individual
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trials[i]
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
