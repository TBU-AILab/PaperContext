#Here is a fully operational, self-contained Python implementation using **Differential Evolution (DE)**.
#
#Differential Evolution is a robust, stochastic global optimization algorithm that works well for continuous parameter spaces without requiring gradient information. It is generally more efficient than random search for finding minima in multi-dimensional spaces.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the minimum of a function using Differential Evolution within a time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Algorithm Parameters ---
    # Population size: larger means more diversity but slower generations.
    # We scale it with dimension but cap it to ensure iterations run within time limits.
    pop_size = max(10, 15 * dim)
    pop_size = min(pop_size, 100) # Cap population to prevent slow initialization
    
    mutation_factor = 0.8     # F: Controls the amplification of differential variation
    crossover_prob = 0.7      # CR: Probability of crossover
    
    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population fitness
    fitness = np.full(pop_size, float('inf'))
    
    # We must check time even during initialization if evaluation is very slow
    best = float('inf')
    best_idx = -1
    
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best if best != float('inf') else func(population[0])
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val
            best_idx = i

    # --- Evolution Loop ---
    while True:
        # Check time constraint at the start of each generation
        if (datetime.now() - start_time) >= time_limit:
            return best
            
        # Iterate through the population
        for i in range(pop_size):
            # Check time frequently (inside loop) to adhere strictly to max_time
            if (datetime.now() - start_time) >= time_limit:
                return best
            
            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != i
            # We use a quick method to select indices
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            
            # Create mutant vector: V = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant to ensure it stays within bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover: Create trial vector
            # Recombine target (population[i]) and mutant
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is changed (taken from mutant)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            # Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better, replace target
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best found so far
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
