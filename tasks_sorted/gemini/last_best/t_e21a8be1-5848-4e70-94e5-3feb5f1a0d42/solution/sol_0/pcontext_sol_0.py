#Here is a fully operational, self-contained Python implementation using **Differential Evolution**.
#
#Differential Evolution is a robust global optimization algorithm that does not require gradients. It works by maintaining a population of candidate solutions and iteratively improving them by mixing existing candidates (mutation and crossover) to explore the search space. This implementation strictly adheres to the provided template and time constraints.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm.
    
    Args:
        func: The objective function to minimize.
        dim: Integer dimension of the input vector.
        bounds: List of [low, high] for each dimension.
        max_time: Maximum run time in seconds.
        
    Returns:
        The best fitness value found (float).
    """
    
    # --- Configuration ---
    # Population size: Compromise between diversity and speed.
    # 10 * dim is standard, but we cap at 50 to ensure many generations 
    # run if time is short or evaluation is slow.
    pop_size = max(10, min(50, 10 * dim))
    
    # DE hyperparameters (Standard strategy: rand/1/bin)
    F = 0.8         # Differential weight [0, 2]
    CR = 0.9        # Crossover probability [0, 1]
    
    # Time management
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    
    # We loop to evaluate initial population to respect time limits immediately
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_fitness
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Optimization Loop ---
    while True:
        # Check time at the start of every generation
        if datetime.now() - start_time >= time_limit:
            return best_fitness
            
        for i in range(pop_size):
            # Check time frequently (inside the loop) for expensive functions
            if datetime.now() - start_time >= time_limit:
                return best_fitness
            
            # 1. Mutation: Select 3 distinct random individuals a, b, c where idx != i
            # We select indices efficiently
            candidates = np.random.randint(0, pop_size, 3)
            
            # Simple retry logic to ensure distinct indices distinct from i
            # This is generally faster than np.random.choice on small sets inside loops
            while len(set(candidates)) < 3 or i in candidates:
                candidates = np.random.randint(0, pop_size, 3)
            
            a, b, c = candidates
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            # Create mutant vector: v = x_a + F * (x_b - x_c)
            mutant = x_a + F * (x_b - x_c)
            
            # Clip mutant vector to bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover: Binomial crossover
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter is changed
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
