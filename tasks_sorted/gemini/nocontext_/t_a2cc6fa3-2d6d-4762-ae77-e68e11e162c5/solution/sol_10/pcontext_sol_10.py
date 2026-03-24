#Here is the corrected and fully operational Python code implementing a Differential Evolution algorithm. This algorithm is robust for finding global minima and adheres strictly to the provided template and variable naming rules.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Record start time to enforce the time limit
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Differential Evolution Parameters ---
    # F: Mutation factor (typically 0.5 to 1.0)
    # CR: Crossover probability (typically 0.7 to 0.9)
    F = 0.8
    CR = 0.9
    
    # Population size: usually 10-15 times the dimension, capped for performance
    pop_size = max(10, 15 * dim)
    if pop_size > 100: 
        pop_size = 100

    # --- Initialization ---
    # Extract lower and upper bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    bound_diff = upper_bounds - lower_bounds
    
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = lower_bounds + np.random.rand(pop_size, dim) * bound_diff
    fitness = np.full(pop_size, float('inf'))
    
    best_value = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Check time constraint
        if (datetime.now() - start_time) >= time_limit:
            return best_value if best_value != float('inf') else float('inf')

        val = func(population[i])
        fitness[i] = val
        if val < best_value:
            best_value = val

    # --- Main Loop ---
    while True:
        for i in range(pop_size):
            # Check time constraint frequently
            if (datetime.now() - start_time) >= time_limit:
                return best_value

            # 1. Mutation
            # Select 3 distinct random individuals (a, b, c) excluding current (i)
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            x_a = population[a_idx]
            x_b = population[b_idx]
            x_c = population[c_idx]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = x_a + F * (x_b - x_c)
            
            # Enforce bounds
            mutant = np.clip(mutant, lower_bounds, upper_bounds)
            
            # 2. Crossover
            # Create trial vector by mixing current vector and mutant
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one dimension is changed
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            trial_fitness = func(trial)
            
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                # Update global best
                if trial_fitness < best_value:
                    best_value = trial_fitness

    return best_value
