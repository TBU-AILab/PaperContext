#Here is a fully operational, self-contained Python algorithm implementing **Differential Evolution (DE)**. DE is a robust global optimization method that works well without gradient information and is significantly more efficient than random search.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Differential Evolution Configuration ---
    # Population size: heuristic based on dimension, capped to ensure speed
    pop_size = max(5, 10 * dim) 
    if pop_size > 40: 
        pop_size = 40
        
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_prob = 0.9   # Crossover probability (CR)
    
    # Prepare bounds as numpy arrays for vectorized operations
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    bound_diff = upper_bounds - lower_bounds
    
    # --- Initialization ---
    # Generate random initial population within bounds
    # Shape: (pop_size, dim)
    population = lower_bounds + np.random.rand(pop_size, dim) * bound_diff
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')

    # Evaluate initial population
    for i in range(pop_size):
        if datetime.now() - start_time >= limit:
            return best_fitness

        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Optimization Loop ---
    while True:
        for i in range(pop_size):
            # Check time constraint strictly before every new operation
            if datetime.now() - start_time >= limit:
                return best_fitness

            # 1. Mutation: DE/rand/1 strategy
            # Select 3 random indices distinct from current index i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Generate mutant vector
            mutant = a + mutation_factor * (b - c)
            
            # Enforce bounds (clipping)
            mutant = np.clip(mutant, lower_bounds, upper_bounds)
            
            # 2. Crossover: Binomial
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one gene is mutated (avoid duplication of parent)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Create trial vector
            trial_vector = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            # Evaluate trial vector
            trial_fitness = func(trial_vector)
            
            # Greedy selection: if trial is better, replace current
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best if necessary
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
