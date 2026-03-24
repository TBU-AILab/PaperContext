import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Executes a Differential Evolution algorithm to minimize the given function 
    within a specified time limit.
    """
    start_time = time.time()
    
    # --- Algorithm Configuration ---
    # Population size: Heuristic based on dimension, clamped to reasonable limits
    # to ensure efficiency within time constraints.
    pop_size = min(max(20, 10 * dim), 100)
    
    # Differential Evolution Hyperparameters
    F = 0.8   # Mutation factor (differential weight)
    CR = 0.9  # Crossover probability

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Array to store fitness values
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')

    # Evaluate initial population
    # We check time during initialization in case calculation is very slow
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # --- Main Optimization Loop ---
    # We use Asynchronous update (updates population immediately) 
    # allowing for granular time checks.
    while True:
        for i in range(pop_size):
            # Check time constraint
            if (time.time() - start_time) >= max_time:
                return best_val

            # 1. Mutation: Select 3 distinct random agents (a, b, c) != i
            # Using random choice is safer to ensure distinctness
            candidates = np.random.choice(pop_size, 3, replace=False)
            while i in candidates:
                candidates = np.random.choice(pop_size, 3, replace=False)
            
            a_idx, b_idx, c_idx = candidates
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)

            # 2. Crossover: Create trial vector
            # Generate random mask for crossover
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter comes from mutant (avoids exact copy of target)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Boundary Control: Clip values to stay within bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection
            trial_fitness = func(trial)

            # If trial is better, replace the individual in the population
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best
                if trial_fitness < best_val:
                    best_val = trial_fitness

    return best_val
