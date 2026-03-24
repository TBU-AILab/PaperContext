#Here is the fully operational and self-contained algorithm using Differential Evolution (DE) to minimize the output value. DE is a robust global optimization method well-suited for black-box functions and handles continuous parameter spaces effectively within a limited time frame.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using the Differential Evolution algorithm
    within the specified 'max_time'.
    """
    # Start the execution timer
    start_time = time.time()
    
    # --- Algorithm Configuration ---
    # Differential Evolution parameters
    # Population size: heuristic based on dimension, clamped to [5, 50] to balance 
    # diversity against the time constraint.
    pop_size = int(10 * np.sqrt(dim))
    pop_size = max(5, pop_size)
    pop_size = min(50, pop_size)
    
    mutation_factor = 0.8  # F: Weight for differential mutation [0, 2]
    crossover_prob = 0.9   # CR: Crossover probability [0, 1]
    
    # Prepare bounds as numpy arrays for vectorized operations
    bounds_arr = np.array(bounds)
    min_bound = bounds_arr[:, 0]
    max_bound = bounds_arr[:, 1]
    diff_bound = max_bound - min_bound
    
    # --- Initialization ---
    # Initialize population with random uniform values within bounds
    # Shape: (pop_size, dim)
    population = min_bound + np.random.rand(pop_size, dim) * diff_bound
    
    # Array to store fitness of the population
    fitness = np.full(pop_size, float('inf'))
    
    # Track the global best fitness found
    best_fitness = float('inf')
    
    # Evaluate initial population
    # We check time within this loop to return early results if max_time is very short
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_fitness
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Evolution Loop ---
    # Iteratively improve the population until time runs out
    while True:
        for i in range(pop_size):
            # Check time constraints before every evaluation to ensure responsiveness
            if time.time() - start_time >= max_time:
                return best_fitness
            
            # 1. Mutation: Select 3 distinct random individuals (a, b, c) different from current (i)
            # Create a pool of indices excluding i
            idxs = np.arange(pop_size)
            candidates = idxs[idxs != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            # Generate mutant vector: v = a + F * (b - c)
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # 2. Constraints: Clip mutant vector to stay within bounds
            mutant = np.clip(mutant, min_bound, max_bound)
            
            # 3. Crossover: Create trial vector by mixing mutant and current target
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one dimension is taken from the mutant (DE requirement)
            cross_points[np.random.randint(dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])
            
            # 4. Selection: Evaluate trial vector
            f_trial = func(trial_vector)
            
            # Greedy selection: if trial is better, replace current individual
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_vector
                
                # Update global best if found
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    
    return best_fitness
