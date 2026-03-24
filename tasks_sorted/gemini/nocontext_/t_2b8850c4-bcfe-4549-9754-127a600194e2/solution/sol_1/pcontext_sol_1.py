#Here is the fully operational, self-contained Python code implementing a Differential Evolution algorithm. This approach is robust for finding global minima of black-box functions within a fixed time frame.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the output of func using Differential Evolution within max_time.
    """
    start_time = time.time()
    
    # --- Hyperparameters ---
    # Population size: Scaled by dimension but capped to ensure speed 
    # if the function evaluation is slow or time is short.
    pop_size = min(max(20, 10 * dim), 50)
    mutation_factor = 0.8  # F
    crossover_prob = 0.7   # CR
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    
    # --- Initial Evaluation ---
    # We iterate manually to check time constraints strictly between evaluations
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # --- Main Optimization Loop ---
    while True:
        # Check time before generating next generation
        if time.time() - start_time >= max_time:
            return best_val

        # 1. Mutation: V = a + F * (b - c)
        # Generate random indices for a, b, c
        idxs_a = np.random.randint(0, pop_size, pop_size)
        idxs_b = np.random.randint(0, pop_size, pop_size)
        idxs_c = np.random.randint(0, pop_size, pop_size)
        
        # Create mutant vectors
        mutant = population[idxs_a] + mutation_factor * (population[idxs_b] - population[idxs_c])
        
        # Clip mutants to bounds to remain in valid search space
        mutant = np.clip(mutant, min_b, max_b)
        
        # 2. Crossover: Create trial vectors
        # Create a mask: True where we take from mutant, False from original
        cross_points = np.random.rand(pop_size, dim) < crossover_prob
        
        # Ensure at least one parameter is taken from the mutant (Standard DE rule)
        random_dims = np.random.randint(0, dim, pop_size)
        cross_points[np.arange(pop_size), random_dims] = True
        
        # Construct trial population
        trial_pop = np.where(cross_points, mutant, population)
        
        # 3. Selection
        for i in range(pop_size):
            # Strict time check before every function evaluation
            if time.time() - start_time >= max_time:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            # If trial is better, replace the individual in the population
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                # Update global best found so far
                if f_trial < best_val:
                    best_val = f_trial

    return best_val
