#Here is a fully operational, self-contained implementation using the **Differential Evolution** algorithm. This approach is significantly more robust and efficient than random search for finding global minima in continuous spaces, while remaining computationally lightweight enough to fit within time constraints.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Finds the input parameters that minimize func within max_time using 
    Differential Evolution (DE).
    """
    start_time = time.time()
    
    # --- DE Hyperparameters ---
    # Population size: generally 10x-15x dimension is a good rule of thumb.
    # We enforce a minimum of 4 because DE requires at least 4 distinct vectors
    # (target + 3 candidates) for mutation.
    pop_size = max(5, 10 * dim) 
    mutation_factor = 0.8  # 'F' parameter (typically 0.5 - 1.0)
    crossover_prob = 0.7   # 'CR' parameter (typically 0.5 - 0.9)

    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    bound_diff = upper_bounds - lower_bounds

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = lower_bounds + np.random.rand(pop_size, dim) * bound_diff
    
    # Array to store fitness values
    fitness = np.full(pop_size, float('inf'))
    
    # Track global best
    best_val = float('inf')
    best_vector = None

    # --- Initial Evaluation ---
    # We must evaluate the initial population to start the evolution.
    # We check time inside this loop to handle cases where func is very slow.
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vector = population[i].copy()

    # --- Main Evolution Loop ---
    while True:
        # Iterate through the population
        for i in range(pop_size):
            # Check time constraint before every evaluation
            if (time.time() - start_time) >= max_time:
                return best_val

            # 1. Mutation Strategy (DE/rand/1): v = a + F * (b - c)
            # Select 3 distinct random indices distinct from current index 'i'
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            
            mutant_vector = a + mutation_factor * (b - c)
            
            # Constrain mutant vector to bounds (clipping)
            mutant_vector = np.clip(mutant_vector, lower_bounds, upper_bounds)

            # 2. Crossover (Binomial)
            # Create a trial vector by mixing target vector and mutant vector
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter changes to avoid evaluating the same point
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # trial_vector takes from mutant where cross_points is True, else from population[i]
            trial_vector = np.where(cross_points, mutant_vector, population[i])

            # 3. Selection
            # Evaluate trial vector
            trial_fitness = func(trial_vector)

            # Greedy selection: if trial is better or equal, it replaces the target
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best if found
                if trial_fitness < best_val:
                    best_val = trial_fitness
                    best_vector = trial_vector

    return best_val
