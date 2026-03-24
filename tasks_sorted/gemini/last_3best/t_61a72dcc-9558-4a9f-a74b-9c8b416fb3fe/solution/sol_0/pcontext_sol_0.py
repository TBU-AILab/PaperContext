import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm within a specified time limit.
    """
    start_time = time.time()
    
    # --- Hyperparameters ---
    # Population size: Balance between exploration (large pop) and speed (small pop).
    # We scale with dimension but cap it to ensure the loop runs fast enough.
    pop_size = int(max(10, min(100, 10 * dim))) 
    mutation_factor = 0.8  # 'F': Controls differential weight
    crossover_prob = 0.7   # 'CR': Controls probability of crossover

    # --- Initialization ---
    # Convert bounds to numpy arrays for efficient broadcasting
    bounds_arr = np.array(bounds)
    lower_bounds = bounds_arr[:, 0]
    upper_bounds = bounds_arr[:, 1]
    bound_diff = upper_bounds - lower_bounds

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = lower_bounds + np.random.rand(pop_size, dim) * bound_diff
    
    fitnesses = np.full(pop_size, float('inf'))
    best = float('inf')

    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Time check
        if time.time() - start_time >= max_time:
            return best
        
        score = func(population[i])
        fitnesses[i] = score
        
        if score < best:
            best = score

    # --- Evolution Loop ---
    while True:
        # Iterate through each individual in the population
        for i in range(pop_size):
            # Check time constraints frequently (before every expensive function call)
            if time.time() - start_time >= max_time:
                return best

            # 1. Mutation (Strategy: DE/rand/1/bin)
            # Select 3 distinct random indices other than the current individual 'i'
            # Note: np.random.choice is safer, but rand ints are faster. 
            # Given time constraints, we ensure distinctness via choice.
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Create mutant vector: V = X_a + F * (X_b - X_c)
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            
            # Bound constraints: Clip values to stay within specified bounds
            mutant = np.clip(mutant, lower_bounds, upper_bounds)

            # 2. Crossover
            # Create trial vector by mixing target (population[i]) and mutant
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is taken from the mutant (standard DE rule)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection
            # Evaluate the trial vector
            trial_fitness = func(trial_vector)

            # Greedy selection: if trial is better or equal, replace the individual
            if trial_fitness <= fitnesses[i]:
                fitnesses[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
