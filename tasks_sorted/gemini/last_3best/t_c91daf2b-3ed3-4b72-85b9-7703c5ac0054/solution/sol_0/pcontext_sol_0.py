#Here is the fully operational and self-contained Python code implementing a **Differential Evolution** algorithm. This approach is generally more efficient and robust than random search for finding global minima in continuous search spaces.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function within given bounds and time limit using Differential Evolution.
    """
    
    # Record start time
    start_time = time.time()
    
    # --- Differential Evolution Hyperparameters ---
    # Population size: 
    # A size of 10 * dim is standard, but we clamp it (10 to 50) to ensure 
    # the algorithm runs enough generations within the time limit.
    pop_size = int(max(10, 10 * dim))
    if pop_size > 50:
        pop_size = 50
    
    F = 0.5    # Mutation factor (scale of differential vector, typ. 0.5-0.9)
    CR = 0.7   # Crossover probability (typ. 0.5-0.9)
    
    # --- Initialization ---
    # Convert bounds to numpy array for vectorization
    bounds_arr = np.array(bounds, dtype=float)
    min_bound = bounds_arr[:, 0]
    max_bound = bounds_arr[:, 1]
    diff_bound = max_bound - min_bound
    
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_bound + np.random.rand(pop_size, dim) * diff_bound
    
    # Array to store fitness of the population
    fitness = np.full(pop_size, float('inf'))
    
    # Track the global best value found so far
    best = float('inf')
    
    # --- Initial Evaluation ---
    # Evaluate the initial random population
    for i in range(pop_size):
        # Time check
        if time.time() - start_time >= max_time:
            return best
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val
            
    # --- Main Optimization Loop ---
    while True:
        # Iterate through each individual in the population
        for i in range(pop_size):
            
            # Strict time check inside the loop to adhere to max_time
            if time.time() - start_time >= max_time:
                return best
            
            # 1. Mutation
            # Select 3 distinct random indices (a, b, c) from population, excluding current index 'i'
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            
            # Create mutant vector: a + F * (b - c)
            mutant = population[a] + F * (population[b] - population[c])
            
            # 2. Crossover
            # Binomial crossover: mix target vector with mutant vector
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter is taken from the mutant (to ensure evolution)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial_vector = np.where(cross_points, mutant, population[i])
            
            # 3. Boundary Handling
            # Clip the trial vector to stay within the defined search space
            trial_vector = np.clip(trial_vector, min_bound, max_bound)
            
            # 4. Selection
            # Evaluate trial vector
            trial_fitness = func(trial_vector)
            
            # Greedy selection: if trial is better, replace the individual
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best if this is the new best solution
                if trial_fitness < best:
                    best = trial_fitness
                    
    return best
