#Here is the corrected and fully operational Python code. This implementation uses a Differential Evolution algorithm, which is robust for finding global minima. It avoids nested function scopes to prevent the `nonlocal` syntax error completely.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function 'func' within 'bounds' using Differential Evolution
    under a time constraint 'max_time'.
    """
    # Record the start time
    start_time = time.time()
    
    # ---------------------------
    # Algorithm Configuration
    # ---------------------------
    # Population size: larger is more robust but slower per generation
    # We adapt it to dimension but cap it to ensure many generations within time limit
    pop_size = max(10, min(dim * 10, 50)) 
    mutation_factor = 0.6  # F
    crossover_prob = 0.7   # CR
    
    # ---------------------------
    # Initialization
    # ---------------------------
    # Convert bounds to numpy arrays for efficient vector operations
    bounds_np = np.array(bounds)
    lower_bound = bounds_np[:, 0]
    upper_bound = bounds_np[:, 1]
    bound_diff = upper_bound - lower_bound
    
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
    
    # Array to store fitness of each individual
    fitness = np.full(pop_size, float('inf'))
    
    # Variable to track global best
    best_fitness = float('inf')
    
    # ---------------------------
    # Initial Evaluation
    # ---------------------------
    for i in range(pop_size):
        # Check time constraint strictly
        if time.time() - start_time >= max_time:
            return best_fitness
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # ---------------------------
    # Main Optimization Loop
    # ---------------------------
    while True:
        # Check time at start of generation
        if time.time() - start_time >= max_time:
            break
            
        for i in range(pop_size):
            # Check time frequently (inside the loop)
            if time.time() - start_time >= max_time:
                return best_fitness
            
            # 1. Mutation: Select 3 distinct random individuals (excluding current i)
            # Create a list of available indices
            candidates = list(range(pop_size))
            candidates.remove(i)
            # Randomly select 3 indices
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            # Extract vectors
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Generate mutant vector: V = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Ensure mutant stays within bounds
            mutant = np.clip(mutant, lower_bound, upper_bound)
            
            # 2. Crossover: Create trial vector
            # Select indices to perform crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection: Evaluate trial vector
            trial_fitness_val = func(trial)
            
            # If trial is better, replace the individual
            if trial_fitness_val < fitness[i]:
                fitness[i] = trial_fitness_val
                population[i] = trial
                
                # Update global best if necessary
                if trial_fitness_val < best_fitness:
                    best_fitness = trial_fitness_val

    return best_fitness
