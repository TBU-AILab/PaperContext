#Here is a fully operational, self-contained Python algorithm based on **Differential Evolution (DE)**.
#
#Differential Evolution is chosen because it is a robust, gradient-free global optimization method that works well on continuous parameter spaces without requiring external optimization libraries (like SciPy). It is significantly more efficient than random search.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the minimum value of func using Differential Evolution algorithm 
    within max_time.
    """
    
    # --- Configuration ---
    # Setup time limits
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: usually 10*dim is good, but we cap it to ensure 
    # fast generations if dim is high or time is short.
    pop_size = max(10, int(5 * dim)) 
    mutation_factor = 0.8  # F
    crossover_prob = 0.9   # CR

    # --- Initialization ---
    
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population: random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    best_idx = -1

    # We evaluate the initial population one by one to respect the timer strictly
    # in case func() is very slow.
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            # If time is up before we finish init, return best found so far
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # --- Main Optimization Loop ---
    
    while True:
        # Loop through the population
        for i in range(pop_size):
            
            # Check time constraint at every individual step to be safe
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random agents (a, b, c) != i
            # We use a simple method to get indices excluding current i
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # Clip mutant to ensure it stays within bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover: Create trial vector u from target x[i] and mutant v
            # Pick random crossover points
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            trial_fitness = func(trial)
            
            # Greedy selection: if trial is better, replace existing individual
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    # return fitness of the best found solution
    return best_fitness
