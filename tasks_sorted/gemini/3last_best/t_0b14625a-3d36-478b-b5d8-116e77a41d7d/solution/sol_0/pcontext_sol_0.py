#Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is robust for global optimization problems and works well within time constraints by iteratively improving a population of solutions.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a time limit.
    """
    
    # --- Configuration ---
    # Population size: Higher allows more exploration but slower generations.
    # 15 * dim is a standard rule of thumb. 
    # We clip it to ensure it's at least 5 (needed for mutation logic) and not excessive.
    pop_size = max(5, int(15 * dim))
    
    # DE Hyperparameters
    mutation_factor = 0.8  # (F) Weighting of the difference vector (0.5 - 1.0)
    crossover_prob = 0.7   # (CR) Probability of crossover (0.5 - 1.0)
    
    # Timing
    start_time = time.time()
    
    # --- Initialization ---
    
    # Prepare bounds as numpy arrays for vectorization
    bounds_np = np.array(bounds)
    lower_bounds = bounds_np[:, 0]
    upper_bounds = bounds_np[:, 1]
    diff_bounds = upper_bounds - lower_bounds
    
    # Initialize population: pop_size x dim
    # Random values scaled to the bounds
    population = lower_bounds + np.random.rand(pop_size, dim) * diff_bounds
    
    # Evaluate initial population
    fitness = np.zeros(pop_size)
    best_idx = 0
    best_fitness = float('inf')
    
    for i in range(pop_size):
        # Check time constraint during initialization
        if (time.time() - start_time) >= max_time:
            # If we time out during init, return best found so far (or inf if none)
            return best_fitness if best_fitness != float('inf') else func(population[i])
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # --- Main Optimization Loop ---
    
    while True:
        # Loop over the population
        for i in range(pop_size):
            
            # 1. Check Time Constraint strictly inside the loop
            if (time.time() - start_time) >= max_time:
                return best_fitness

            # 2. Mutation
            # Select 3 random distinct indices (a, b, c) that are not i
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Create mutant vector: V = X_a + F * (X_b - X_c)
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # Clip mutant to ensure it stays within bounds
            mutant = np.clip(mutant, lower_bounds, upper_bounds)
            
            # 3. Crossover
            # Create trial vector by mixing target vector (population[i]) and mutant
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is changed from the mutant to avoid stagnation
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # 4. Selection
            f_trial = func(trial)
            
            # Greedy selection: if trial is better or equal, replace current individual
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if needed
                if f_trial < best_fitness:
                    best_fitness = f_trial
        
    return best_fitness
