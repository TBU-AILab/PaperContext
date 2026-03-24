#Here is a fully operational and self-contained algorithm implementing **Differential Evolution (DE)** to solve the optimization problem. DE is a robust, global optimization method well-suited for black-box functions and continuous parameter spaces.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a black-box function using the Differential Evolution algorithm
    within a specified time limit.
    """
    start_time = time.time()
    
    # --- Algorithm Configuration ---
    # Population size: A trade-off between search diversity and speed.
    # We use a dynamic size based on dimension but capped at 50 to ensure
    # the algorithm performs sufficient generations if func is computationally expensive.
    pop_size = max(10, min(50, 5 * dim))
    
    # DE Hyperparameters (Standard values)
    mutation_factor = 0.8  # F: Weight for differential mutation
    crossover_prob = 0.7   # CR: Probability of crossover
    
    # --- Initialization ---
    # Convert bounds to numpy arrays for efficient vector operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Initialize population uniformly within bounds
    # Shape: (pop_size, dim)
    population = np.random.uniform(low=min_b, high=max_b, size=(pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    
    # --- Initial Evaluation ---
    # Evaluate the starting population, checking time constraints strictly
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # --- Main Optimization Loop ---
    while True:
        # Iterate through each individual in the population
        for i in range(pop_size):
            # Check time limit at every step to ensure strict adherence
            if time.time() - start_time >= max_time:
                return best_val
            
            # 1. Mutation: Select 3 distinct random individuals (a, b, c) distinct from i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector: V = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # 2. Crossover: Create trial vector by mixing mutant and target
            rand_vals = np.random.rand(dim)
            cross_points = rand_vals < crossover_prob
            
            # Ensure at least one parameter comes from the mutant (DE requirement)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Boundary Handling: Clip values to stay within bounds
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection: Evaluate trial vector
            trial_fitness = func(trial)
            
            # Greedy selection: if trial is better or equal, replace target
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best found so far
                if trial_fitness < best_val:
                    best_val = trial_fitness
                    
    return best_val
