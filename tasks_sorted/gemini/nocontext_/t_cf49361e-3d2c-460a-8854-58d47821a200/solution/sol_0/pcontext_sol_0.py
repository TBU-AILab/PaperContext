#Here is a fully operational, self-contained implementation using the **Differential Evolution** algorithm. This approach is generally superior to random search for continuous optimization problems as it balances exploration (searching the whole space) and exploitation (refining good solutions) without requiring gradients.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using Differential Evolution within 'max_time'.
    """
    
    # --- Hyperparameters ---
    # F: Differential weight (scaling factor), typically 0.5 to 1.0
    F = 0.8  
    # CR: Crossover probability, typically 0.7 to 0.9
    CR = 0.9 
    
    # Population size: 
    # Must be sufficient to explore dimensions, but limited to ensure 
    # the algorithm runs enough generations within max_time.
    pop_size = max(10, int(dim * 10))
    if pop_size > 50:
        pop_size = 50

    # --- Initialization ---
    start_time = time.time()
    
    # Convert bounds to numpy arrays for efficient vectorization
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Store fitness for each individual
    fitness = np.full(pop_size, float('inf'))
    
    # Track the global best
    best_fitness = float('inf')

    # --- Initial Population Evaluation ---
    for i in range(pop_size):
        # Time check before expensive function call
        if time.time() - start_time >= max_time:
            # If we time out immediately, ensure we return a valid float if possible
            return best_fitness if best_fitness != float('inf') else func(population[0])

        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Evolution Loop ---
    while True:
        for i in range(pop_size):
            # Check remaining time
            if time.time() - start_time >= max_time:
                return best_fitness

            # 1. Mutation: Select 3 distinct random indices (a, b, c) != i
            # Using manual selection loop is efficient for small sample sizes
            idxs = [i]
            while len(idxs) < 4:
                candidate = np.random.randint(0, pop_size)
                if candidate not in idxs:
                    idxs.append(candidate)
            
            a, b, c = idxs[1], idxs[2], idxs[3]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = population[a] + F * (population[b] - population[c])
            
            # Clip mutant vector to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover
            # Create trial vector by mixing target vector and mutant vector
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter is taken from the mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            f_trial = func(trial)
            
            # Greedy selection: if trial is better or equal, it replaces the parent
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if a new minimum is found
                if f_trial < best_fitness:
                    best_fitness = f_trial

    # return fitness of the best found solution
    return best_fitness
