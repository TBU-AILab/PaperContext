#Here is a fully operational, self-contained Python implementation of the Differential Evolution algorithm. This algorithm is superior to random search for continuous optimization problems as it intelligently evolves the population of solutions towards the minimum.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the output of func using Differential Evolution within a time limit.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Higher allows more exploration but is slower per generation.
    # We use a heuristic based on dimension, capped to ensure responsiveness.
    pop_size = max(10, int(10 * dim))
    if pop_size > 50:
        pop_size = 50 
        
    mutation_factor = 0.8    # Scaling factor for mutation (F)
    crossover_prob = 0.7     # Crossover probability (CR)

    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # Generate initial population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    best_idx = -1

    # Evaluate initial population
    # We iterate manually to check time constraints during heavy initialization
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_fitness if best_idx != -1 else float('inf')
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # --- Algorithm Body (Main Loop) ---
    while True:
        # Iterate through the population
        for i in range(pop_size):
            # Time Check
            if (time.time() - start_time) >= max_time:
                return best_fitness

            # 1. Mutation
            # Select 3 random distinct indices distinct from current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Enforce bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover
            # Create trial vector by mixing target and mutant
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            # Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better, replace current
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if necessary
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
