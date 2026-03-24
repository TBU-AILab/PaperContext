#Here is a fully operational, self-contained Python implementation of the Differential Evolution algorithm. This algorithm is well-suited for global optimization problems where the function is a "black box" and gradients are unknown. It is designed to continuously improve the solution until the `max_time` expires.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Differential Evolution (DE) algorithm to minimize func within max_time.
    Strategy: DE/rand/1/bin
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size (NP). 
    # A size of 10-15 * dim is standard, but we cap it to ensure 
    # we get enough generations if the time limit is tight or dim is high.
    pop_size = max(20, 5 * dim)
    if pop_size > 100: 
        pop_size = 100
        
    # Differential weight (F) usually between [0.5, 1.0]
    F = 0.8
    # Crossover probability (CR) usually between [0.8, 1.0]
    CR = 0.9

    # --- Initialization ---
    # Pre-process bounds for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track fitness of population
    fitnesses = np.full(pop_size, float('inf'))
    
    # Track global best
    best_fitness = float('inf')
    best_vector = None

    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Check time before every expensive function call
        if (datetime.now() - start_time) >= limit:
            return best_fitness

        val = func(population[i])
        fitnesses[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_vector = population[i].copy()

    # --- Main Evolution Loop ---
    # Continue evolving until time runs out
    while True:
        for i in range(pop_size):
            # Time check
            if (datetime.now() - start_time) >= limit:
                return best_fitness

            # 1. Mutation (DE/rand/1)
            # Select 3 random indices distinct from i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            x_a = population[a_idx]
            x_b = population[b_idx]
            x_c = population[c_idx]

            # Generate mutant vector
            mutant = x_a + F * (x_b - x_c)
            
            # Bound constraints: Clip mutant to stay inside bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover (Binomial)
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter is changed from the target vector
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            f_trial = func(trial)

            if f_trial <= fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial
                
                # Update global best if found
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
