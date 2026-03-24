#Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is highly effective for global optimization of continuous functions without requiring gradients, and it is robust enough to find good solutions within a limited time frame.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the minimum value of func using Differential Evolution within max_time.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Algorithm Parameters ---
    # Population size: Keep it moderate (15-20) to ensure generations update 
    # frequently enough within the time limit, but large enough for diversity.
    pop_size = max(15, dim * 2) 
    mutation_factor = 0.8  # Differential weight (F)
    crossover_prob = 0.9   # Crossover probability (CR)

    # --- Initialization ---
    best_fitness = float('inf')
    
    # Pre-process bounds into numpy arrays for faster vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = np.fabs(max_b - min_b)

    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitnesses = np.full(pop_size, float('inf'))

    # --- Initial Evaluation Loop ---
    # We evaluate the initial population carefully checking time after each step
    # to ensure we return a result even if max_time is extremely short.
    for i in range(pop_size):
        # Check time constraint
        if datetime.now() - start_time >= time_limit:
            return best_fitness

        val = func(population[i])
        fitnesses[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Differential Evolution Loop ---
    while True:
        # If time is up, break immediately
        if datetime.now() - start_time >= time_limit:
            break

        # Iterate through the population to create new candidate solutions
        for i in range(pop_size):
            # Check time periodically inside the generation loop
            if datetime.now() - start_time >= time_limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random vectors (a, b, c) excluding current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # 2. Crossover: Create trial vector by mixing target and mutant
            # Pick random crossover points
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one dimension is changed (taken from mutant)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Boundary Handling: Clip values to stay within bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection: Evaluate trial vector
            f_trial = func(trial)

            # If trial is better, replace the individual in the population
            if f_trial < fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial
                
                # Update global best found so far
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
