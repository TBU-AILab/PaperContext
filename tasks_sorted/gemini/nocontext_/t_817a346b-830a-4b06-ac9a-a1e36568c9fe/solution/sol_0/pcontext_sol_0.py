#Here is a fully operational, self-contained Python algorithm based on the **Differential Evolution** (DE) strategy. DE is a robust metaheuristic for global optimization that does not require gradients and works well within defined bounds and time constraints.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution algorithm 
    within a specified time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Hyperparameters ---
    # Population size: Balance between diversity and speed. 
    # We use dynamic sizing but cap it to ensure iterations run quickly.
    pop_size = min(max(20, 5 * dim), 100) 
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_rate = 0.9   # Crossover probability (CR)

    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    bounds_diff = max_b - min_b

    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * bounds_diff
    
    # Store fitness of population
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')

    # --- Initial Evaluation ---
    # We must check time even during initialization in case 'func' is very slow
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Optimization Loop ---
    # DE Strategy: mutant = a + F * (b - c)
    while True:
        for i in range(pop_size):
            # Check time limit before every heavy calculation
            if datetime.now() - start_time >= time_limit:
                return best

            # 1. Mutation: Select 3 distinct random vectors distinct from current vector i
            # Create a list of candidate indices excluding the current index 'i'
            candidates = list(range(pop_size))
            candidates.remove(i)
            a_idx, b_idx, c_idx = random.sample(candidates, 3)
            
            x_a = population[a_idx]
            x_b = population[b_idx]
            x_c = population[c_idx]
            
            # Create mutant vector
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # Enforce bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector by mixing target and mutant
            # Generate mask for crossover
            cross_points = np.random.rand(dim) < crossover_rate
            
            # Ensure at least one parameter is changed from the mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection: Evaluate trial vector
            f_trial = func(trial_vector)

            # If trial is better, replace current individual
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_vector
                
                # Update global best found so far
                if f_trial < best:
                    best = f_trial

    return best
