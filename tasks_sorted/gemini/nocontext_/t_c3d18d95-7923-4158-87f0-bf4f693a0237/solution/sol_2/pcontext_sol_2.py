#Here is a fully operational, self-contained Python implementation using a **Differential Evolution** algorithm. 
#
#This algorithm is superior to a simple random search as it uses the information from the current population to guide the search towards the minimum, balancing exploration and exploitation. It is robust for black-box optimization problems and handles the time limit by checking the clock after every function evaluation.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function 'func' using Differential Evolution 
    within a specified time limit.
    """
    
    # --- Setup Time Management ---
    start_time = datetime.now()
    # Subtract a tiny buffer (0.05s) to ensure safe return before external timeout
    time_limit = timedelta(seconds=max_time - 0.05)

    # --- Hyperparameters ---
    # Population size (NP): 
    # A size of 10-20 is often sufficient for lower dimensions.
    # We cap it at 50 to ensure we complete generations even if 'func' is slow.
    pop_size = max(10, min(50, dim * 10))
    
    # Differential Weight (F): scaling factor for mutation [0, 2]
    mutation_factor = 0.6
    
    # Crossover Probability (CR): probability of swapping parameters [0, 1]
    crossover_prob = 0.8

    # --- Initialization ---
    # Convert list bounds to numpy arrays for efficient vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Initialize Population: Shape (pop_size, dim)
    # Randomly distributed within the bounds
    population = min_b + (max_b - min_b) * np.random.rand(pop_size, dim)
    
    # Store fitness of each individual
    pop_fitness = np.full(pop_size, float('inf'))
    
    # Global best tracker
    best_fitness = float('inf')

    # --- Initial Evaluation ---
    # We must evaluate the random population first.
    # We check time inside this loop in case 'func' is computationally expensive.
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_fitness
        
        # Evaluate
        val = func(population[i])
        pop_fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Optimization Loop (Differential Evolution) ---
    while True:
        # Loop over the population
        for i in range(pop_size):
            # 1. Time Check
            if datetime.now() - start_time >= time_limit:
                return best_fitness

            # 2. Mutation Strategy: DE/rand/1/bin
            # Select 3 distinct random individuals (a, b, c) different from current (i)
            idxs = [idx for idx in range(pop_size) if idx != i]
            # Fast random sampling
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Create mutant vector: V = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # 3. Constraints Handling
            # Clip the mutant vector to stay within defined bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 4. Crossover
            # Create trial vector by mixing target(i) and mutant based on CR
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is taken from mutant (prevents stagnation)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial_vector = np.where(cross_points, mutant, population[i])

            # 5. Selection
            # Evaluate the trial vector
            f_trial = func(trial_vector)

            # If trial is better or equal, it replaces the current individual
            if f_trial <= pop_fitness[i]:
                population[i] = trial_vector
                pop_fitness[i] = f_trial
                
                # Update global best if this is the new best found so far
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
