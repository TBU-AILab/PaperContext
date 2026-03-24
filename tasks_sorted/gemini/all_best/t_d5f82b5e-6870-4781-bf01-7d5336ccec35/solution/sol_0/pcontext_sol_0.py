#Here is a fully operational, self-contained Python implementation using **Differential Evolution**.
#
#Differential Evolution (DE) is chosen here because it is a robust, stochastic, population-based optimization algorithm that works very well for continuous functions with multi-dimensional inputs without requiring gradient information. It is significantly more efficient than random search while remaining simple enough to implement without external optimization libraries.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the objective function using Differential Evolution.
    """
    
    # --- Configuration ---
    # F: Differential weight (0.5 to 1.0 usually). Controls amplification of differential variation.
    F = 0.8
    # CR: Crossover probability (0.0 to 1.0). Controls population diversity.
    CR = 0.9
    
    # Population size. 
    # A size of 10-15 * dim is standard, but we clamp it between 10 and 50 
    # to ensure the algorithm runs sufficient generations within limited time.
    pop_size = max(10, min(50, 15 * dim))

    # --- Initialization ---
    start_time = datetime.now()
    # We set a hard deadline slightly before max_time to ensure we return safely
    deadline = start_time + timedelta(seconds=max_time)

    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population: Shape (pop_size, dim)
    # Start with random values uniformly distributed within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Store fitness values. Initialize with infinity.
    fitness = np.full(pop_size, float('inf'))
    
    best_idx = -1
    best_val = float('inf')

    # Evaluate initial population
    for i in range(pop_size):
        # Check time budget before every expensive function call
        if datetime.now() >= deadline:
            return best_val if best_val != float('inf') else fitness[0]

        val = func(population[i])
        fitness[i] = val

        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Loop (Evolution) ---
    while True:
        # Iterate through every individual in the population
        for i in range(pop_size):
            
            # Check time constraints
            if datetime.now() >= deadline:
                return best_val

            # 1. Mutation: Target vector is population[i]
            # Select 3 distinct random individuals (a, b, c) different from i
            idxs = [idx for idx in range(pop_size) if idx != i]
            # We need 3 random indices. 
            a, b, c = np.random.choice(idxs, 3, replace=False)

            x_a = population[a]
            x_b = population[b]
            x_c = population[c]

            # Compute mutant vector: v = a + F * (b - c)
            mutant = x_a + F * (x_b - x_c)

            # 2. Crossover: Create trial vector u
            # Binomial crossover
            rand_vals = np.random.rand(dim)
            mask = rand_vals < CR
            
            # Ensure at least one parameter comes from mutant to avoid duplication
            # (Force one index to be True)
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, population[i])

            # 3. Boundary Constraints
            # Clip values to ensure they stay within bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection
            trial_fitness = func(trial)

            # Greedy selection: if trial is better or equal, replace target
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best found so far
                if trial_fitness < best_val:
                    best_val = trial_fitness

    return best_val
