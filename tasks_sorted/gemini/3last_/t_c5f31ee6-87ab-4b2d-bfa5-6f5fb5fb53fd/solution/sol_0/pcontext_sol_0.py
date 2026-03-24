#Here is a fully operational, self-contained implementation using the **Differential Evolution** algorithm. This algorithm is generally superior to random search for continuous optimization problems as it balances exploration (searching new areas) and exploitation (refining existing solutions) effectively.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm.
    
    Strategy: DE/rand/1/bin
    - Generates a population of candidate solutions.
    - Creates new candidates by combining existing ones (mutation & crossover).
    - Keeps the better solution (greedy selection).
    """
    start = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Hyperparameters ---
    # Population size: Standard is 10*dim, but we cap it to ensuring 
    # many generations can run within the time limit.
    pop_size = max(10, int(10 * dim)) 
    if pop_size > 50: 
        pop_size = 50
    
    mutation_factor = 0.8  # (F) Weight of difference vector
    crossover_prob = 0.9   # (CR) Probability of crossover

    # --- Initialization ---
    # Convert bounds to numpy for vector operations
    bounds_np = np.array(bounds)
    min_bound = bounds_np[:, 0]
    max_bound = bounds_np[:, 1]
    diff_bound = max_bound - min_bound

    # Initialize population within bounds
    # Shape: (pop_size, dim)
    population = min_bound + np.random.rand(pop_size, dim) * diff_bound
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')

    # Initial evaluation loop with time check
    for i in range(pop_size):
        if (datetime.now() - start) >= time_limit:
            # If time runs out during initialization, return best found so far
            return best_fitness if best_fitness != float('inf') else func(population[i])
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Evolution Loop ---
    # Pre-allocate indices list for mutation selection speed
    pop_indices = list(range(pop_size))

    while True:
        # Iterate through the population
        for i in range(pop_size):
            # Check time limit at individual level to ensure strict adherence
            if (datetime.now() - start) >= time_limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random vectors (a, b, c) excluding current (i)
            # Efficiently pick candidates
            candidates = pop_indices[:]
            candidates.pop(i) # Remove current index
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # 2. Crossover: Create trial vector u
            # Binomial crossover
            rand_vals = np.random.rand(dim)
            cross_points = rand_vals < crossover_prob
            
            # Ensure at least one parameter comes from mutant (avoid duplicating target)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Boundary Handling: Clip trial vector to bounds
            trial = np.clip(trial, min_bound, max_bound)

            # 4. Selection: Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better or equal, replace target
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best
                if f_trial < best_fitness:
                    best_fitness = f_trial

    # return fitness of the best found solution
    return best_fitness
