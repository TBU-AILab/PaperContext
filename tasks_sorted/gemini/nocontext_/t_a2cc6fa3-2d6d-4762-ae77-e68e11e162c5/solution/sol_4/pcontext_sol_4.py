#Here is a fully operational, self-contained Python algorithm implementing **Differential Evolution**. This method is robust, requires no gradient information, handles continuous bounds well, and fits the specific constraints of the problem (numpy usage and time limitations).
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a time limit.
    """
    # 1. Setup timing
    start_time = time.time()
    # Use a small buffer (0.05s) to ensure we return before a strict timeout
    end_time = start_time + max_time - 0.05

    # 2. Hyperparameters for Differential Evolution
    # Population size: usually 10x dimension, but clamped to ensure speed in high dim
    pop_size = max(10, dim * 10)
    if pop_size > 50: pop_size = 50 # Cap population to ensure generations run quickly
    
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_prob = 0.7   # Crossover probability (CR)

    # 3. Parse bounds into numpy arrays for vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # 4. Initialize Population
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial fitness
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    best_idx = -1

    # We evaluate initial population cautiously regarding time
    for i in range(pop_size):
        if time.time() > end_time:
            # If we run out of time during init, return best found so far
            return best_val if best_val != float('inf') else 0.0
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # 5. Main Optimization Loop
    while True:
        # Iterate through the population
        for i in range(pop_size):
            # Time Check
            if time.time() > end_time:
                return best_val

            # --- Mutation ---
            # Select 3 random distinct indices (a, b, c) excluding current i
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)

            x_a = population[a]
            x_b = population[b]
            x_c = population[c]

            # Create mutant vector: v = a + F * (b - c)
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # Bound constraints: Clip values to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover ---
            # Create trial vector by mixing target(i) and mutant
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter comes from mutant to avoid stagnation
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            trial = np.where(cross_points, mutant, population[i])

            # --- Selection ---
            trial_fitness = func(trial)

            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best immediately
                if trial_fitness < best_val:
                    best_val = trial_fitness

    return best_val
