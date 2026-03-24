#Here is a fully operational, self-contained Python algorithm using **Differential Evolution**. This method is generally superior to random search for continuous function minimization as it learns from the distribution of the current population to guide the search toward the global minimum.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a black-box function using Differential Evolution (DE/rand/1/bin).
    """
    # Initialize timing
    start_time = time.time()
    time_limit = start_time + max_time

    # --- Hyperparameters ---
    # Population size: usually 10x-15x dimension is a good rule of thumb. 
    # We clamp it to ensure it runs reasonably even in high dimensions within limited time.
    pop_size = max(10, 10 * dim) 
    
    # Differential weight (F): controls the amplification of differential variation (0, 2]
    F = 0.8
    # Crossover probability (CR): controls the fraction of parameter values copied from the mutant (0, 1]
    CR = 0.9

    # --- Setup Bounds ---
    # Convert bounds to numpy arrays for vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = np.fabs(max_b - min_b)

    # --- Initialization ---
    # Create initial population with random values within bounds
    # Shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Store fitness of population
    fitness = np.full(pop_size, float('inf'))
    
    # Track global best
    best_fitness = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Safety time check during initialization
        if time.time() >= time_limit:
            return best_fitness

        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Differential Evolution Loop ---
    while True:
        # Check overall time constraint before starting a new generation
        if time.time() >= time_limit:
            return best_fitness

        # Iterate through the population
        for i in range(pop_size):
            # Check time strictly inside the loop to ensure we respect max_time
            if time.time() >= time_limit:
                return best_fitness

            # 1. Mutation (Strategy: DE/rand/1)
            # Select 3 random distinct indices excluding current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            
            # Create mutant vector
            mutant = a + F * (b - c)
            
            # Clip mutant to ensure it stays within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover (Binomial)
            # Select which parameters to take from mutant vs target
            # Ensure at least one parameter is taken from mutant to ensure evolution
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, pop[i])

            # 3. Selection
            # Evaluate trial vector
            trial_fitness = func(trial_vector)

            # Greedy selection: if trial is better or equal, replace target
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                pop[i] = trial_vector
                
                # Update global best found so far
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
