#Here is a fully operational, self-contained Python implementation using **Differential Evolution**.
#
#Differential Evolution is a robust, global optimization algorithm that does not require gradients. It works well for non-differentiable, non-linear, and multimodal cost functions. This implementation includes a strict time-check mechanism to ensure the `max_time` constraint is respected, returning the best solution found up to that point.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a limited time.
    
    DE is a stochastic population-based method that is effective for global 
    optimization of real-valued functions.
    """
    
    # 1. Configuration Parameters
    # Dynamic population size: scales with dimension but capped to ensure speed
    pop_size = int(np.clip(dim * 10, 20, 100)) 
    mutation_factor = 0.8  # (F) Weighting factor for differential evolution [0, 2]
    crossover_prob = 0.7   # (CR) Crossover probability [0, 1]
    
    # 2. Time Management Setup
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # 3. Initialization
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = np.abs(max_b - min_b)

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    best_vector = None

    # We evaluate the initial population one by one to respect time limits immediately
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val
            best_vector = population[i].copy()

    # 4. Main Optimization Loop (Generational)
    while True:
        # Loop through the population
        for i in range(pop_size):
            # Strict time check before every new evaluation
            if (datetime.now() - start_time) >= time_limit:
                return best

            # --- Mutation ---
            # Select 3 distinct random indices excluding the current index 'i'
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]

            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Bound handling: clip mutant to stay within search space
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover ---
            # Create trial vector by mixing target(i) and mutant based on CR
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter comes from the mutant (to avoid duplicating 'i')
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])

            # --- Selection ---
            trial_fitness = func(trial)

            # Greedy selection: if trial is better, it replaces the current individual
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best if this is the new all-time minimum
                if trial_fitness < best:
                    best = trial_fitness
                    best_vector = trial.copy()

    return best
