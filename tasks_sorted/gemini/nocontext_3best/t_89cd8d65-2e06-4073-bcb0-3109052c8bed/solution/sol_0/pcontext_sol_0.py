#Here is a fully operational implementation using the **Differential Evolution (DE)** algorithm.
#
#DE is a stochastic, population-based optimization algorithm that is generally much more effective than random search for continuous input spaces. It is robust, requires few control parameters, and converges well on non-differentiable or multi-modal functions.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Algorithm Parameters ---
    # Population size: Scale with dimension, but cap at 50 to ensure 
    # reasonable iteration count within limited time.
    pop_size = min(max(10, 10 * dim), 50)
    
    # Differential weight (0.5 to 1.0 is standard)
    F = 0.8 
    # Crossover probability (0.5 to 1.0 is standard)
    CR = 0.9 

    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness tracking
    best_fitness = float('inf')
    fitnesses = np.full(pop_size, float('inf'))

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_fitness
            
        val = func(population[i])
        fitnesses[i] = val
        if val < best_fitness:
            best_fitness = val

    # --- Main Loop (Evolution) ---
    while True:
        for i in range(pop_size):
            # Check time budget strictly before every potentially expensive evaluation
            if datetime.now() - start_time >= time_limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random agents (a, b, c) distinct from i
            idxs = [idx for idx in range(pop_size) if idx != i]
            # Select 3 random indices
            choices = np.random.choice(idxs, 3, replace=False)
            a, b, c = population[choices[0]], population[choices[1]], population[choices[2]]

            # Create mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Constrain mutant to bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector u from x (current) and v (mutant)
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter is taken from the mutant 
            # to avoid duplicating the parent
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection: Greedy approach
            try:
                trial_fitness = func(trial_vector)
            except Exception:
                # Handle potential errors in black-box func gracefully
                trial_fitness = float('inf')

            # If trial is better or equal, replace the individual
            if trial_fitness <= fitnesses[i]:
                fitnesses[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best found so far
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
