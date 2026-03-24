#Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is well-suited for black-box global optimization problems over continuous domains and is robust enough to find good solutions within limited time frames.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the set of input parameter values that lead to the minimum output value
    within a limited time using Differential Evolution.
    """
    
    # 1. Setup Timing and Constants
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    # 2. Parse Bounds
    # Convert bounds to numpy arrays for efficient broadcasting
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff = max_b - min_b

    # 3. Algorithm Hyperparameters
    # Population size: Adjusted dynamically but clamped to ensure responsiveness
    # A size of 15-20 is often sufficient for DE to make progress quickly.
    pop_size = max(10, min(10 * dim, 50))
    mutation_factor = 0.8  # F: Weight for differential variation
    crossover_prob = 0.7   # CR: Probability of crossover
    
    # 4. Initialization
    population = np.zeros((pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    
    # Initialize population randomly and evaluate
    # We check time during initialization to handle very short max_time constraints
    for i in range(pop_size):
        if is_time_up() and i > 0: 
            # If time is up but we have at least one solution, return best
            return best_val
            
        # Generate random individual within bounds
        population[i] = min_b + np.random.rand(dim) * diff
        
        # Evaluate
        val = func(population[i])
        fitness[i] = val
        
        # Update best found so far
        if val < best_val:
            best_val = val

    # 5. Differential Evolution Loop
    while not is_time_up():
        for i in range(pop_size):
            if is_time_up():
                return best_val
            
            # --- Mutation Strategy (DE/rand/1/bin) ---
            
            # Select 3 distinct random indices distinct from current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # Enforce bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover ---
            # Create trial vector by mixing target and mutant
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter comes from the mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial_vector = np.where(cross_points, mutant, population[i])
            
            # --- Selection ---
            trial_fitness = func(trial_vector)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial_vector
                fitness[i] = trial_fitness
                
                # Update global best if this is a new record
                if trial_fitness < best_val:
                    best_val = trial_fitness

    return best_val
