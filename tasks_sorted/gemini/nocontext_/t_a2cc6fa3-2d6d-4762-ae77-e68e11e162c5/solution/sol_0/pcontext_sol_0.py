#Here is the implementation of a Differential Evolution algorithm. This algorithm is robust for continuous global optimization problems and works well within time constraints by iteratively improving a population of candidate solutions.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Differential Evolution (DE) algorithm to minimize a function within a time limit.
    Strategy: DE/rand/1/bin
    """
    # 1. Setup Time Management
    start_time = datetime.now()
    # Subtract a small buffer to ensure we return before the external timeout
    time_limit = timedelta(seconds=max_time) 
    
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    # 2. Parse Bounds
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff = np.fabs(max_b - min_b)

    # 3. Algorithm Hyperparameters
    # Population size: Adaptive based on dimension, but capped for speed
    # A size of 10-20 is often sufficient for DE to make progress quickly
    pop_size = int(max(10, min(15 * dim, 50))) 
    mutation_factor = 0.8   # F in literature (typically 0.5-0.9)
    crossover_rate = 0.7    # CR in literature (typically 0.7-0.9)

    # 4. Initialize Population
    # Generate random particles within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_idx = -1

    # 5. Initial Evaluation
    # We must evaluate the initial population to establish a baseline
    for i in range(pop_size):
        if is_time_up() and best_val != float('inf'):
            return best_val
        
        # Ensure values are within bounds before calling func
        current_vector = np.clip(population[i], min_b, max_b)
        
        try:
            val = func(current_vector)
        except Exception:
            val = float('inf')

        fitness[i] = val

        if val < best_val:
            best_val = val
            best_idx = i

    # If time ran out during initialization and we have at least one valid result
    if is_time_up():
        return best_val

    # 6. Main Evolution Loop
    while not is_time_up():
        for i in range(pop_size):
            # Check time strictly inside the loop
            if is_time_up():
                return best_val

            # Mutation: DE/rand/1
            # Select 3 random indices distinct from current i
            candidates = list(range(pop_size))
            candidates.remove(i)
            idxs = np.random.choice(candidates, 3, replace=False)
            
            a = population[idxs[0]]
            b = population[idxs[1]]
            c = population[idxs[2]]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # Crossover: Binomial
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < crossover_rate
            # Ensure at least one parameter is changed from the mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            # Create trial vector
            trial = np.where(cross_points, mutant, population[i])

            # Boundary Constraint Handling: Clip to bounds
            trial = np.clip(trial, min_b, max_b)

            # Selection: Greedy
            try:
                f_trial = func(trial)
            except Exception:
                f_trial = float('inf')

            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best found so far
                if f_trial < best_val:
                    best_val = f_trial

    return best_val
