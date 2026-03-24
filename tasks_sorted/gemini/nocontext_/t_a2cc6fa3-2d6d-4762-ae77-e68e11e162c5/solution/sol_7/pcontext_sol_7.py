#Here is a fully operational, self-contained Python algorithm using **Differential Evolution**.
#
#This algorithm is chosen because it is robust for continuous, multi-dimensional optimization problems where the gradient is unknown (black-box), and it generally finds better global minima than simple random search within a limited time frame.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution algorithm 
    within a specified time limit.
    """
    start = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # ---------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------
    # Population size: balanced between diversity and iteration speed.
    pop_size = max(10, int(10 * dim)) 
    
    # Mutation factor (F) and Crossover probability (CR)
    F = 0.8
    CR = 0.7

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population matrix: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Store fitness values
    fitness = np.full(pop_size, float('inf'))
    
    # Track global best
    best_fitness = float('inf')
    best_idx = -1

    # ---------------------------------------------------------
    # Initial Evaluation Loop (Safe against very short max_time)
    # ---------------------------------------------------------
    for i in range(pop_size):
        # Time Check
        if (datetime.now() - start) >= limit:
            return best_fitness

        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # ---------------------------------------------------------
    # Main Differential Evolution Loop
    # ---------------------------------------------------------
    while True:
        # Loop through the population
        for i in range(pop_size):
            # Time Check (performed inside the loop for granularity)
            if (datetime.now() - start) >= limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random agents a, b, c != i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Clip mutant to ensure it stays within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector
            # Select random indices for crossover based on CR
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one dimension is changed (guarantees evolution)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            f_trial = func(trial)

            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best immediately if improved
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
