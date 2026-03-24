#Here is a fully operational, self-contained Python implementation using the **Differential Evolution (DE)** algorithm.
#
#DE is a robust evolutionary algorithm suitable for finding global minima in continuous search spaces. It generally performs significantly better than random search or simple gradient descent on black-box functions, especially when derivative information is unavailable.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution within a specified time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Algorithm Parameters ---
    # Population size: Adaptive to dimension, but capped to ensure speed if dim is high
    pop_size = max(10, 3 * dim)
    if pop_size > 50:
        pop_size = 50
        
    mutation_factor = 0.8    # F: Controls the amplification of the differential variation
    crossover_prob = 0.7     # CR: Probability of crossover
    
    # --- Initialization ---
    # Initialize best fitness found so far
    best = float('inf')
    
    # Initialize population with random values within bounds
    # pop shape: (pop_size, dim)
    pop = np.zeros((pop_size, dim))
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        # Check time constraint even during initialization
        if (datetime.now() - start_time) >= time_limit:
            return best
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Evolution Loop ---
    # Continue evolving until time runs out
    while True:
        # Check time at the start of each generation
        if (datetime.now() - start_time) >= time_limit:
            return best

        for i in range(pop_size):
            # Check time strictly before every potentially expensive function call
            if (datetime.now() - start_time) >= time_limit:
                return best

            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]
            
            # Create mutant vector: a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # 2. Crossover: Create trial vector by mixing target(i) and mutant
            # Ensure at least one parameter comes from mutant to introduce change
            cross_points = np.random.rand(dim) < crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, pop[i])
            
            # 3. Boundary Constraint Handling (Clip values to stay valid)
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection: Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better, replace current
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
