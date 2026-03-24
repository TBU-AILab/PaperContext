#Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is a robust stochastic method for global optimization of continuous, black-box functions, making it significantly more efficient than random search for finding minimum values within a constrained time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Parameters
    # F: Mutation factor (typically 0.5 to 1.0)
    # CR: Crossover probability (typically 0.5 to 1.0)
    F = 0.8
    CR = 0.9
    
    # Population size. 
    # We use a dynamic size based on dimension but capped to ensure 
    # the algorithm can perform generations even with short time limits or slow functions.
    pop_size = max(10, min(100, 10 * dim))

    # Pre-process bounds into numpy arrays for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b

    # Initialize Population: Randomly distributed within bounds
    # pop shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best = float('inf')

    # Initial Evaluation
    for i in range(pop_size):
        if (datetime.now() - start) >= time_limit:
            return best if best != float('inf') else fitness[0]
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # Main Evolutionary Loop
    while True:
        for i in range(pop_size):
            # Strict time check before every new evaluation
            if (datetime.now() - start) >= time_limit:
                return best

            # Mutation Strategy: DE/rand/1
            # Select 3 distinct random indices distinct from current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]

            # Create mutant vector
            mutant = a + F * (b - c)

            # Enforce bounds (Clamping)
            mutant = np.clip(mutant, min_b, max_b)

            # Crossover (Binomial)
            # Create a trial vector by mixing target vector and mutant
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter is taken from mutant to avoid duplication
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, pop[i])

            # Selection
            # Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better or equal, it replaces the target
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best
                if f_trial < best:
                    best = f_trial

    return best
