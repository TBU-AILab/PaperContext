#Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is generally much more efficient than random search for finding global minima, as it balances exploring the search space and exploiting promising regions.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm
    within a specified time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Algorithm Configuration ---
    # Population size (NP): Standard heuristic is 10 * dim.
    # We set a minimum to ensure diversity.
    pop_size = max(20, 10 * dim)
    
    # DE Hyperparameters:
    # F (Differential Weight): typically 0.5 to 1.0. Controls exploration.
    # CR (Crossover Probability): typically 0.7 to 0.9. Controls diversity.
    F = 0.8
    CR = 0.9

    # Convert bounds to numpy arrays for efficient vectorized clipping
    # bounds is a list of [(min, max), ...]
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # --- Initialization ---
    best = float('inf')
    
    # Initialize population with random values uniformly distributed within bounds
    # pop shape: (pop_size, dim)
    pop = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(pop_size, dim))
    
    # Array to store fitness values of the population
    fitness = np.full(pop_size, float('inf'))

    # Evaluate initial population
    # We iterate explicitly to check time limits even during initialization
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Optimization Loop (DE/rand/1/bin) ---
    while True:
        for i in range(pop_size):
            # Check time limit before every potentially expensive function call
            if datetime.now() - start_time >= time_limit:
                return best

            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != current (i)
            # Create a list of available indices excluding the current one
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Ensure the mutant vector stays within the search space bounds
            mutant = np.clip(mutant, lower_bounds, upper_bounds)

            # 2. Crossover: Create trial vector
            # Generate mask for crossover
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter changes (standard DE requirement)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector: take from mutant if cross_points is True, else from original
            trial = np.where(cross_points, mutant, pop[i])

            # 3. Selection
            f_trial = func(trial)

            # Greedy selection: if trial is better or equal, replace the individual
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best found so far
                if f_trial < best:
                    best = f_trial

    # Fallback return (technically unreachable due to while True loop)
    return best
