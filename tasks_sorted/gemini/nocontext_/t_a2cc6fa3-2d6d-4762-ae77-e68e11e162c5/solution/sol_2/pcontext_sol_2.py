#Here is the implementation of a Differential Evolution algorithm. This approach is generally more efficient and robust than random search for continuous optimization problems, as it uses information from the current population to guide the search toward better solutions.
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
    
    # --- DE Hyperparameters ---
    # Population size: usually 5x to 10x the dimension is a good rule of thumb.
    # We use a minimum of 10 to ensure enough genetic diversity.
    pop_size = max(10, 5 * dim)
    
    # F: Differential weight [0, 2], usually ~0.8
    # CR: Crossover probability [0, 1], usually ~0.9
    F = 0.8
    CR = 0.9

    # --- Initialization ---
    # Create the initial population within bounds
    # bounds is a list of tuples: [(min, max), (min, max), ...]
    pop = np.zeros((pop_size, dim))
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # Generate random population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    best_idx = -1

    # Evaluate initial population
    # We check time inside this loop in case function evaluation is very slow
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best:
            best = val
            best_idx = i

    # --- Main Optimization Loop ---
    while True:
        # Loop through the population
        for i in range(pop_size):
            # Check time constraint strictly before every evaluation
            if datetime.now() - start_time >= time_limit:
                return best

            # 1. Mutation
            # Select 3 distinct random indices (a, b, c) different from current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            
            # Create mutant vector
            mutant = a + F * (b - c)
            
            # 2. Crossover
            # Generate random probabilities for crossover
            rand_vals = np.random.rand(dim)
            # Ensure at least one parameter changes (to avoid duplicating parent)
            j_rand = np.random.randint(dim)
            
            # Create trial vector (mask: True if we take from mutant)
            mask = (rand_vals < CR)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, pop[i])
            
            # 3. Constrain boundaries
            # Clip the trial vector to ensure it stays within bounds
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Trial is better or equal, replace existing individual
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best if necessary
                if f_trial < best:
                    best = f_trial

    return best
