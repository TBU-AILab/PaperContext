import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE).
    DE is a robust stochastic global optimization algorithm suitable for 
    black-box functions where gradients are unavailable.
    """
    
    start = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Initialize best found value
    best_fitness = float('inf')

    # --- Configuration ---
    # Population size (NP). 
    # A standard choice is 10*dim, but we cap it at 50 to ensure we get 
    # multiple generations within the time limit if the function is slow.
    pop_size = max(10, min(50, 10 * dim))
    
    # Differential Evolution parameters
    F = 0.8   # Differential weight (scaling factor) [0, 2]
    CR = 0.9  # Crossover probability [0, 1]

    # Pre-process bounds into numpy arrays for vectorization
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # --- Initialization ---
    # Create initial population uniformly distributed within bounds
    # Shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Store fitness for each individual
    fitness = np.full(pop_size, float('inf'))

    # Evaluate initial population
    for i in range(pop_size):
        # Time check before expensive function call
        if datetime.now() - start >= time_limit:
            return best_fitness
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Evolutionary Loop ---
    while True:
        for i in range(pop_size):
            # Strict time check inside the loop
            if datetime.now() - start >= time_limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random individuals a, b, c from population
            # excluding the current individual i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Bound constraints: Clip mutant to stay within search space
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Binomial crossover
            # Generate mask where crossover occurs
            cross_points = np.random.rand(dim) < CR
            
            # Guaranteed crossover: ensure at least one parameter comes from mutant
            # to avoid trying the exact same vector as pop[i] (unless CR=0)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Create trial vector
            trial = np.where(cross_points, mutant, pop[i])

            # 3. Selection
            f_trial = func(trial)

            # Greedy selection: if trial is better, replace parent
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
