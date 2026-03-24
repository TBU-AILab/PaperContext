import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Record start time
    start = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Initialize global best
    best = float('inf')
    
    # --- Algorithm Configuration (Differential Evolution) ---
    # Population size logic: 
    # Standard is 10*dim, but we cap it to ensure generations run within limited max_time.
    pop_size = max(10, 5 * dim)
    if pop_size > 50: 
        pop_size = 50
    
    F = 0.8  # Differential weight (Mutation factor)
    CR = 0.7 # Crossover probability
    
    # Pre-process bounds for numpy vectorization
    bounds_array = np.array(bounds)
    lower_b = bounds_array[:, 0]
    upper_b = bounds_array[:, 1]
    diff_b = upper_b - lower_b
    
    # Initialize Population
    # Shape: (pop_size, dim)
    population = lower_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # --- Initialization Phase ---
    # Evaluate the initial population
    for i in range(pop_size):
        # Strict time check
        if (datetime.now() - start) >= limit:
            return best
        
        val = func(population[i])
        fitness[i] = val
        if val < best:
            best = val

    # --- Evolution Phase ---
    while True:
        # Check time before starting a new generation
        if (datetime.now() - start) >= limit:
            return best
            
        for i in range(pop_size):
            # Check time before every individual evaluation to ensure strict adherence
            if (datetime.now() - start) >= limit:
                return best
            
            # 1. Mutation strategy: DE/rand/1/bin
            # Select 3 distinct random individuals excluding current index i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector
            mutant = a + F * (b - c)
            
            # 2. Crossover
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter comes from the mutant (standard DE requirement)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Boundary handling: Clip values to stay within bounds
            trial = np.clip(trial, lower_b, upper_b)
            
            # 4. Selection
            # Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better, replace current
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                # Update global best
                if f_trial < best:
                    best = f_trial
                    
    return best
