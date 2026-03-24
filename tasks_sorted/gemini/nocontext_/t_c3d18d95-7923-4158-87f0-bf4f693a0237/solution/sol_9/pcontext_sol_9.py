import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using Differential Evolution (DE).
    
    Parameters:
    func : callable
        Objective function taking a 1D numpy array and returning a float.
    dim : int
        Dimension of the input vector.
    bounds : list
        List of (min, max) tuples for each dimension.
    max_time : int
        Maximum execution time in seconds.
        
    Returns:
    float
        The best (minimum) output value found.
    """
    
    # --- Configuration ---
    # Differential Evolution Hyperparameters
    # F: Mutation factor (typically 0.5-1.0). Controls amplification of differential variation.
    # CR: Crossover probability (0.0-1.0). Controls how many parameters are taken from the mutant.
    F = 0.8
    CR = 0.7
    
    # Population Strategy
    # Dynamically size population based on dimension, capped to ensure speed.
    # A multiplier of 10 is standard, but we use a smaller dynamic range 
    # to ensure convergence within time limits for expensive functions.
    pop_size = int(max(10, min(100, dim * 5)))
    
    # --- Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Prepare bounds for vectorized operations
    # Convert list of tuples to numpy arrays
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # Data structures
    population = np.zeros((pop_size, dim))
    fitnesses = np.full(pop_size, float('inf'))
    best = float('inf')

    # --- Phase 1: Initialize Population ---
    # We fill the population iteratively. If time runs out during initialization,
    # we return the best found so far.
    for i in range(pop_size):
        # Time Check
        if (datetime.now() - start_time) >= time_limit:
            return best

        # Generate random individual within bounds
        # x = min + random * (max - min)
        ind = min_b + np.random.rand(dim) * diff_b
        
        # Evaluate
        val = func(ind)
        
        # Store
        population[i] = ind
        fitnesses[i] = val
        
        # Update global best
        if val < best:
            best = val

    # --- Phase 2: Evolution Loop ---
    while True:
        # Check time at start of generation
        if (datetime.now() - start_time) >= time_limit:
            return best
            
        for j in range(pop_size):
            # Check time frequently (before every evaluation) to strictly adhere to max_time
            if (datetime.now() - start_time) >= time_limit:
                return best
            
            # --- Mutation ---
            # Select 3 distinct random indices distinct from current index j
            candidates = [idx for idx in range(pop_size) if idx != j]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # --- Crossover ---
            # Binomial crossover: swap parameters based on CR
            rand_vals = np.random.rand(dim)
            mask = rand_vals < CR
            
            # Ensure at least one parameter changes to ensure evolution
            if not np.any(mask):
                mask[np.random.randint(0, dim)] = True
                
            trial = np.where(mask, mutant, population[j])
            
            # --- Boundary Handling ---
            # Clip trial vector to ensure it stays within specified bounds
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            try:
                f_trial = func(trial)
            except Exception:
                # Fallback if func fails for specific parameters
                f_trial = float('inf')
            
            # Greedy selection: if trial is better or equal, replace target
            if f_trial <= fitnesses[j]:
                fitnesses[j] = f_trial
                population[j] = trial
                
                # Update global best found
                if f_trial < best:
                    best = f_trial

    return best
