#The following code implements an improved **Differential Evolution (DE)** algorithm. It builds upon the most successful previous approach (DE/rand/1/bin) by adding **vectorized operations** for speed, **parameter dithering** for robustness, and a **variance-based restart mechanism** to escape local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Differential Evolution (DE/rand/1/bin)
    with Parameter Dithering and Variance-Based Restarts.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Clamped to [20, 50] to balance exploration with generation speed.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # Parameters
    # CR: Crossover Rate. 0.8 allows substantial genetic mixing.
    # F: Mutation Factor. Will be dithered (randomized) per generation.
    CR = 0.8
    
    # Pre-process bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # Initialize population randomly within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    # Note: Using a list comprehension or loop is necessary as func expects 1D array
    fitness = np.zeros(pop_size)
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # Indices array for vectorization
    idxs = np.arange(pop_size)
    
    # --- Main Optimization Loop ---
    while True:
        # Check time budget at the start of generation
        if time.time() - start_time >= max_time:
            return best_val
            
        # --- Restart Mechanism ---
        # If the population variance is effectively zero, the algorithm has converged.
        # If the best found is a local optimum, we must restart to find the global one.
        if np.std(fitness) < 1e-7:
            # Re-initialize population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Elitism: Inject the best solution found so far
            pop[0] = best_vec
            
            # Re-evaluate (skipping index 0)
            fitness[0] = best_val
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            continue # Start fresh generation

        # --- Vectorized Mutation (DE/rand/1) ---
        # Select r1, r2, r3 distinct from i and each other
        # Using modular arithmetic with random offsets is faster than np.random.choice loop
        r1 = (idxs + np.random.randint(1, pop_size, pop_size)) % pop_size
        r2 = (r1 + np.random.randint(1, pop_size, pop_size)) % pop_size
        r3 = (r2 + np.random.randint(1, pop_size, pop_size)) % pop_size
        
        # Dither F: Randomize scale factor [0.5, 0.9] per generation
        # This prevents the algorithm from getting stuck in search patterns
        F = 0.5 + 0.4 * np.random.rand()
        
        # Compute Mutant Vectors: V = X_r1 + F * (X_r2 - X_r3)
        # Vectorized operation for the whole population
        mutant = pop[r1] + F * (pop[r2] - pop[r3])
        
        # --- Crossover (Binomial) ---
        # Create mask: True if we take from mutant, False if from target
        mask = np.random.rand(pop_size, dim) < CR
        
        # Force at least one dimension to come from mutant (Standard DE rule)
        j_rand = np.random.randint(0, dim, pop_size)
        mask[idxs, j_rand] = True
        
        # Create trial population
        trial_pop = np.where(mask, mutant, pop)
        
        # --- Boundary Handling ---
        # Clip values to stay within valid bounds
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # --- Selection ---
        # Evaluate trial vectors and replace target if trial is better or equal
        # We must loop for evaluation as func is black-box
        for i in range(pop_size):
            # Strict time check inside evaluation loop
            if time.time() - start_time >= max_time:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()
                    
    return best_val
