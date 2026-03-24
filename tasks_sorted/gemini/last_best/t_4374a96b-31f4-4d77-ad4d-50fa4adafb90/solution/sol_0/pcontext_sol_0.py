#Here is the fully operational and self-contained algorithm using Differential Evolution, a robust global optimization method suitable for various function landscapes.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function `func` within `bounds` and `max_time` 
    using a Differential Evolution algorithm.
    """
    
    # Record start time to manage the time budget
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Determines diversity.
    # We use a dynamic size based on dimension, clamped between 20 and 100 
    # to ensure the algorithm can perform sufficient generations within limited time.
    pop_size = min(max(20, dim * 5), 100)
    
    # Differential Evolution Parameters
    F = 0.8      # Mutation factor (scale of difference vectors)
    CR = 0.9     # Crossover probability
    
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # Initialize population randomly within the specified bounds
    # Shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Array to store fitness (output value) of each individual
    fitness = np.full(pop_size, float('inf'))
    
    # Variable to track the best solution found so far
    best = float('inf')
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Check time limit before potentially expensive function call
        if time.time() - start_time >= max_time:
            return best
        
        # Calculate fitness
        val = func(pop[i])
        fitness[i] = val
        
        # Update global best
        if val < best:
            best = val

    # --- Main Evolution Loop ---
    while True:
        # Check time at the start of the generation
        if time.time() - start_time >= max_time:
            return best
        
        for i in range(pop_size):
            # Check time strictly before every evaluation
            if time.time() - start_time >= max_time:
                return best
            
            # --- Mutation (DE/rand/1) ---
            # Select 3 distinct random individuals excluding the current one (i)
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]
            
            # Create mutant vector
            mutant = a + F * (b - c)
            
            # Bound handling: Clip mutant vector to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover (Binomial) ---
            # Generate random mask based on Crossover Probability (CR)
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter comes from the mutant (standard DE rule)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            # Create trial vector: mix mutant and target
            trial = np.where(cross_points, mutant, pop[i])
            
            # --- Selection ---
            # Evaluate the trial vector
            f_trial = func(trial)
            
            # If trial is better than current target, replace it
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best if new minimum found
                if f_trial < best:
                    best = f_trial

    return best
