#Here is the improved, fully operational, and self-contained algorithm. 
#
#This implementation optimizes the **Differential Evolution (DE)** strategy by adding **Vectorized Operations** for faster generation of trial solutions and a **Restart Mechanism** with **Latin Hypercube Sampling (LHS)** initialization. This allows the algorithm to escape local minima and explore the search space more efficiently within the limited time.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function `func` using an optimized Differential Evolution 
    algorithm with restarts, Latin Hypercube initialization, and vectorized operations.
    """
    
    # Initialize timing
    start_time = time.time()
    # Define a safe time limit to return results (leave small buffer)
    limit_time = max_time - 0.05
    
    # Convert bounds to numpy array for efficient vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Population size: Adaptive to dimension but clamped to balance exploration/speed.
    # Larger populations explore better but reduce generation count.
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # Crossover Probability
    CR = 0.9  
    
    # Track the global best solution found across all restarts
    global_best_fit = float('inf')
    
    # --- Main Loop (Restarts) ---
    # We use restarts to escape local optima if convergence happens early.
    # If the function is simple, it converges fast and restarts to double-check.
    # If complex, it uses available time to find better basins of attraction.
    while True:
        # Check time before starting a new optimization cycle
        if time.time() - start_time > limit_time:
            return global_best_fit

        # --- Initialization with Latin Hypercube Sampling (LHS) ---
        # LHS ensures a stratified initial distribution, covering the space 
        # better than pure random uniform sampling.
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            # Distribute samples evenly across the dimension
            perm = np.random.permutation(pop_size)
            pop[:, d] = (perm + np.random.rand(pop_size)) / pop_size
        
        # Map 0-1 LHS samples to actual bounds
        pop = min_b + pop * diff_b
        
        # Evaluate initial population
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            # Strict time check inside evaluation loop
            if time.time() - start_time > limit_time:
                return global_best_fit
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_fit:
                global_best_fit = val
        
        # --- Evolution Loop ---
        while True:
            # Time check
            if time.time() - start_time > limit_time:
                return global_best_fit
            
            # Check for convergence to trigger restart
            # If the population fitness spread is negligible, we are stuck in a minimum
            if np.max(fitness) - np.min(fitness) < 1e-7:
                break

            # --- Vectorized Offspring Generation ---
            # Instead of looping to create mutants, we generate them in bulk using numpy.
            
            # Dither mutation factor F between 0.5 and 1.0 to handle different surface scales
            F = 0.5 + 0.5 * np.random.rand()
            
            # Select random indices for mutation (r1 != r2 != r3)
            # We generate a matrix of random values and sort them to get random permutations.
            # We select columns 1, 2, 3 to get distinct indices for every individual.
            idxs = np.argsort(np.random.rand(pop_size, pop_size), axis=1)
            r1, r2, r3 = idxs[:, 1], idxs[:, 2], idxs[:, 3]
            
            # Mutation (DE/rand/1): V = X_r1 + F * (X_r2 - X_r3)
            mutants = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Bound Constraint Handling (Clip to valid range)
            mutants = np.clip(mutants, min_b, max_b)
            
            # Crossover (Binomial)
            # Create a mask of random booleans based on CR
            cross_mask = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one variable is taken from the mutant (Standard DE rule)
            j_rand = np.random.randint(0, dim, size=pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            # Create trial population: Mix of Mutants and Parents
            trials = np.where(cross_mask, mutants, pop)
            
            # --- Selection ---
            # Evaluate trials and perform Greedy Selection
            for i in range(pop_size):
                if time.time() - start_time > limit_time:
                    return global_best_fit
                
                f_trial = func(trials[i])
                
                # If trial is better, update population
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trials[i]
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial
    
    return global_best_fit
