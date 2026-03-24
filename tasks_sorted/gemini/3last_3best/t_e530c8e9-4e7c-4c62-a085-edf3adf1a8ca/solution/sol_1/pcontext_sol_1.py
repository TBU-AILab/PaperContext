#The following is a fully operational, self-contained implementation of an improved optimization algorithm.
#
#**Algorithm Explanation:**
#This implementation uses an advanced variant of **Differential Evolution (DE)** with the **"current-to-best/1/bin"** strategy. 
#1.  **Strategy**: Unlike standard `rand/1` (which explores blindly), `current-to-best` directs candidates towards the best solution found so far while maintaining diversity using random difference vectors. This significantly improves convergence speed in time-constrained environments.
#2.  **Adaptive Parameters**: The mutation factor $F$ is "dithered" (randomized) for each candidate to prevent getting stuck in local optima.
#3.  **Restart Mechanism**: If the population converges early (variance becomes negligible), the algorithm triggers a "soft restart." It preserves the best solution found but re-initializes the rest of the population to explore new areas of the search space, maximizing the utility of the available time.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution with 'current-to-best' strategy
    and automatic restarts to handle stagnation.
    """
    start_time = time.time()
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population size: Adapted to dimension but bounded for performance
    # N = 15 * dim is generally robust, clipped to [20, 100] to manage runtime overhead
    pop_size = int(np.clip(dim * 15, 20, 100))
    
    # Initialize population randomly within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    global_best_val = float('inf')
    global_best_idx = -1
    
    # Evaluate initial population
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return global_best_val
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_idx = i

    # --- Hyperparameters ---
    # CR (Crossover Rate): 0.9 promotes convergence by preserving good structures
    # F (Mutation Factor): Randomized between 0.5 and 1.0 (dithering) per vector
    CR = 0.9 
    
    # --- Main Loop ---
    while True:
        # Check time
        if (time.time() - start_time) >= max_time:
            return global_best_val

        # Get the vector of the current best individual
        best_vec = pop[global_best_idx]
        
        # Iterate through population (Asynchronous update)
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            # Select r1, r2 distinct from i
            # Efficient selection logic
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
                
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
            
            # Current vector and random difference vectors
            x_i = pop[i]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Mutation Strategy: current-to-best/1
            # V = X_i + F * (X_best - X_i) + F * (X_r1 - X_r2)
            # This balances exploitation (move to best) and exploration (random difference)
            F = np.random.uniform(0.5, 1.0)
            mutant = x_i + F * (best_vec - x_i) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(dim) < CR
            # Ensure at least one parameter is updated
            cross_mask[np.random.randint(dim)] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # Boundary Constrain (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # Evaluation
            f_trial = func(trial)
            
            # Selection (Greedy)
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best immediately (Asynchronous)
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_idx = i
                    best_vec = trial

        # --- Restart Mechanism ---
        # If population diversity (fitness std dev) is too low, we are stuck.
        # Restart population but keep the best solution.
        if np.std(fitness) < 1e-6:
            # Only restart if we have enough time left (>10% of max_time)
            if (time.time() - start_time) < (max_time * 0.9):
                # Save best
                saved_best = pop[global_best_idx].copy()
                
                # Re-initialize population
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                pop[0] = saved_best # Preserve best at index 0
                
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = global_best_val
                global_best_idx = 0
                
                # Evaluate new random individuals (skip index 0)
                for k in range(1, pop_size):
                    if (time.time() - start_time) >= max_time:
                        return global_best_val
                    
                    val = func(pop[k])
                    fitness[k] = val
                    
                    if val < global_best_val:
                        global_best_val = val
                        global_best_idx = k
    
    return global_best_val
