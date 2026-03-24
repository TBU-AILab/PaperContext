#The following algorithm improves upon the previous submission by implementing a **Vectorized Differential Evolution with Dithering and Restarts**.
#
#### Key Improvements:
#1.  **Vectorization**: Instead of looping through the population (slow in Python), operations are applied to the entire population array at once using NumPy. This significantly increases the number of generations processed within the `max_time`, leading to better convergence.
#2.  **Best/1/Bin Strategy**: Instead of mutating random individuals ("Rand/1/Bin"), this implementation mutates based on the current global best ("Best/1/Bin"). This is an aggressive strategy that converges faster.
#3.  **Dithering**: The mutation factor `F` is randomized slightly every generation. This prevents the population from getting stuck in geometric patterns and helps escape local optima.
#4.  **Restart Mechanism**: If the population converges (all individuals cluster around the same point) before `max_time` runs out, the algorithm effectively "restarts." It keeps the best solution found so far and re-initializes the rest of the population to explore new areas of the search space.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Differential Evolution with Dithering and Restarts.
    """
    # --- Setup and Constants ---
    start_time = time.time()
    
    # Convert bounds to numpy array
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Adaptive Population Size
    # Slightly larger population for exploration, clamped for performance
    pop_size = int(max(20, 15 * dim))
    if pop_size > 100: 
        pop_size = 100

    # DE Hyperparameters
    # CR (Crossover Probability): High CR is generally good for dependent variables
    CR = 0.9 
    # Convergence threshold (Standard deviation of fitness)
    tol = 1e-6 

    # --- Initialization ---
    # Global best tracking
    global_best_val = float('inf')
    global_best_vec = None
    
    # Initial Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return global_best_val
        val = func(pop[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val
            global_best_vec = pop[i].copy()

    # --- Main Loop ---
    while True:
        # Check time overhead
        if time.time() - start_time >= max_time:
            return global_best_val

        # --- Restart Mechanism ---
        # If population diversity is lost (converged), restart the population
        # but keep the best solution found so far.
        if np.std(fitness) < tol:
            # Re-initialize population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Inject the global best into the new population to ensure monotony
            pop[0] = global_best_vec
            
            # Re-evaluate
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return global_best_val
                val = func(pop[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = pop[i].copy()
            continue # Start new generation immediately

        # --- Dithering ---
        # Randomize F (Scaling Factor) between 0.5 and 0.9 each generation
        # This helps balance exploration (high F) and exploitation (low F).
        F = 0.5 + (0.4 * np.random.rand())

        # --- Vectorized Mutation (Best/1/Bin) ---
        # We need indices for r1 and r2 such that r1 != r2
        # Generate random indices
        idxs = np.random.randint(0, pop_size, size=(pop_size, 2))
        
        # Determine best index in current population
        best_idx_current = np.argmin(fitness)
        best_vec_current = pop[best_idx_current]

        # Vector math: V = Best + F * (Pop[r1] - Pop[r2])
        r1 = pop[idxs[:, 0]]
        r2 = pop[idxs[:, 1]]
        
        # Compute mutant vectors
        mutant = best_vec_current + F * (r1 - r2)
        
        # --- Boundary Handling ---
        # Clip to bounds
        mutant = np.clip(mutant, min_b, max_b)

        # --- Crossover (Binomial) ---
        # Create mask: True where we take from mutant, False where we keep target
        cross_points = np.random.rand(pop_size, dim) < CR
        
        # Ensure at least one dimension changes for every individual
        # (Random index for every row)
        j_rand = np.random.randint(0, dim, size=pop_size)
        cross_points[np.arange(pop_size), j_rand] = True
        
        # Create trial population
        trial_pop = np.where(cross_points, mutant, pop)

        # --- Selection ---
        # Evaluate trials
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return global_best_val
            
            f_trial = func(trial_pop[i])
            
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trial_pop[i].copy()

    return global_best_val
