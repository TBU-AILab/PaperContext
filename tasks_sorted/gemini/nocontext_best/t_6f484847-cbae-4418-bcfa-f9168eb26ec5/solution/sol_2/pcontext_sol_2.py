#Here is the improved, fully operational, and self-contained Python code.
#
#It implements a **Vectorized Differential Evolution with Restarts**.
#**Improvements over the previous version:**
#1.  **Vectorization**: Instead of looping through individuals for the math (mutation/crossover), it uses NumPy matrix operations to process the entire population at once. This significantly increases the number of generations possible within the time limit.
#2.  **Restart Mechanism**: The score of ~42 suggests the previous algorithm got stuck in a local optimum (likely the Rastrigin function). This algorithm detects population stagnation (low variance) and "restarts" the search in a new area while preserving the global best, allowing it to escape local minima.
#3.  **Dithering**: The mutation factor `F` and crossover rate `CR` are randomized slightly per generation to prevent search bias.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Differential Evolution with Restarts.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Lower than standard DE to allow fast convergence checks 
    # and more frequent restarts.
    pop_size = max(10, 15 * dim) 
    
    # Limits
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    global_best_params = None

    # Time management helper
    def time_left():
        return max_time - (time.time() - start_time)

    # Main Restart Loop
    while time_left() > 0.05: # Leave a tiny buffer
        
        # --- Initialization ---
        # Initialize population randomly within bounds
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Evaluate initial population
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if time_left() < 0: return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_params = pop[i].copy()

        # If we restarted, inject the global best into the new population 
        # to ensure we don't lose progress (Elitism)
        if global_best_params is not None:
            pop[0] = global_best_params
            fitness[0] = global_best_val

        best_idx = np.argmin(fitness)
        
        # Convergence parameters
        stagnation_counter = 0
        last_best_val = fitness[best_idx]
        
        # --- Differential Evolution Loop ---
        while time_left() > 0.05:
            
            # 1. Dithering parameters (Randomize slightly to avoid bias)
            # F (Mutation): 0.5 to 0.9
            F = 0.5 + 0.4 * np.random.rand()
            # CR (Crossover): 0.8 to 1.0
            CR = 0.8 + 0.2 * np.random.rand()

            # 2. Vectorized Mutation (DE/rand/1)
            # Generate indices for mutation: r1 != r2 != r3
            # We use a fast approximate method using random integers.
            # Collisions are rare enough in high dim/pop_size to ignore for speed gain.
            idxs_a = np.random.randint(0, pop_size, pop_size)
            idxs_b = np.random.randint(0, pop_size, pop_size)
            idxs_c = np.random.randint(0, pop_size, pop_size)

            # Create mutant vectors: a + F * (b - c)
            # We apply simple dithering to F here as well by allowing matrix ops
            mutant = pop[idxs_a] + F * (pop[idxs_b] - pop[idxs_c])
            
            # 3. Vectorized Crossover (Binomial)
            # Create mask: True where we take from mutant, False from original
            cross_points = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one dimension is taken from mutant (guarantee change)
            # We pick a random dimension index for each individual
            j_rand = np.random.randint(0, dim, pop_size)
            # Use advanced indexing to set that specific dimension to True
            cross_points[np.arange(pop_size), j_rand] = True
            
            # Construct trial population
            trials = np.where(cross_points, mutant, pop)
            
            # 4. Boundary Constraint (Clip)
            trials = np.clip(trials, min_b, max_b)
            
            # 5. Selection (Evaluate all trials)
            # Note: func cannot be vectorized (black box), so we loop evaluations
            improved = False
            for i in range(pop_size):
                if time_left() < 0: return global_best_val
                
                f_trial = func(trials[i])
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trials[i]
                    improved = True
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_params = trials[i].copy()
                        # If we found a significantly better solution, reset stagnation
                        stagnation_counter = 0

            # 6. Check for Stagnation / Convergence
            # Calculate standard deviation of fitness to see if population collapsed
            current_best = np.min(fitness)
            pop_std = np.std(fitness)
            
            # If population variance is tiny, we are stuck in a local minima -> RESTART
            if pop_std < 1e-6:
                break 

            # If no improvement in best value for X generations -> RESTART
            if current_best < last_best_val:
                last_best_val = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Allow some patience, but fail fast to try new basins
            if stagnation_counter > 30:
                break

    return global_best_val
