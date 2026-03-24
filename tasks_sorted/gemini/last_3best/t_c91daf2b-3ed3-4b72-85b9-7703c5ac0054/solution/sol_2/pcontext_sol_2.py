# The following algorithm improves upon the previous submissions by implementing a **Vectorized Differential Evolution (DE)** algorithm with a **Rand/1/Bin** strategy, **Parameter Dithering**, and a **Restart Mechanism**.
#
# #### Key Improvements:
# 1.  **Strategy Shift (Rand/1/Bin)**: While the previous "Best/1/Bin" strategy is fast, it is prone to getting stuck in local optima (evidenced by the worse score of 69.6). This implementation uses "Rand/1/Bin", which preserves population diversity better, allowing it to find deeper global minima (aiming to beat the score of 11.2).
# 2.  **Vectorization**: The algorithm leverages NumPy to perform mutation and crossover operations on the entire population simultaneously. This drastically reduces Python loop overhead, allowing for many more generations within the `max_time`.
# 3.  **Correct Vectorized Indexing**: A robust method (using `argsort` on a random matrix with infinite diagonal) is used to ensure that the distinct indices $r_1, r_2, r_3 \neq i$ are selected for mutation, maintaining mathematical correctness and population diversity.
# 4.  **Restart Mechanism**: If the population converges (standard deviation of fitness drops near zero), the algorithm saves the best solution and completely re-initializes the rest of the population. This prevents wasting time on a stagnated search.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Differential Evolution with Rand/1/Bin,
    Dithering, and Restarts.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population Size:
    # A larger population helps exploration but slows down individual generations.
    # We use a dynamic size based on dimension, clamped to [50, 100] to ensure 
    # vectorization efficiency without becoming too sluggish.
    pop_size = int(np.clip(20 * dim, 50, 100))
    
    # Crossover Probability (CR):
    # A high CR (0.9) is generally robust for optimization problems where variables
    # might be dependent (rotationally invariant).
    CR = 0.9
    
    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population randomly within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Storage for fitness values
    fitness = np.full(pop_size, float('inf'))
    
    # Track global best
    best_val = float('inf')
    best_vec = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Check overall time budget
        if time.time() - start_time >= max_time:
            return best_val
            
        # --- Restart Mechanism ---
        # If the population has converged (low variance in fitness), it's stuck.
        # We restart the population to explore new areas, but keep the elite (best) solution.
        if np.std(fitness) < 1e-6:
            # Re-initialize entire population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Restore the best individual found so far (Elitism)
            pop[0] = best_vec
            
            # Reset fitness array
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Re-evaluate the new population (skipping index 0)
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            continue # Skip to next iteration to process the new population

        # --- Vectorized DE Operations ---
        
        # 1. Index Selection
        # We need r1, r2, r3 such that r1 != r2 != r3 != i for every row i.
        # Method: Generate a (Pop x Pop) random matrix, set diagonal to Infinity, then Argsort.
        # The infinite diagonal ensures index 'i' is sorted to the end and not selected.
        rand_mat = np.random.rand(pop_size, pop_size)
        np.fill_diagonal(rand_mat, np.inf)
        
        # Taking the first 3 indices from the sorted order guarantees uniqueness and exclusion of i
        idxs = np.argsort(rand_mat, axis=1)
        r1 = idxs[:, 0]
        r2 = idxs[:, 1]
        r3 = idxs[:, 2]
        
        # 2. Mutation (Rand/1/Bin)
        # V = X[r1] + F * (X[r2] - X[r3])
        # Dithering: F is randomized between 0.5 and 1.0 each generation to vary step sizes.
        F = 0.5 + 0.5 * np.random.rand()
        
        mutant = pop[r1] + F * (pop[r2] - pop[r3])
        
        # Boundary Handling: Clip to bounds
        mutant = np.clip(mutant, min_b, max_b)
        
        # 3. Crossover (Binomial)
        # Create a mask where we pick from the mutant vector
        rand_cross = np.random.rand(pop_size, dim)
        mask = rand_cross < CR
        
        # Ensure at least one parameter is taken from the mutant (Standard DE rule)
        j_rand = np.random.randint(0, dim, size=pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        # Construct the trial population
        trial_pop = np.where(mask, mutant, pop)
        
        # 4. Selection
        # Evaluate trial vectors and replace parents if trial is better or equal
        for i in range(pop_size):
            # Strict time check
            if time.time() - start_time >= max_time:
                return best_val
                
            val = func(trial_pop[i])
            
            if val <= fitness[i]:
                fitness[i] = val
                pop[i] = trial_pop[i]
                
                if val < best_val:
                    best_val = val
                    best_vec = trial_pop[i].copy()
                    
    return best_val