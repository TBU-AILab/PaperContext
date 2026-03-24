#The following algorithm implements a **Vectorized Restarting Differential Evolution (DE)** with **Latin Hypercube Sampling (LHS)** initialization and **Reflective Boundary Handling**.
#
#**Key Improvements over previous attempts:**
#1.  **Reflective Bounds**: Instead of simply clipping values that exceed bounds (which clumps population at the edges), this algorithm uses reflection (`2*bound - value`). This preserves the statistical distribution of the population and helps find minima located near the boundaries more effectively.
#2.  **Latin Hypercube Initialization**: Replaces random initialization with LHS to ensure the initial population covers the search space maximally with minimal variance, providing a better starting point for the first restart.
#3.  **Per-Individual Dithering**: Scaling factor ($F$) and Crossover Rate ($CR$) are randomized *per individual* in every generation (Uniform distribution). This ensures a continuous mix of exploration ($high F$) and exploitation ($low F$) within the same population.
#4.  **Robust Strategy (`rand/1`)**: Reverts to the robust `DE/rand/1/bin` strategy (used in the best-performing algorithm so far) rather than `current-to-best` (which caused premature convergence in the last attempt), but enhances it with the improvements above.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Restarting Differential Evolution
    with Latin Hypercube Sampling and Reflection Boundary Handling.
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size: 
    # A slightly larger population than minimal improves search in complex landscapes.
    # 15 * dim ensures adequate coverage per dimension, clamped to safe limits.
    pop_size = int(np.clip(dim * 15, 30, 80))
    
    # Restart tolerance: if population fitness std deviation < this, restart
    restart_tol = 1e-6
    
    # Pre-process bounds for efficient vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    global_best_val = float('inf')

    # --- Main Loop (Restarts) ---
    while True:
        # Strict time check
        if datetime.now() >= end_time:
            return global_best_val

        # 1. Initialization: Latin Hypercube Sampling (LHS)
        # LHS provides better initial coverage than pure random sampling by stratifying the space.
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            # Create edges for stratified bins
            edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
            # Sample uniformly within each bin
            samples = edges[:-1] + np.random.rand(pop_size) * (edges[1] - edges[0])
            # Shuffle samples to remove correlation between dimensions
            pop[:, d] = np.random.permutation(samples)

        # Evaluate Initial Population
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if datetime.now() >= end_time: return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val

        # 2. Evolution Loop
        while True:
            if datetime.now() >= end_time: return global_best_val
            
            # Check for convergence to trigger restart
            if np.std(fitness) < restart_tol:
                break

            # --- Parameters (Per-Individual Dithering) ---
            # Randomize control parameters for each vector to maintain diversity.
            # F ~ U(0.4, 0.9): Covers wide range for exploration/exploitation.
            # CR ~ U(0.5, 1.0): Generally high CR is preferred for DE.
            F = 0.4 + 0.5 * np.random.rand(pop_size, 1)
            CR = 0.5 + 0.5 * np.random.rand(pop_size, 1)

            # --- Mutation: DE/rand/1 ---
            # Generate random indices efficiently
            idxs = np.arange(pop_size)
            r1 = np.random.permutation(idxs)
            r2 = np.random.permutation(idxs)
            r3 = np.random.permutation(idxs)
            
            # Vectorized mutation: v = x_r1 + F * (x_r2 - x_r3)
            # Uses efficient numpy broadcasting.
            mutant = pop[r1] + F * (pop[r2] - pop[r3])

            # --- Bound Handling: Reflection ---
            # Instead of clipping (which piles mass on bounds), reflect violations.
            # If x < min, x = min + (min - x) = 2*min - x
            mutant = np.where(mutant < min_b, 2 * min_b - mutant, mutant)
            mutant = np.where(mutant > max_b, 2 * max_b - mutant, mutant)
            
            # Final safety clip to handle extreme reflections (e.g. if reflection is still out)
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover: Binomial ---
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < CR
            
            # Forced Gene: Ensure at least one parameter comes from mutant (standard DE rule)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[idxs, j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, pop)

            # --- Selection (Greedy) ---
            # Evaluate trial vectors sequentially as func may not support vectorization
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        
    return global_best_val
