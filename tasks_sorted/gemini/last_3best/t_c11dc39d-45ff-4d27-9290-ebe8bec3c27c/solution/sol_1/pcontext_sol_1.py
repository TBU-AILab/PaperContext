#Here is a fully operational, self-contained Python solution. 
#
#It implements a **Vectorized Restarting Differential Evolution (DE)** algorithm. This improves upon standard DE by:
#1.  **Vectorizing** the mutation and crossover steps (using NumPy matrix operations instead of loops) to maximize the number of generations possible within the time limit.
#2.  **Implementing Restarts**: If the population converges (standard deviation becomes small) or stagnates, the algorithm saves the global best and restarts with a fresh population to explore new basins of attraction.
#3.  **Parameter Dithering**: Randomizes the Mutation ($F$) and Crossover ($CR$) factors slightly every generation to prevent the algorithm from getting stuck due to static hyperparameters.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Restarting Differential Evolution.
    """
    # Initialize timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size: Scale with dimension but cap to ensure speed
    # A larger population explores better, but updates slower.
    pop_size = max(20, int(dim * 5))
    pop_size = min(pop_size, 50)  # Cap at 50 to ensure high generation count
    
    # Restart tolerance: if population fitness std dev is below this, restart
    restart_tol = 1e-6 

    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Global best tracking
    global_best_val = float('inf')

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check time before starting a new population
        if datetime.now() >= end_time:
            return global_best_val

        # 1. Initialization
        # Generate random population within bounds
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))

        # Evaluate initial population
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val

        # 2. Evolution Loop
        while True:
            # Check time strictly
            if datetime.now() >= end_time:
                return global_best_val

            # Check for convergence (to trigger restart)
            if np.std(fitness) < restart_tol:
                break # Break inner loop -> triggers restart in outer loop

            # --- Vectorized Reproduction ---
            
            # Dynamic Parameters (Dithering)
            # Randomize F [0.5, 1.0] and CR [0.5, 1.0] to handle different landscapes
            F = 0.5 + 0.5 * np.random.rand()
            CR = 0.5 + 0.5 * np.random.rand()

            # Create indices for mutation
            # We need vectors a, b, c. Using random permutations is fast and sufficient.
            idxs = np.arange(pop_size)
            a_idx = np.random.permutation(idxs)
            b_idx = np.random.permutation(idxs)
            c_idx = np.random.permutation(idxs)

            # Mutation: rand/1/bin strategy -> V = a + F * (b - c)
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]
            mutant = a + F * (b - c)

            # Enforce bounds
            mutant = np.clip(mutant, min_b, max_b)

            # Crossover: Binomial
            # Create a boolean mask where True means "take from mutant"
            cross_mask = np.random.rand(pop_size, dim) < CR
            
            # Forced Gene: Ensure at least one dimension comes from the mutant
            # to avoid replicating the parent exactly.
            j_rand = np.random.randint(0, dim, size=pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True

            # Create trial population
            trial_pop = np.where(cross_mask, mutant, pop)

            # --- Selection (Sequential Evaluation) ---
            # We must evaluate sequentially as 'func' might not support vectorization
            for i in range(pop_size):
                if datetime.now() >= end_time:
                    return global_best_val
                
                # Evaluate candidate
                f_trial = func(trial_pop[i])

                # Greedy selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # Update global best
                    if f_trial < global_best_val:
                        global_best_val = f_trial

    return global_best_val
