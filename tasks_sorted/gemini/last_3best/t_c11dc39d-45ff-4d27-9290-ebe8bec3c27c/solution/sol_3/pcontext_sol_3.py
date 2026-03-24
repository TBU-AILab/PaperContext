#The algorithm below implements an **Enhanced Vectorized Differential Evolution (DE) with Per-Individual Dithering and Elitist Restarts**.
#
#### Improvements over previous versions:
#1.  **Vectorized Parameter Dithering**: Unlike Algorithm 1 (which used a single scalar $F$ and $CR$ for the whole population per generation), this algorithm generates unique mutation ($F$) and crossover ($CR$) parameters for **each individual** using Cauchy and Normal distributions respectively. This maintains diversity much better than global parameter dithering.
#    *   **$F$ (Mutation)**: Sampled from a **Cauchy distribution**. The heavy tails of the Cauchy distribution produce occasional large mutation factors, allowing individuals to jump out of local optima (Levy flight-like behavior), while the center (0.5) ensures stable convergence.
#    *   **$CR$ (Crossover)**: Sampled from a Normal distribution to provide variety in how much genetic information is exchanged.
#2.  **Elitist Restarts**: When the population stagnates (low standard deviation), the algorithm restarts. However, unlike a random restart, it **injects the global best solution** found so far into the new population. This transforms the restart into a "macro-mutation" event, preserving the best trait while destroying local convergence.
#3.  **Vectorized Operators**: All evolutionary operators (mutation, crossover, bounds checking) are batched using NumPy, minimizing interpreter overhead and maximizing the number of function evaluations within the time limit.
#4.  **Dynamic Population Sizing**: The population size is scaled based on dimension but clamped to a safe range `[40, 100]` to balance exploration capability with iteration speed.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Differential Evolution with 
    Per-Individual Dithering and Elitist Restarts.
    """
    # Initialize timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size: 
    # Scaled with dimension to ensure coverage, but clamped to ensure
    # enough generations can run within the time limit.
    pop_size = np.clip(int(dim * 10), 40, 100)
    
    # Restart tolerance: if std dev of fitness drops below this, restart.
    restart_tol = 1e-7

    # Pre-process bounds for efficient broadcasting
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Track global best across restarts
    global_best_val = float('inf')
    global_best_pos = None

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check time before starting a new population
        if datetime.now() >= end_time:
            return global_best_val

        # 1. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitist Restart: 
        # If we have a previous best, inject it into the new population (at index 0)
        # to ensure we don't lose progress and to guide the new random population.
        if global_best_pos is not None:
            pop[0] = global_best_pos.copy()

        fitness = np.full(pop_size, float('inf'))

        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() >= end_time: return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_pos = pop[i].copy()

        # 2. Evolutionary Cycle
        while True:
            # Strict Time Check
            if datetime.now() >= end_time: return global_best_val

            # Convergence Check for Restart
            # If population variance is too low, we are likely stuck in a local optimum.
            if np.std(fitness) < restart_tol:
                break # Break inner loop to trigger restart

            # --- Vectorized Parameter Generation (Dithering) ---
            
            # Generate F (Mutation Factor) per individual
            # Cauchy distribution (loc=0.5, scale=0.3). 
            # Heavy tails allow occasional large F (>1.0) to jump out of local basins.
            F = 0.5 + 0.3 * np.random.standard_cauchy(pop_size)
            F = np.where(F <= 0.1, 0.1, F) # Clamp lower bound (prevent stagnation)
            F = np.where(F >= 1.0, 1.0, F) # Clamp upper bound
            F = F[:, np.newaxis] # Reshape for broadcasting

            # Generate CR (Crossover Rate) per individual
            # Normal distribution (loc=0.6, scale=0.1).
            CR = np.random.normal(0.6, 0.1, pop_size)
            CR = np.clip(CR, 0.0, 1.0)
            CR = CR[:, np.newaxis]

            # --- Vectorized Mutation (DE/rand/1) ---
            # We need indices r1 != r2 != r3. 
            # Using 3 random permutations is a fast, vectorized approximation.
            idxs = np.arange(pop_size)
            r1 = np.random.permutation(idxs)
            r2 = np.random.permutation(idxs)
            r3 = np.random.permutation(idxs)

            # Calculate Mutant Vector: V = X_r1 + F * (X_r2 - X_r3)
            # F is broadcasted across dimensions.
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Bound Handling: Clip to constraints
            mutant = np.clip(mutant, min_b, max_b)

            # --- Vectorized Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < CR
            
            # Forced Gene: Ensure at least one dimension comes from the mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            # Create Trial Population
            trial_pop = np.where(cross_mask, mutant, pop)

            # --- Selection (Sequential Evaluation) ---
            # We must evaluate sequentially as 'func' takes a single array
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # Update Global Best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_pos = trial_pop[i].copy()

    return global_best_val
