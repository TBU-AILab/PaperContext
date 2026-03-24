#The following Python code implements a **Hybrid Vectorized jDE (Self-Adaptive Differential Evolution)**.
#
#### Improvements Explained:
#1.  **Hybrid Mutation Strategy**: This algorithm simultaneously utilizes two distinct mutation strategies within the population:
#    *   **Exploration (`rand/1`)**: Standard DE mutation used by ~50% of the population to maintain genetic diversity and explore the search space globally.
#    *   **Exploitation (`current-to-best/1`)**: A greedy strategy used by the other ~50% that pulls individuals towards the global best solution found so far.
#    *   By vectorizing both and applying a mask, the algorithm balances global search and local convergence dynamically without the computational cost of sorting the population (like in `current-to-pbest`).
#2.  **Vectorized jDE**: The control parameters ($F$ and $CR$) are self-adaptive. Each individual learns which parameter values result in successful offspring. This is fully vectorized using NumPy for maximum performance in Python.
#3.  **Elitist Restarts**: To solve multimodal problems where the population might converge to a local minimum, the algorithm detects stagnation (low variance). It restarts the population but **injects the global best solution** into the new generation, ensuring monotonic improvement and efficient use of the full `max_time`.
#4.  **Boundary Reflection**: Instead of clipping (which accumulates population on the edges), out-of-bound variables are reflected back into the search space, preserving the statistical distribution of the trial vectors.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Hybrid Vectorized jDE (Self-Adaptive Differential Evolution)
    with Restarts.
    
    Strategies:
    - 50% population: rand/1/bin (Exploration)
    - 50% population: current-to-best/1/bin (Exploitation)
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 
    # Needs to be large enough for diversity but small enough for speed.
    # 15 * dim is a standard heuristic, capped at 80 to ensure many generations.
    pop_size = int(max(20, 15 * dim))
    if pop_size > 80:
        pop_size = 80

    # jDE Constants (Probabilities to update control parameters)
    tau_F = 0.1
    tau_CR = 0.1
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracker
    global_best_val = float('inf')
    global_best_x = None

    # --- Restart Loop ---
    while True:
        # Check time before starting a new run
        if (datetime.now() - start_time) >= time_limit:
            return global_best_val
        
        # 1. Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: If we have a previous best, inject it into the new population
        # This turns the restart into a "Local Search" around the best found so far
        # or simply preserves progress.
        if global_best_x is not None:
            population[0] = global_best_x.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # 2. Evaluate Initial Population
        for i in range(pop_size):
            if i % 10 == 0 and (datetime.now() - start_time) >= time_limit:
                return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_x = population[i].copy()
        
        # Initialize Control Parameters (F and CR)
        # F: Mutation factor [0.1, 1.0]
        # CR: Crossover probability [0.0, 1.0]
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # --- Evolution Loop ---
        while True:
            # Check time
            if (datetime.now() - start_time) >= time_limit:
                return global_best_val
            
            # Check Stagnation
            # If population fitness variance is negligible, restart to escape local optima.
            if np.std(fitness) < 1e-6 or (np.max(fitness) - np.min(fitness)) < 1e-6:
                break
                
            # --- Parameter Adaptation (jDE) ---
            # Create trial F and CR
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            F_trial = F.copy()
            CR_trial = CR.copy()
            
            # F -> 0.1 + 0.9 * rand
            F_trial[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
            # CR -> rand
            CR_trial[mask_CR] = np.random.rand(np.sum(mask_CR))
            
            # --- Hybrid Mutation ---
            # Strategy A: rand/1 (Exploration)
            # Strategy B: current-to-best/1 (Exploitation)
            # We randomly assign strategies to individuals this generation
            strategy_mask = np.random.rand(pop_size) < 0.5  # True = current-to-best
            
            # Select indices
            # r1 != r2 != r3
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            r3 = np.random.randint(0, pop_size, pop_size)
            
            # Current Best Index
            best_idx = np.argmin(fitness)
            x_best = population[best_idx]
            
            # 1. Compute 'rand/1' vectors for everyone (vectorized)
            # V = X_r1 + F * (X_r2 - X_r3)
            mutant_rand = population[r1] + F_trial[:, None] * (population[r2] - population[r3])
            
            # 2. Compute 'current-to-best/1' vectors for everyone (vectorized)
            # V = X_current + F * (X_best - X_current) + F * (X_r1 - X_r2)
            # Note: We reuse r1, r2.
            diff_best = x_best - population
            diff_r1r2 = population[r1] - population[r2]
            mutant_best = population + F_trial[:, None] * diff_best + F_trial[:, None] * diff_r1r2
            
            # 3. Combine based on strategy mask
            mutant = np.where(strategy_mask[:, None], mutant_best, mutant_rand)
            
            # --- Crossover (Binomial) ---
            rand_j = np.random.randint(0, dim, pop_size)
            cross_mask = np.random.rand(pop_size, dim) < CR_trial[:, None]
            # Ensure at least one dimension is taken from mutant
            cross_mask[np.arange(pop_size), rand_j] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # --- Bound Handling (Reflection) ---
            # Reflect out-of-bound values: x' = min + (min - x) or max - (x - max)
            mask_l = trial_pop < min_b
            if np.any(mask_l):
                trial_pop[mask_l] = 2 * min_b[np.where(mask_l)[1]] - trial_pop[mask_l]
            
            mask_u = trial_pop > max_b
            if np.any(mask_u):
                trial_pop[mask_u] = 2 * max_b[np.where(mask_u)[1]] - trial_pop[mask_u]
                
            # Final clip to ensure numerical stability
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection ---
            for i in range(pop_size):
                if i % 10 == 0 and (datetime.now() - start_time) >= time_limit:
                    return global_best_val
                
                t_val = func(trial_pop[i])
                
                if t_val <= fitness[i]:
                    population[i] = trial_pop[i]
                    fitness[i] = t_val
                    F[i] = F_trial[i]
                    CR[i] = CR_trial[i]
                    
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_x = trial_pop[i].copy()
                        
    return global_best_val
