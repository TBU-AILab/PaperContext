#An improved algorithm for finding the minimum of a function within a limited time. This implementation utilizes a **Vectorized jDE (Self-Adaptive Differential Evolution)** enhanced with **current-to-pbest mutation**, **Boundary Reflection**, and an **Elitist Restart Strategy**.
#
##### Improvements Explained:
#1.  **Current-to-pbest Mutation**: Replaces the standard `rand/1` mutation with `current-to-pbest/1`. This strategy directs the search towards the top $p\%$ of the population while maintaining diversity through difference vectors. It significantly speeds up convergence compared to random mutation.
#2.  **Corrected jDE Logic**: The previous best algorithm updated control parameters ($F$, $CR$) probabilistically regardless of success. This version strictly implements jDE logic where parameters are only updated (learned) if the trial vector yields a better fitness, allowing the algorithm to adapt effectively to the landscape.
#3.  **Elitist Restarts**: Unlike the previous implementation which reset the population entirely upon stagnation, this version injects the global best solution found so far into the new population. This prevents the loss of progress and turns the restart into a heavy exploration phase around the known optimum.
#4.  **Vectorized Operations & Optimized Time Management**: The algorithm leverages `numpy` for batch creation of trial vectors and reduces the overhead of time-checking system calls by batching checks, maximizing the number of function evaluations within `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized jDE with current-to-pbest mutation
    and Elitist Restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 
    # Larger populations explore better but slow down generations.
    # We use a dynamic size roughly 15x dimension, capped for performance.
    pop_size = int(max(20, 15 * dim))
    if pop_size > 80:
        pop_size = 80
        
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    global_best_x = None
    
    # jDE Adaptive Parameters Configuration
    # Probability to update F and CR
    tau_f = 0.1
    tau_cr = 0.1
    
    # --- Restart Loop ---
    # Restarts the algorithm if the population stagnates (converges)
    while True:
        # Check time before starting a new session
        if (datetime.now() - start_time) >= time_limit:
            return global_best_val
        
        # 1. Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitist Restart: Inject global best to preserve progress
        if global_best_x is not None:
            population[0] = global_best_x.copy()
            
        # 2. Evaluate Initial Population
        fitness = np.full(pop_size, float('inf'))
        
        for i in range(pop_size):
            # Optimization: Check time periodically (every 10 evals) to reduce overhead
            if i % 10 == 0 and (datetime.now() - start_time) >= time_limit:
                return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_x = population[i].copy()
                
        # Initialize jDE Control Parameters (F and CR)
        # F: Mutation scale factor, CR: Crossover probability
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # --- Evolution Loop ---
        while True:
            # Check time
            if (datetime.now() - start_time) >= time_limit:
                return global_best_val
            
            # Sort population by fitness (Best at index 0)
            # This is required for 'current-to-pbest' mutation strategy
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            # Keep F and CR attached to their individuals
            F = F[sorted_indices]
            CR = CR[sorted_indices]
            
            # Check Stagnation (Convergence)
            # If standard deviation is tiny, or range is negligible, restart
            if np.std(fitness) < 1e-6 or (fitness[-1] - fitness[0]) < 1e-7:
                break
            
            # --- Vectorized Generation of Trials ---
            
            # 1. Generate new F and CR trials (jDE logic)
            # If a trial is successful, these values are adopted.
            F_trial = F.copy()
            CR_trial = CR.copy()
            
            # Masks for probabilistic update
            mask_f = np.random.rand(pop_size) < tau_f
            mask_cr = np.random.rand(pop_size) < tau_cr
            
            # Update F: 0.1 + 0.9 * rand (Range 0.1 to 1.0)
            F_trial[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
            # Update CR: rand (Range 0.0 to 1.0)
            CR_trial[mask_cr] = np.random.rand(np.sum(mask_cr))
            
            # 2. Mutation: current-to-pbest/1
            # Formula: V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            # This balances exploitation (towards pbest) and exploration (difference vector).
            
            # Select p-best (randomly from top 10%)
            top_p_count = max(2, int(pop_size * 0.1))
            # Indices [0, ..., top_p_count-1] are best due to sorting
            pbest_indices = np.random.randint(0, top_p_count, pop_size)
            x_pbest = population[pbest_indices]
            
            # Select r1, r2 randomly from population
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Compute Mutant Vector
            # Note: 'population' represents X_current
            diff_pbest = x_pbest - population
            diff_r = x_r1 - x_r2
            
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            mutant = population + F_trial[:, None] * diff_pbest + F_trial[:, None] * diff_r
            
            # 3. Crossover (Binomial)
            rand_j = np.random.randint(0, dim, pop_size)
            cross_mask = np.random.rand(pop_size, dim) < CR_trial[:, None]
            # Ensure at least one dimension is taken from mutant
            cross_mask[np.arange(pop_size), rand_j] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # 4. Bound Handling (Reflection)
            # Reflection preserves distribution better than clipping
            
            # Lower bounds
            mask_l = trial_pop < min_b
            if np.any(mask_l):
                # x' = min + (min - x)
                trial_pop[mask_l] = 2 * min_b[np.where(mask_l)[1]] - trial_pop[mask_l]
                
            # Upper bounds
            mask_u = trial_pop > max_b
            if np.any(mask_u):
                # x' = max - (x - max)
                trial_pop[mask_u] = 2 * max_b[np.where(mask_u)[1]] - trial_pop[mask_u]
                
            # Final safety clip
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 5. Selection
            # Evaluate trials and apply greedy selection
            for i in range(pop_size):
                # Check time inside loop periodically
                if i % 10 == 0 and (datetime.now() - start_time) >= time_limit:
                    return global_best_val
                
                t_val = func(trial_pop[i])
                
                if t_val <= fitness[i]:
                    # Trial is better: Update population and control parameters
                    population[i] = trial_pop[i]
                    fitness[i] = t_val
                    F[i] = F_trial[i]
                    CR[i] = CR_trial[i]
                    
                    # Update Global Best
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_x = trial_pop[i].copy()
                        
    return global_best_val
