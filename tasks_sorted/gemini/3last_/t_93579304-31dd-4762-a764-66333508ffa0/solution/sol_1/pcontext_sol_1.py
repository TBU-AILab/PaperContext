#The following algorithm implements a **Self-Adaptive Differential Evolution (DE)** strategy with a **'current-to-pbest'** mutation scheme and a **Restart Mechanism**. 
#
#This approach improves upon standard DE by:
#1.  **Accelerating Convergence**: The `current-to-pbest` strategy guides the search towards the best solutions found so far (the top $p\%$ of the population) rather than random individuals.
#2.  **Adaptability**: Mutation factor ($F$) and Crossover rate ($CR$) are randomized per individual based on normal distributions. This eliminates the need for manual parameter tuning and handles different fitness landscape types effectively.
#3.  **Escaping Local Optima**: It monitors the population's diversity. If the population converges (standard deviation of fitness drops below a threshold) but time remains, it triggers a restart, re-initializing the population while preserving the best solution found (elitism).
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution with Restarts.
    
    Strategy: 'current-to-pbest/1' with randomized control parameters.
    Includes a restart mechanism to escape local optima if convergence is detected
    before max_time expires.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size logic: 
    # Moderate size allows for more generations within time limits.
    # We cap the size to ensure performance in high dimensions.
    pop_size = int(max(20, min(10 * dim, 60)))
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population randomly within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_sol = None
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Strict time check
        if (datetime.now() - start_time) >= limit:
            return best_val if best_val != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
            
    # --- Optimization Loop ---
    while (datetime.now() - start_time) < limit:
        
        # 1. Sort population to identify top performers (p-best)
        sorted_idxs = np.argsort(fitness)
        
        # 2. Restart Mechanism
        # If population diversity is lost (converged), restart to explore new areas
        # Threshold: Standard deviation of fitness is very small
        if np.std(fitness) < 1e-6:
            # Re-initialize population
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Elitism: Keep the global best solution in slot 0
            population[0] = best_sol
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Evaluate new population (skip elite at index 0)
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= limit: return best_val
                val = func(population[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = population[i].copy()
            
            # Re-sort after restart
            sorted_idxs = np.argsort(fitness)

        # 3. Evolution Cycle
        for i in range(pop_size):
            if (datetime.now() - start_time) >= limit:
                return best_val
            
            # --- Adaptive Parameters ---
            # Sample F from Normal(0.5, 0.3) clipped to [0.1, 1.0]
            F = np.random.normal(0.5, 0.3)
            F = np.clip(F, 0.1, 1.0)
            
            # Sample CR from Normal(0.5, 0.1) clipped to [0.0, 1.0]
            CR = np.random.normal(0.5, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Select X_pbest: Randomly from top 10% (p=0.1)
            p_top = max(2, int(0.1 * pop_size))
            pbest_idx = sorted_idxs[np.random.randint(0, p_top)]
            x_pbest = population[pbest_idx]
            
            # Select r1, r2: Random distinct indices, distinct from i
            while True:
                r1 = np.random.randint(0, pop_size)
                if r1 != i: break
            while True:
                r2 = np.random.randint(0, pop_size)
                if r2 != i and r2 != r1: break
            
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Create Mutant Vector
            mutant = population[i] + F * (x_pbest - population[i]) + F * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            # Mask where crossover occurs
            mask = np.random.rand(dim) < CR
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, population[i])
            
            # --- Boundary Handling ---
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            trial_val = func(trial)
            
            if trial_val < fitness[i]:
                fitness[i] = trial_val
                population[i] = trial
                if trial_val < best_val:
                    best_val = trial_val
                    best_sol = trial.copy()

    return best_val
