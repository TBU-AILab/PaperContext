#The algorithm below implements a **Self-Adaptive Differential Evolution (jDE)** strategy enhanced with a **Current-to-pBest** mutation scheme and a **Restart** mechanism. This approach is superior to standard Differential Evolution because:
#1.  **Adaptation**: It automatically tunes the control parameters ($F$ and $CR$) for each individual during the search, adapting to the function's landscape.
#2.  **Greedy Mutation**: The "Current-to-pBest" strategy guides the search towards the best solutions found so far (top 10%), converging significantly faster than random strategies.
#3.  **Restarts**: It detects premature convergence (when the population loses diversity) and restarts the search in unexplored areas while preserving the best-found solution, helping to escape local optima within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE)
    with Current-to-pBest mutation and Restart mechanism.
    """
    start_time = datetime.now()
    # Safety buffer: stop slightly before max_time to ensure return
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    # --- Configuration ---
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Population Size
    # 15*dim is a balanced choice. We cap it to avoid slowness in high dims.
    pop_size = min(100, max(20, 15 * dim))

    # jDE Control Parameters Initialization
    # Each individual has its own F and CR
    F = np.full(pop_size, 0.5)   # Scale factor
    CR = np.full(pop_size, 0.9)  # Crossover rate

    # --- Initialization ---
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, np.inf)

    best_val = np.inf
    best_vec = None

    # Initial Evaluation
    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_val if best_vec is not None else func(population[0])
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()

    # --- Optimization Loop ---
    while datetime.now() < end_time:
        
        # Sort population to identify top p% (p-best)
        # Using top 10% (0.1) provides a good balance of greediness and diversity
        sorted_indices = np.argsort(fitness)
        p_limit = max(1, int(pop_size * 0.10))
        top_p_indices = sorted_indices[:p_limit]
        
        # Iterate over population
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_val

            # 1. Parameter Adaptation (jDE Logic)
            # Create trial parameters
            f_i = F[i]
            cr_i = CR[i]
            
            # Update F with probability 0.1
            if np.random.rand() < 0.1:
                f_i = 0.1 + 0.9 * np.random.rand() # F in [0.1, 1.0]
            
            # Update CR with probability 0.1
            if np.random.rand() < 0.1:
                cr_i = np.random.rand() # CR in [0.0, 1.0]

            # 2. Mutation: Current-to-pBest/1
            # V = X_i + F_i * (X_pbest - X_i) + F_i * (X_r1 - X_r2)
            
            # Select X_pbest randomly from top p%
            pbest_idx = np.random.choice(top_p_indices)
            x_pbest = population[pbest_idx]
            
            # Select r1, r2 distinct from i
            # Simple rejection sampling
            r1, r2 = i, i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pop_size)
            
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Compute Mutant Vector
            mutant = population[i] + f_i * (x_pbest - population[i]) + f_i * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            cross_points = np.random.rand(dim) < cr_i
            # Ensure at least one dimension is mutated
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            
            trial_vec = np.where(cross_points, mutant, population[i])
            
            # Boundary Constraints (Clip)
            trial_vec = np.clip(trial_vec, min_b, max_b)
            
            # 4. Selection
            f_trial = func(trial_vec)
            
            if f_trial < fitness[i]:
                # Trial is better: replace individual and update control parameters
                fitness[i] = f_trial
                population[i] = trial_vec
                F[i] = f_i
                CR[i] = cr_i
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_vec.copy()
        
        # --- Restart Mechanism ---
        # If population has converged (low standard deviation), restart to find other optima
        if np.std(fitness) < 1e-6:
             # Check if enough time remains for a meaningful restart (e.g., > 10% of time)
             # This prevents restarting right before timeout
             if (end_time - datetime.now()) > timedelta(seconds=max_time * 0.1):
                 
                 # Preserve best, re-initialize the rest
                 population = min_b + np.random.rand(pop_size, dim) * diff_b
                 population[0] = best_vec # Keep the global best
                 
                 # Reset fitness array
                 fitness = np.full(pop_size, np.inf)
                 fitness[0] = best_val
                 
                 # Reset parameters for new exploration
                 F = np.full(pop_size, 0.5)
                 CR = np.full(pop_size, 0.9)
                 
                 # Evaluate new population (skip index 0)
                 for k in range(1, pop_size):
                     if datetime.now() >= end_time:
                         return best_val
                     
                     val = func(population[k])
                     fitness[k] = val
                     if val < best_val:
                         best_val = val
                         best_vec = population[k].copy()

    return best_val
