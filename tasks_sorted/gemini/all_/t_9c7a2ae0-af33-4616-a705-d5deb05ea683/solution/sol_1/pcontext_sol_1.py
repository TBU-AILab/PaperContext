#Here is an improved algorithm using **Self-Adaptive Differential Evolution (jDE)** with a **Current-to-Best** mutation strategy and a **Restart Mechanism**.
#
#### Rationale for Improvement
#The previous algorithm used a standard DE with fixed parameters and a random mutation strategy (`rand/1/bin`). This often leads to slow convergence or getting stuck in local optima. The improved version addresses this by:
#1.  **Current-to-Best Strategy**: Mutations are guided towards the best solution found so far, significantly speeding up convergence.
#2.  **Self-Adaptation (jDE)**: The algorithm automatically tunes the mutation factor `F` and crossover rate `CR` for each individual, removing the need for manual hyperparameter guessing.
#3.  **Restart Mechanism**: If the population converges (stagnates) before time runs out, the algorithm saves the best solution and restarts the rest of the population to explore new areas of the search space.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Self-Adaptive Differential Evolution (jDE) 
    with current-to-best mutation and restart mechanism.
    """
    # --- Configuration ---
    start_time = datetime.now()
    # Reserve a small buffer (5%) to ensure we return safely before timeout
    end_time = start_time + timedelta(seconds=max_time * 0.95)

    # Pre-process bounds for efficient numpy operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b

    # Population Size
    # Heuristic: 10-20 * dim is standard, but we cap it to ensure 
    # many generations can run within max_time.
    pop_size = int(max(10, min(50, 10 * dim)))

    # --- Initialization ---
    # Random initial population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))

    # jDE Adaptive Parameters (F and CR per individual)
    # Initialize with standard values: F=0.5, CR=0.9
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)

    # Global Best Tracking
    best_val = float('inf')
    best_vec = np.zeros(dim)

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_val if best_val != float('inf') else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()

    # --- Main Optimization Loop ---
    while datetime.now() < end_time:
        
        # 1. Restart Mechanism
        # If population diversity is too low (converged), restart to escape local optima.
        # We check the standard deviation of fitness values.
        if np.std(fitness) < 1e-8 and pop_size > 4:
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Elitism: Keep the best found so far at index 0
            pop[0] = best_vec
            
            # Reset adaptive parameters
            F.fill(0.5)
            CR.fill(0.9)
            
            # Re-evaluate (skip index 0 as it is already best_val)
            fitness[0] = best_val
            for i in range(1, pop_size):
                if datetime.now() >= end_time:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            continue

        # 2. Parameter Adaptation (jDE)
        # With probability 0.1, assign new random values
        # F ~ U(0.1, 1.0), CR ~ U(0.0, 1.0)
        mask_F = np.random.rand(pop_size) < 0.1
        mask_CR = np.random.rand(pop_size) < 0.1
        
        F[mask_F] = 0.1 + 0.9 * np.random.rand(mask_F.sum())
        CR[mask_CR] = np.random.rand(mask_CR.sum())

        # 3. Evolution Cycle
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_val
            
            # Select r1, r2 distinct from i
            # Efficient random choice
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
                
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
            
            # Mutation Strategy: current-to-best/1
            # V = X_i + F * (X_best - X_i) + F * (X_r1 - X_r2)
            # This balances exploitation (pull to best) and exploration (differential)
            x_i = pop[i]
            mutant = x_i + F[i] * (best_vec - x_i) + F[i] * (pop[r1] - pop[r2])
            
            # Crossover (Binomial)
            cross_points = np.random.rand(dim) < CR[i]
            # Ensure at least one parameter is taken from mutant
            cross_points[np.random.randint(dim)] = True
            
            trial = np.where(cross_points, mutant, x_i)
            
            # Bound constraints (Clip to valid range)
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()

    return best_val
