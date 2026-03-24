#Here is a fully operational, self-contained implementation using a **Restarted "Current-to-Best" Differential Evolution** algorithm.
#
#### Improvements over the previous Random/Standard DE:
#1.  **Current-to-Best Strategy:** Instead of mutating random vectors (which is slow to converge), this algorithm directs the search towards the best solution found so far while maintaining exploration. This significantly speeds up convergence.
#2.  **Restart Mechanism:** If the population converges (agents become too similar) and time remains, the algorithm saves the best solution and "explodes" the rest of the population to new random positions to find other potential global minima.
#3.  **Latin Hypercube Initialization:** Instead of pure random starting points, it uses stratified sampling to ensure the initial population covers the search space evenly.
#4.  **Dithered Parameters:** The mutation factor (`F`) and crossover probability (`CR`) are randomized slightly every generation to prevent the algorithm from getting stuck in search patterns.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # --- Configuration ---
    # Population size: 
    # A multiplier of 15x dim provides good coverage, clamped to keep speed high.
    pop_size = int(np.clip(dim * 15, 20, 60))
    
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    # Convert bounds to numpy arrays for vectorized math
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Helper: Check if time is up
    def is_time_up():
        return datetime.now() >= end_time

    # --- Initialization (Latin Hypercube Sampling) ---
    # Ensures better initial coverage than pure random
    # Generate stratified samples
    rng = np.random.rand(pop_size, dim)
    for d in range(dim):
        rng[:, d] = (np.argsort(rng[:, d]) + np.random.rand(pop_size)) / pop_size
    
    population = min_b + rng * diff_b
    fitness = np.full(pop_size, float('inf'))

    best_val = float('inf')
    best_vec = None
    best_idx = -1

    # Initial Evaluation
    for i in range(pop_size):
        if is_time_up(): return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()
            best_idx = i

    # --- Main Optimization Loop ---
    while not is_time_up():
        
        # 1. Restart Mechanism
        # If population variance is too low (converged), restart to find new basins
        # while keeping the elite (best) solution.
        std_dev = np.std(fitness)
        if std_dev < 1e-6:
            # Re-initialize population randomly
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_vec # Keep the elite
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Re-evaluate new population (skip elite at index 0)
            for i in range(1, pop_size):
                if is_time_up(): return best_val
                val = func(population[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = population[i].copy()
                    best_idx = i
            continue

        # 2. Dynamic Parameters (Dithering)
        # Randomize F and CR slightly per generation to improve robustness
        F = 0.5 + 0.4 * np.random.rand()  # Mutation factor [0.5, 0.9]
        CR = 0.8 + 0.2 * np.random.rand() # Crossover prob [0.8, 1.0]

        # 3. Evolution Cycle
        for i in range(pop_size):
            if is_time_up(): return best_val

            # Target: Current-to-Best Strategy
            # V = Xi + F*(Best - Xi) + F*(r1 - r2)
            # This balances greediness (move to best) with exploration (difference)
            
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            x_i = population[i]
            x_best = best_vec
            x_r1 = population[r1]
            x_r2 = population[r2]

            # Mutation
            mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            
            # Boundary Constraint (Clip)
            mutant = np.clip(mutant, min_b, max_b)

            # Crossover (Binomial)
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points): 
                # Ensure at least one parameter changes
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, x_i)

            # Selection
            f_trial = func(trial)

            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
                    best_idx = i

    return best_val
