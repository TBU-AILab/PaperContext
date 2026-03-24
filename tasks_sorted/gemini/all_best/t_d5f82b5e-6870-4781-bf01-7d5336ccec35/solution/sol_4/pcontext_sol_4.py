#The improved algorithm is a self-contained implementation of **Restarting Differential Evolution (DE)** using the **Current-to-Best/1/bin** strategy with **Dithered Mutation**.
#
##### Improvements Overview:
#1.  **Restart Mechanism**: The standard DE algorithm can sometimes get trapped in local optima or stagnate. This algorithm monitors population variance and improvement. If the population converges (variance drops) or stagnates (no improvement for a set number of generations), it triggers a **restart**.
#2.  **Seeded Restarts**: When restarting, the algorithm keeps the global best solution found so far and places it into the new random population. This effectively turns subsequent restarts into focused **intensification** phases around the best-known area, while the rest of the population provides fresh **exploration**.
#3.  **Strategy**: Uses `DE/current-to-best/1/bin`. This strategy converges faster than `DE/rand` by utilizing gradient information inherent in the population's best members.
#4.  **Dithered Mutation**: The mutation factor `F` is randomized (dithered) for each individual to increase diversity and robustness against different fitness landscapes.
#
##### Algorithm Code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Adaptive to dimension.
    # A size of 15 * dim is a robust heuristic, clamped to [20, 60]
    # to ensure responsiveness within time limits while maintaining diversity.
    pop_size = int(15 * dim)
    if pop_size < 20: pop_size = 20
    if pop_size > 60: pop_size = 60
    
    CR = 0.9      # Crossover Probability
    # F (Mutation Factor) is dithered dynamically in the loop
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution across all restarts
    global_best_val = float('inf')
    global_best_vec = np.zeros(dim)
    
    # Helper to check time budget strictly
    def time_is_up():
        return datetime.now() - start_time >= limit

    # --- Main Optimization Loop (Outer Loop for Restarts) ---
    while not time_is_up():
        
        # 1. Initialize Population
        # Uniform random distribution within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # 2. Elitism/Memory Injection
        # If we have a best solution from a previous run, inject it.
        # This guides the new random population towards the promising region.
        start_idx = 0
        if global_best_val != float('inf'):
            population[0] = global_best_vec
            fitness[0] = global_best_val
            start_idx = 1
            
        # 3. Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if time_is_up(): return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = population[i].copy()
                
        # 4. Evolution Loop (Inner Loop for Current Run)
        stagnation_count = 0
        max_stagnation = 25  # Threshold to trigger restart
        prev_run_best = np.min(fitness)
        
        while not time_is_up():
            
            # Identify the best individual in the current population
            # (Used for the current-to-best mutation strategy)
            best_idx = np.argmin(fitness)
            current_best_val = fitness[best_idx]
            x_best = population[best_idx]
            
            # --- Generation Iteration ---
            for i in range(pop_size):
                if time_is_up(): return global_best_val
                
                # Selection: Choose distinct random indices r1, r2 != i
                # Optimized random selection
                idxs = [idx for idx in range(pop_size) if idx != i]
                r1 = idxs[np.random.randint(0, len(idxs))]
                r2 = idxs[np.random.randint(0, len(idxs))]
                while r2 == r1:
                    r2 = idxs[np.random.randint(0, len(idxs))]
                
                x_i = population[i]
                x_r1 = population[r1]
                x_r2 = population[r2]
                
                # Mutation: DE/current-to-best/1
                # V = Xi + F * (Xbest - Xi) + F * (Xr1 - Xr2)
                # F is dithered between 0.5 and 1.0 to improve search diversity
                F = 0.5 + 0.5 * np.random.rand()
                
                mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
                
                # Crossover: Binomial
                mask = np.random.rand(dim) < CR
                # Ensure at least one parameter is taken from mutant
                if not np.any(mask):
                    mask[np.random.randint(0, dim)] = True
                    
                trial = np.where(mask, mutant, x_i)
                
                # Boundary Control: Clip values
                trial = np.clip(trial, min_b, max_b)
                
                # Evaluation
                f_trial = func(trial)
                
                # Selection: Greedy
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    
                    # Update Global Best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_vec = trial.copy()

            # --- Restart Logic ---
            # Check for stagnation or convergence
            
            # 1. Update stagnation counter
            # If current run's best improved, reset counter
            if current_best_val < prev_run_best:
                prev_run_best = current_best_val
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # 2. Check Variance (Convergence)
            # If the population is too clustered, we are stuck.
            fit_std = np.std(fitness)
            
            # Trigger restart if stagnant or converged
            if fit_std < 1e-8 or stagnation_count >= max_stagnation:
                break # Break inner loop -> triggers re-initialization in outer loop

    return global_best_val
