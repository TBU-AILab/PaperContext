#Here is the fully operational and self-contained Python code implementing a **Restarting Adaptive Differential Evolution** algorithm. 
#
#This algorithm improves upon the standard Differential Evolution by implementing:
#1.  **Restart Mechanism**: Prevents the algorithm from wasting time in local minima. If the population converges or stagnates, it restarts with a fresh population while preserving the global best.
#2.  **Current-to-Best Strategy**: Instead of random exploration, it biases mutation towards the best found solution (`DE/current-to-best/1`), significantly speeding up convergence.
#3.  **Dithering**: Randomizes the mutation factor ($F$) and crossover rate ($CR$) per step to maintain population diversity and adapt to different surface landscapes.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Adaptive Differential Evolution 
    (DE/current-to-best/1/bin).
    """
    t_start = time.time()
    best_fitness = float('inf')

    # --- Pre-processing ---
    # Convert bounds to numpy arrays for vectorized math
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    bound_diff = ub - lb
    
    # --- Hyperparameters ---
    # Population size: Smaller than standard DE to allow fast iterations and restarts.
    # Clamped between 20 and 50 to ensure balance.
    pop_size = min(max(20, 5 * dim), 50)
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # 1. Initialize Population
        # Using random uniform initialization
        pop = lb + np.random.rand(pop_size, dim) * bound_diff
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if time.time() - t_start >= max_time:
                return best_fitness
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val

        # Track best in current population for the strategy
        current_best_idx = np.argmin(fitness)
        
        # Stagnation counter for this restart
        stagnation_count = 0
        last_gen_best = fitness[current_best_idx]

        # --- Evolution Loop ---
        while True:
            # Check global time limit
            if time.time() - t_start >= max_time:
                return best_fitness

            # 2. Check for Restart Conditions
            # Condition A: Stagnation (no improvement for 25 generations)
            if stagnation_count > 25:
                break 
            
            # Condition B: Convergence (standard deviation of fitness is tiny)
            if np.std(fitness) < 1e-6:
                break
                
            # Prepare for next generation
            best_in_gen_val = fitness[current_best_idx]
            
            # 3. Iterate over population
            for i in range(pop_size):
                if time.time() - t_start >= max_time:
                    return best_fitness

                # --- Mutation: DE/current-to-best/1 ---
                # V = Xi + F * (Xbest - Xi) + F * (Xr1 - Xr2)
                # This strategy pulls individuals towards the best, speeding up convergence.
                
                # Select distinct random indices r1, r2 != i
                candidates = [idx for idx in range(pop_size) if idx != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                
                # --- Adaptive Parameters (Dithering) ---
                # Randomize F and CR slightly to handle different scales and avoid local minima
                F = 0.5 + 0.5 * np.random.rand()  # [0.5, 1.0]
                CR = 0.8 + 0.2 * np.random.rand() # [0.8, 1.0]

                x_current = pop[i]
                x_best = pop[current_best_idx]
                x_r1 = pop[r1]
                x_r2 = pop[r2]
                
                # Compute Mutant Vector
                mutant = x_current + F * (x_best - x_current) + F * (x_r1 - x_r2)
                
                # --- Crossover: Binomial ---
                cross_points = np.random.rand(dim) < CR
                # Ensure at least one dimension is changed
                j_rand = np.random.randint(dim)
                cross_points[j_rand] = True
                
                # Construct Trial Vector
                trial = np.where(cross_points, mutant, x_current)
                
                # --- Boundary Handling ---
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                
                # --- Selection ---
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                    
                    # Update Current Best (Greedy update)
                    if f_trial < fitness[current_best_idx]:
                        current_best_idx = i

            # End of Generation Check
            current_gen_best = fitness[current_best_idx]
            if current_gen_best < last_gen_best:
                last_gen_best = current_gen_best
                stagnation_count = 0
            else:
                stagnation_count += 1

    return best_fitness
