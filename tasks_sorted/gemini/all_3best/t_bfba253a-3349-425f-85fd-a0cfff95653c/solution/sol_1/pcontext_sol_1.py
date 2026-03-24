#An improved algorithm implementation is provided below. This solution utilizes **Differential Evolution (DE)** with a **'current-to-best' mutation strategy** and a **random restart mechanism**.
#
#### Key Improvements:
#1.  **Current-to-Best Mutation**: Unlike standard DE which blindly combines random vectors, this strategy incorporates the direction towards the best solution found so far ($\vec{x}_{best}$). This accelerates convergence significantly.
#    *   Equation: $\vec{v} = \vec{x}_i + F \cdot (\vec{x}_{best} - \vec{x}_i) + F \cdot (\vec{x}_{r1} - \vec{x}_{r2})$.
#2.  **Restart Mechanism**: DE can sometimes converge prematurely to a local minimum. The algorithm monitors population variance and improvement stagnation. If the population becomes stagnant, it triggers a restart (re-initializing the population) while preserving the global best solution (elitism). This allows the algorithm to explore different regions of the search space within the remaining time.
#3.  **Dynamic Parameter Control**: The population size is dynamically scaled based on dimension but capped to ensure rapid iteration. The mutation factor $F$ is randomized (dithered) per step to maintain diversity.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) with a 'current-to-best' 
    mutation strategy and a restart mechanism to escape local optima within a time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper to check strict time limits ---
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    # --- Configuration ---
    # Population size: Balance between diversity and iteration speed.
    # We clip it to ensure we don't spend too much time on one generation.
    pop_size = int(np.clip(10 * dim, 15, 50))
    
    # DE Parameters
    # F (Mutation Factor): Randomized per individual (Dithering)
    # CR (Crossover Probability): High probability to preserve good linkages
    CR = 0.9 

    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track global best across restarts
    global_best_fitness = float('inf')
    global_best_sol = None

    # --- Main Optimization Loop (Handles Restarts) ---
    while True:
        if is_time_up(): return global_best_fitness

        # 1. Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))

        # Elitism: If we restarted, carry over the single best solution found so far
        start_idx = 0
        if global_best_sol is not None:
            population[0] = global_best_sol
            fitness[0] = global_best_fitness
            start_idx = 1
        
        # Evaluate initial population
        for i in range(start_idx, pop_size):
            if is_time_up(): return global_best_fitness
            val = func(population[i])
            fitness[i] = val
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_sol = population[i].copy()

        # Restart triggers
        stagnation_counter = 0
        prev_min_val = np.min(fitness)
        
        # --- Evolution Loop ---
        while True:
            if is_time_up(): return global_best_fitness
            
            # Identify best in current population for 'current-to-best' logic
            idx_best = np.argmin(fitness)
            x_best = population[idx_best]
            current_best_val = fitness[idx_best]

            # --- Convergence/Stagnation Checks ---
            # If population variance is negligible, we have converged
            if np.std(fitness) < 1e-6:
                break # Break inner loop to Restart
            
            # If no significant improvement for many generations
            if current_best_val >= prev_min_val - 1e-8:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                prev_min_val = current_best_val
            
            if stagnation_counter > 40: # Stagnation threshold
                break # Break inner loop to Restart

            # --- Iterate through Population ---
            for i in range(pop_size):
                if is_time_up(): return global_best_fitness
                
                # Mutation Strategy: DE/current-to-best/1/bin
                # V = Xi + F * (Xbest - Xi) + F * (Xr1 - Xr2)
                
                # Select two random distinct individuals r1, r2, distinct from i
                # (Simple rejection sampling for speed)
                candidates = [k for k in range(pop_size) if k != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                
                x_i = population[i]
                x_r1 = population[r1]
                x_r2 = population[r2]
                
                # Dither F between 0.5 and 1.0 to prevent stagnation
                F = 0.5 + 0.5 * np.random.rand()
                
                # Create Mutant Vector
                # This vector is pulled towards the best solution (exploitation)
                # while maintaining difference vector variance (exploration)
                mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
                
                # Boundary Constraint (Clip)
                mutant = np.clip(mutant, min_b, max_b)
                
                # Crossover (Binomial)
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                    
                trial = np.where(cross_points, mutant, x_i)
                
                # Selection (Greedy)
                f_trial = func(trial)
                
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    
                    # Update Global Best
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial
                        global_best_sol = trial.copy()
                    
                    # Opportunistic update of local best for faster convergence in this generation
                    if f_trial < current_best_val:
                        current_best_val = f_trial
                        x_best = trial

    return global_best_fitness
