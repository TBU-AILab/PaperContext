#To improve upon the previous algorithm (standard Differential Evolution), we can implement a **Self-Adaptive Differential Evolution with Restarts and Local Search (Polishing)**.
#
#**Key Improvements:**
#1.  **Strategy Change (Current-to-Best):** Instead of random mutation, we use `current-to-best`. This directs the search towards the best solution found so far while maintaining diversity, accelerating convergence.
#2.  **Dynamic Parameters (Adaptation):** Instead of fixed `F` and `CR`, we jitter these parameters per generation/individual. This prevents the algorithm from getting stuck due to poor parameter choices.
#3.  **Restarts:** The previous algorithm ran one continuous loop. If it got stuck in a local minimum, it stayed there. The new algorithm detects stagnation (lack of improvement), performs a local search, and then restarts the population to find other potential basins of attraction.
#4.  **Local Search (Polishing):** When the DE phase stagnates or time is running out, we switch to a localized coordinate descent search around the best candidate to "polish" the result and find the precise bottom of the valley.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Self-Adaptive Differential Evolution with Restarts
    and Local Search Polishing.
    """
    # --- Configuration ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Population Strategy
    # Smaller population allows more generations/restarts within time limit
    pop_size = max(10, int(5 * dim)) 
    pop_size = min(pop_size, 40) # Cap size to ensure speed
    
    # Pre-process bounds
    bounds_array = np.array(bounds)
    min_b = bounds_array[:, 0]
    max_b = bounds_array[:, 1]
    diff_b = max_b - min_b
    
    # Global state
    global_best_val = float('inf')
    global_best_pos = None

    def check_timeout():
        return (datetime.now() - start_time) >= limit

    # --- Local Search Helper ---
    def local_search(current_best_pos, current_best_val):
        """
        Performs a simple coordinate descent (Pattern Search) around the best point 
        to refine the solution.
        """
        pos = current_best_pos.copy()
        val = current_best_val
        
        # Initial step size relative to bounds
        step_size = 0.05
        min_step = 1e-6
        
        while step_size > min_step:
            if check_timeout():
                return pos, val
            
            improved = False
            # Try moving in each dimension
            for d in range(dim):
                original_d = pos[d]
                step = step_size * diff_b[d]
                
                # Try positive direction
                pos[d] = np.clip(original_d + step, min_b[d], max_b[d])
                new_val = func(pos)
                if new_val < val:
                    val = new_val
                    improved = True
                    continue # Keep the change
                
                # Try negative direction
                pos[d] = np.clip(original_d - step, min_b[d], max_b[d])
                new_val = func(pos)
                if new_val < val:
                    val = new_val
                    improved = True
                    continue
                
                # Revert if no improvement
                pos[d] = original_d
                
            if not improved:
                step_size *= 0.5 # Refine step size
                
        return pos, val

    # --- Main Optimization Loop (Restarts) ---
    while not check_timeout():
        
        # 1. Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Initial evaluation
        for i in range(pop_size):
            if check_timeout():
                return global_best_val if global_best_val != float('inf') else fitness[0]
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_pos = population[i].copy()

        best_idx = np.argmin(fitness)
        pop_best_val = fitness[best_idx]
        
        # 2. Evolutionary Loop
        stagnation_counter = 0
        max_stagnation = 20 + dim  # Allow some time to converge
        
        while stagnation_counter < max_stagnation:
            if check_timeout():
                return global_best_val
            
            prev_best = pop_best_val
            
            # Create a new population array to store updates
            # We use "Current-to-Best/1" strategy:
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            
            # Generate random indices
            # We need r1 != r2 != i
            idxs = np.random.randint(0, pop_size, size=(pop_size, 2))
            
            # Jitter F and CR per generation for robustness
            # F around 0.5 - 0.8, CR around 0.8 - 0.9
            F = 0.5 + 0.4 * np.random.rand()
            CR = 0.8 + 0.1 * np.random.rand()
            
            best_pos = population[best_idx]
            
            for i in range(pop_size):
                if check_timeout():
                    return global_best_val

                r1, r2 = idxs[i]
                # Ensure distinct indices
                while r1 == i or r1 == best_idx:
                    r1 = np.random.randint(0, pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(0, pop_size)
                    
                x = population[i]
                xr1 = population[r1]
                xr2 = population[r2]
                
                # Mutation (Current-to-best)
                mutant = x + F * (best_pos - x) + F * (xr1 - xr2)
                
                # Crossover (Binomial)
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                
                trial = np.where(cross_points, mutant, x)
                trial = np.clip(trial, min_b, max_b)
                
                # Selection
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < pop_best_val:
                        pop_best_val = f_trial
                        best_idx = i
                        if f_trial < global_best_val:
                            global_best_val = f_trial
                            global_best_pos = trial.copy()
            
            # Stagnation check
            if pop_best_val < prev_best:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
        # 3. Post-Evolution Polishing (Local Search)
        # If the population converged or stagnated, refine the best solution found in this run
        refined_pos, refined_val = local_search(population[best_idx], fitness[best_idx])
        
        if refined_val < global_best_val:
            global_best_val = refined_val
            global_best_pos = refined_pos
            
        # Loop restarts with new random population to escape local optima
        
    return global_best_val
