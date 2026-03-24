#Here is the improved algorithm. It implements a **Differential Evolution (DE) strategy with a "current-to-best" mutation scheme and a Restart mechanism**. This approach generally converges faster than standard DE and uses the restart capability to escape local optima, maximizing the utility of the available `max_time`.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution with 'current-to-best' strategy 
    and random restarts to utilize the full time budget.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Balance between speed and diversity.
    # Scaled by dim but clamped to ensure generations are fast.
    pop_size = int(max(20, min(100, 15 * dim)))
    
    # Pre-process bounds for efficient broadcasting
    bounds_arr = np.array(bounds)
    lower_bound = bounds_arr[:, 0]
    upper_bound = bounds_arr[:, 1]
    bound_diff = upper_bound - lower_bound
    
    # Track the global best fitness found across all restarts
    global_best_fitness = float('inf')
    
    # --- Main Loop (Restarts) ---
    while True:
        # Check time budget before starting a new population
        if time.time() - start_time >= max_time:
            return global_best_fitness
        
        # --- Initialization ---
        # Initialize population randomly within bounds
        population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
        fitnesses = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            val = func(population[i])
            fitnesses[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val

        # Index of the best individual in the current population
        current_best_idx = np.argmin(fitnesses)
        
        # --- Evolution Loop ---
        while True:
            # Check time budget frequently
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            # Restart Check: If population fitness variance is negligible, we have converged.
            # Break inner loop to restart with new random population.
            if np.max(fitnesses) - np.min(fitnesses) < 1e-8:
                break

            # Dynamic Parameters
            # Dither 'F' (mutation factor) between 0.5 and 1.0 to help exploration
            F = 0.5 + 0.5 * np.random.rand() 
            # High 'CR' (crossover probability) works well with current-to-best
            CR = 0.9 
            
            # --- Vectorized Trial Generation ---
            # Strategy: DE/current-to-best/1/bin
            # Equation: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            
            # Generate indices for difference vectors
            # r1 and r2 are permutations of the population indices
            r1 = np.random.permutation(pop_size)
            r2 = np.roll(r1, 1) # Shift to ensure r1 != r2
            
            x_best = population[current_best_idx]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Mutation Step (Vectorized)
            # x_best is broadcasted to match population shape
            mutant = population + F * (x_best - population) + F * (x_r1 - x_r2)
            
            # Crossover Step (Binomial)
            cross_mask = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one parameter comes from the mutant (DE requirement)
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            
            final_mask = np.logical_or(cross_mask, j_rand_mask)
            
            # Create trial population
            trial_population = np.where(final_mask, mutant, population)
            
            # Boundary Constraint (Clip values to bounds)
            trial_population = np.clip(trial_population, lower_bound, upper_bound)
            
            # --- Selection Step ---
            # Evaluate trial solutions and replace parents if trial is better
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return global_best_fitness
                
                trial_val = func(trial_population[i])
                
                # Greedy selection
                if trial_val <= fitnesses[i]:
                    fitnesses[i] = trial_val
                    population[i] = trial_population[i]
                    
                    # Update global best found so far
                    if trial_val < global_best_fitness:
                        global_best_fitness = trial_val
                    
                    # Update local best index if current individual improves beyond current local best
                    if trial_val < fitnesses[current_best_idx]:
                        current_best_idx = i

    return global_best_fitness
