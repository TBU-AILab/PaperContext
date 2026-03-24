#The following algorithm implements a **Restarting Adaptive Differential Evolution (DE)** strategy. This approach improves upon standard DE by:
#1.  **Strategy**: Using `DE/current-to-best/1/bin`, which balances exploitation (moving towards the best solution) and exploration (random perturbations).
#2.  **Adaptivity**: Dithering the mutation factor ($F$) and crossover rate ($CR$) to prevent getting stuck in local optima and to handle various landscape types.
#3.  **Restarts**: Detecting convergence or stagnation to restart the population, preventing the algorithm from wasting time in a local minimum.
#4.  **Vectorization**: Utilizing NumPy for efficient generation of trial vectors, maximizing the number of function evaluations within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Adaptive Differential Evolution (DE/current-to-best/1).
    """
    start_time = datetime.now()
    # Set a safe time limit slightly less than max_time to ensure return
    time_limit = timedelta(seconds=max_time * 0.98 if max_time > 0.5 else max_time)
    
    # Pre-process bounds for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    best_fitness = float('inf')
    
    # Population size: A trade-off. 15*dim is robust; capped at 100 for speed on high dims.
    pop_size = min(max(20, 15 * dim), 100)
    
    # Helper to check remaining time
    def has_time():
        return (datetime.now() - start_time) < time_limit

    # Main Loop: Restarts the population if converged or stagnated
    while has_time():
        
        # Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if not has_time():
                return best_fitness
            val = func(population[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
        
        # Track the index of the best individual in the current population
        best_idx = np.argmin(fitness)
        stagnation_counter = 0
        
        # Evolution Loop
        while has_time():
            
            # --- Adaptive Parameters ---
            # F (Mutation Factor): Randomize between 0.5 and 1.0 to vary step sizes
            F = 0.5 + 0.5 * np.random.rand()
            # CR (Crossover Rate): Randomize between 0.8 and 1.0 to preserve variable dependencies
            CR = 0.8 + 0.2 * np.random.rand()
            
            # --- Vectorized Trial Generation (DE/current-to-best/1/bin) ---
            # V = X_current + F * (X_best - X_current) + F * (X_r1 - X_r2)
            
            # Select random indices for r1 and r2
            # (Collisions with i or best_idx are ignored for efficiency as they just reduce perturbation slightly)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            current = population
            x_best = population[best_idx]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Create Mutant Vectors
            # "current-to-best" strategy pulls individuals towards the best found so far
            mutant = current + F * (x_best - current) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            # Create a mask of boolean values
            cross_mask = np.random.rand(pop_size, dim) < CR
            # Ensure at least one parameter is mutated for each individual
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            # Create Trial Vectors
            trial_pop = np.where(cross_mask, mutant, current)
            
            # Boundary Constraint Handling (Clip)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection (Evaluation) ---
            improved_in_generation = False
            
            for i in range(pop_size):
                if not has_time():
                    return best_fitness
                
                # Evaluate fitness
                f_trial = func(trial_pop[i])
                
                # Greedy selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update global best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                    
                    # Update local population best tracker
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        improved_in_generation = True
            
            # --- Restart Criteria ---
            # Check for stagnation
            if not improved_in_generation:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # Restart if stagnated for too long (e.g., 20 gens) or if population converged (low std dev)
            if stagnation_counter > 20:
                break
                
            if np.std(fitness) < 1e-6:
                break
                
    return best_fitness
