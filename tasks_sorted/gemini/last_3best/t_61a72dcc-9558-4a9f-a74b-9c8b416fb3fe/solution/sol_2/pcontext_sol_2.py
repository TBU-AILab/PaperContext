#Here is the improved algorithm. It implements a **Self-Adaptive Differential Evolution (jDE)** strategy with **Current-to-pBest** mutation and a **Restart** mechanism.
#
#**Key improvements over previous versions:**
#1.  **Current-to-pBest Mutation:** Instead of following only the single best individual (which causes premature convergence), individuals are attracted to a random solution from the top $p\%$ (p-best). This balances convergence speed with population diversity.
#2.  **Self-Adaptive Parameters (jDE):** The mutation factor ($F$) and crossover rate ($CR$) are adapted for each individual during evolution. Successful parameters are propagated, while unsuccessful ones are discarded. This removes the need for manual tuning and adapts the search behavior to the landscape.
#3.  **Restarts with Stagnation Detection:** The algorithm monitors population variance and fitness stagnation. If the search stalls or converges to a local optimum, it automatically restarts with a fresh population to utilize the remaining time budget effectively.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE) 
    with current-to-pbest mutation and restart mechanism.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    # Convert bounds to numpy array for fast broadcasting
    bounds_arr = np.array(bounds)
    lower_bound = bounds_arr[:, 0]
    upper_bound = bounds_arr[:, 1]
    bound_diff = upper_bound - lower_bound
    
    # --- Configuration ---
    # Population size: Balance between diversity and speed.
    # We clip it to ensure generations are fast enough.
    pop_size = int(max(20, min(100, 15 * dim)))
    
    # Global best tracker across all restarts
    global_best_fitness = float('inf')
    
    # --- Main Loop (Restarts) ---
    while True:
        # Check total time budget
        if time.time() - start_time >= max_time:
            return global_best_fitness
        
        # --- Initialization ---
        # Initialize population randomly within bounds
        population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
        fitnesses = np.full(pop_size, float('inf'))
        
        # Initialize adaptive parameters for each individual (jDE strategy)
        # F: Mutation factor (controls step size)
        # CR: Crossover probability
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            val = func(population[i])
            fitnesses[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val

        # Convergence tracking variables
        gen_best_fitness = np.min(fitnesses)
        stagnation_counter = 0
        
        # --- Evolution Loop ---
        while True:
            # Check time budget at start of generation
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            # --- 1. Parameter Adaptation (jDE) ---
            # Create trial parameters. 
            # With prob 0.1, assign new random values; otherwise keep old.
            F_trial = F.copy()
            CR_trial = CR.copy()
            
            mask_F = np.random.rand(pop_size) < 0.1
            mask_CR = np.random.rand(pop_size) < 0.1
            
            # F takes value in [0.1, 1.0]
            F_trial[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
            # CR takes value in [0.0, 1.0]
            CR_trial[mask_CR] = np.random.rand(np.sum(mask_CR))
            
            # --- 2. Mutation Strategy: current-to-pbest/1 ---
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            
            # Identify p-best (top p% of population)
            # p is typically 5-20%. We use 10% with a minimum of 2.
            p_limit = max(2, int(0.1 * pop_size))
            sorted_indices = np.argsort(fitnesses)
            pbest_indices = sorted_indices[:p_limit]
            
            # Select random pbest target for each individual
            chosen_pbest_indices = np.random.choice(pbest_indices, pop_size)
            x_pbest = population[chosen_pbest_indices]
            
            # Select r1 and r2 (Difference vectors)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Simple check to reduce r1==r2 cases (improves difference vector quality)
            collision_mask = (r1 == r2)
            if np.any(collision_mask):
                r2[collision_mask] = np.random.randint(0, pop_size, np.sum(collision_mask))
            
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Calculate Mutant Vector
            # Reshape F for broadcasting: (pop_size, 1)
            F_matrix = F_trial[:, np.newaxis]
            mutant = population + F_matrix * (x_pbest - population) + F_matrix * (x_r1 - x_r2)
            
            # --- 3. Crossover (Binomial) ---
            CR_matrix = CR_trial[:, np.newaxis]
            cross_mask = np.random.rand(pop_size, dim) < CR_matrix
            
            # Ensure at least one parameter comes from mutant (DE standard)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_population = np.where(cross_mask, mutant, population)
            
            # --- 4. Boundary Constraints ---
            trial_population = np.clip(trial_population, lower_bound, upper_bound)
            
            # --- 5. Selection ---
            # Evaluate trial solutions
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return global_best_fitness
                
                trial_val = func(trial_population[i])
                
                # Greedy selection
                if trial_val <= fitnesses[i]:
                    fitnesses[i] = trial_val
                    population[i] = trial_population[i]
                    
                    # Update adaptive parameters on success
                    F[i] = F_trial[i]
                    CR[i] = CR_trial[i]
                    
                    if trial_val < global_best_fitness:
                        global_best_fitness = trial_val

            # --- 6. Convergence / Restart Check ---
            current_best = np.min(fitnesses)
            
            # Check for stagnation (best value didn't improve significantly)
            if np.abs(current_best - gen_best_fitness) < 1e-8:
                stagnation_counter += 1
            else:
                gen_best_fitness = current_best
                stagnation_counter = 0
            
            # Restart triggers:
            # 1. Low variance (population converged)
            # 2. Long stagnation (stuck in local minima)
            pop_std = np.std(fitnesses)
            if pop_std < 1e-6 or stagnation_counter > 30:
                break

    return global_best_fitness
