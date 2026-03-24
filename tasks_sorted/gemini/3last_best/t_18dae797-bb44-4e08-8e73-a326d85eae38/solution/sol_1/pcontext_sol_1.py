#Here is an improved algorithm. It utilizes **Differential Evolution with a "Current-to-Best" mutation strategy and Adaptive Restart**.
#
#**Reasoning for Improvement:**
#1.  **Current-to-Best Strategy**: The previous algorithm used `DE/rand/1`, which explores well but converges slowly. This improved version uses `DE/current-to-best/1`, which incorporates the vector to the global best solution ($X_{best} - X_{current}$) into the mutation. This significantly speeds up convergence (exploitation) while still exploring via the standard difference vector.
#2.  **Dithering (Randomized F)**: Instead of a fixed Mutation Factor ($F$), we randomly sample $F$ for each generation. This helps the algorithm escape local minima and prevents stagnation.
#3.  **Restart Mechanism**: If the population converges (the spread of fitness values drops below a threshold), the algorithm effectively "restarts" by keeping the best solution and randomizing the rest of the population. This allows it to hunt for better minima in other areas of the search space if time permits.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using Differential Evolution with 
    'Current-to-Best' mutation, Dithering, and Population Restart.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Use 95% of max_time to ensure safe return
    time_limit = timedelta(seconds=max_time * 0.95)
    
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Initialization ---
    # Convert bounds to numpy for vector math
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Heuristic population size: 15 * dim (slightly higher for diversity)
    # Clamped between 20 and 150 to manage time complexity
    pop_size = int(max(20, min(150, 15 * dim)))
    
    # Parameters
    # CR: Crossover Probability
    CR = 0.9 
    # F will be dithered (randomized) during execution
    
    # Initialize Population
    # shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Best Solution Tracking
    best_idx = 0
    best_val = float('inf')
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if check_time():
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Loop ---
    while not check_time():
        
        # Check for Convergence / Stagnation
        # If the standard deviation of fitness is very low, the population has converged.
        # We perform a "Soft Restart": Keep the best, randomize the rest.
        if np.std(fitness) < 1e-6:
            # Preserve best
            best_vector = population[best_idx].copy()
            
            # Re-initialize rest
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_vector # Put best back at index 0
            
            # Re-evaluate
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            best_idx = 0
            
            # Re-eval loop (skipping index 0)
            for i in range(1, pop_size):
                if check_time(): return best_val
                val = func(population[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_idx = i
            
            # If we restarted, continue to next iteration of while loop
            continue

        # --- Generation Loop ---
        # We create a new generation array to store updates
        # Using a loop here allows us to handle the r1!=r2!=i logic strictly
        
        for i in range(pop_size):
            if check_time():
                return best_val
            
            # 1. Select Random Indices (r1, r2) != i
            # We select from the whole range excluding i
            idxs = [idx for idx in range(pop_size) if idx != i]
            # Fast random selection
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # 2. Mutation Strategy: DE/current-to-best/1
            # V = X_i + F * (X_best - X_i) + F * (X_r1 - X_r2)
            # This balances greediness (move to best) with exploration (diff vector)
            
            # Dithering: Randomize F per vector to prevent stagnation
            # Sample F from [0.5, 1.0]
            F = 0.5 + (np.random.rand() * 0.5)
            
            x_i = population[i]
            x_best = population[best_idx]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Compute Mutant Vector
            mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            # Generate mask: True where we take from mutant, False where we keep old
            cross_mask = np.random.rand(dim) < CR
            # Guaranteed parameter inheritance
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # 4. Bound Constraint Handling
            # Clip is the safest method to stay within bounds
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection (Greedy)
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i

    return best_val
