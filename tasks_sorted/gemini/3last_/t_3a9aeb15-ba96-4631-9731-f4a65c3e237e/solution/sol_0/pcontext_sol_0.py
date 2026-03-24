#Here is a fully operational, self-contained implementation using the Differential Evolution algorithm. This method is generally more efficient and robust than random search for finding global minima within bound constraints.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    # Start the timer immediately
    start_time = time.time()
    
    # ---------------------------------------------------------
    # Algorithm: Differential Evolution (DE)
    # Strategy: DE/rand/1/bin
    # ---------------------------------------------------------

    # --- Hyperparameters ---
    # Population size: 
    # A heuristic balance. Too small = premature convergence. 
    # Too large = slow iterations. We cap it to ensure the loop runs 
    # frequently enough for the time check.
    pop_size = int(np.clip(10 * dim, 20, 50)) 
    
    F = 0.8   # Differential weight [0, 2], typically 0.8
    CR = 0.9  # Crossover probability [0, 1], typically 0.9

    # --- Initialization ---
    
    # Convert bounds to numpy array for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = np.fabs(max_b - min_b)
    
    # Initialize Population (pop_size x dim)
    # Randomly distributed within the specified bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Array to store fitness values
    fitness = np.full(pop_size, np.inf)
    
    # Track the global best value found so far
    best_val = np.inf

    # --- Initial Evaluation ---
    # We evaluate individuals one by one to ensure we don't exceed 
    # max_time during the initialization phase if func is slow.
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # --- Main Optimization Loop ---
    while True:
        # Iterate over each individual in the population
        for i in range(pop_size):
            
            # Check time constraint strictly before every new evaluation
            if (time.time() - start_time) >= max_time:
                return best_val

            # 1. Mutation
            # Select 3 distinct random indices (a, b, c) distinct from current i
            idxs = [idx for idx in range(pop_size) if idx != i]
            # Random choice is safe here given pop_size is small/moderate
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]
            
            # Generate mutant vector
            mutant = a + F * (b - c)
            
            # Enforce bounds (clip values that jumped out of bounds)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover
            # Create a trial vector by mixing mutant and current parent (target)
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one dimension is taken from the mutant 
            # to guarantee the trial vector is different from the parent
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, pop[i])
            
            # 3. Selection
            f_trial = func(trial)
            
            # Greedy selection: if trial is better, replace the parent
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial

    return best_val
