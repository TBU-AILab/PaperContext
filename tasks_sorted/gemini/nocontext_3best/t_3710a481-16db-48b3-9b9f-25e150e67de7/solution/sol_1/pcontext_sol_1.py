#Here is the Python code implementing a Self-Adaptive Differential Evolution (jDE) algorithm with a restart strategy. This approach improves upon standard Differential Evolution by automatically tuning parameters and restarting the population if convergence to a local optimum is detected.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 10 * dim is a standard robust choice for DE.
    # We ensure a minimum of 20 to maintain diversity.
    pop_size = int(max(20, 10 * dim))
    
    # Pre-process bounds for efficient vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # Initialize Population: Shape (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Fitness array to store score of each individual
    fitness = np.full(pop_size, float('inf'))
    
    # jDE Self-Adaptive Parameters
    # F (Mutation factor) and CR (Crossover rate) are stored per individual
    # Initial values: F=0.5, CR=0.9
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Track the global best solution
    best_val = float('inf')
    best_idx = 0
    
    # Evaluate Initial Population
    # We check time strictly inside the evaluation loop to handle expensive functions
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Optimization Loop ---
    while True:
        # Check time before generating the next batch
        if datetime.now() - start_time >= time_limit:
            return best_val

        # 1. Parameter Adaptation (jDE)
        # With probability tau1 (0.1), assign new random F in [0.1, 1.0]
        rand_f = np.random.rand(pop_size)
        mask_f = rand_f < 0.1
        F_new = F.copy()
        F_new[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
        
        # With probability tau2 (0.1), assign new random CR in [0.0, 1.0]
        rand_cr = np.random.rand(pop_size)
        mask_cr = rand_cr < 0.1
        CR_new = CR.copy()
        CR_new[mask_cr] = np.random.rand(np.sum(mask_cr))
        
        # 2. Mutation (DE/rand/1/bin)
        # Generate random indices (r1, r2, r3) for the whole population
        # We use random choice with replacement for speed (acceptable for DE)
        idxs = np.random.randint(0, pop_size, (pop_size, 3))
        r1, r2, r3 = idxs[:, 0], idxs[:, 1], idxs[:, 2]
        
        # Vectorized mutation: V = X_r1 + F * (X_r2 - X_r3)
        # Using the adapted F_new values
        mutant = pop[r1] + F_new[:, None] * (pop[r2] - pop[r3])
        
        # 3. Crossover
        rand_cross = np.random.rand(pop_size, dim)
        cross_mask = rand_cross < CR_new[:, None]
        
        # Ensure at least one parameter comes from mutant to avoid duplication
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        # Create trial vectors
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 4. Boundary Handling (Clip)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 5. Selection & Evaluation
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            # Greedy selection: if trial is better or equal, replace target
            if f_trial <= fitness[i]:
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                # Update parameters only if offspring is successful
                F[i] = F_new[i]
                CR[i] = CR_new[i]
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i
        
        # 6. Restart Strategy
        # If population diversity (std dev of fitness) is too low, we are likely stuck in a local optimum.
        # We trigger a restart: keep the best solution, randomize the rest.
        # We skip this check if fitness contains 'inf' (e.g., just after a restart).
        if not np.isinf(fitness).any():
            std_dev = np.std(fitness)
            if std_dev < 1e-6:
                # Save the best vector found so far
                saved_best = pop[best_idx].copy()
                
                # Re-initialize the population randomly
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                
                # Restore the best solution to index 0
                pop[0] = saved_best
                
                # Reset fitness to inf (forces re-evaluation/replacement in next generation)
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_val
                
                # Reset adaptive parameters
                F = np.full(pop_size, 0.5)
                CR = np.full(pop_size, 0.9)
                
                # Update best index
                best_idx = 0

    return best_val
