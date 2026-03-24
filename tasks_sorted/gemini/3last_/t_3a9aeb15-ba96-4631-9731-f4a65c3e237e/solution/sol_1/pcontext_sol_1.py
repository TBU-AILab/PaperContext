#The following code implements a **Self-Adaptive Differential Evolution (jDE)** algorithm with a **Restart Mechanism**. This approach is generally more robust than standard Differential Evolution for finding the global minimum within a time limit, as it adapts its parameters ($F$ and $CR$) during execution and restarts the population if convergence stagnates (escaping local optima).
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Self-Adaptive Differential Evolution (jDE) with Restart Mechanism.
    
    Algorithmic improvements over standard DE:
    1. Self-Adaptive Parameters: F and CR evolve with individuals, removing the need for manual tuning.
    2. Mutation Strategy: Uses 'current-to-pbest/1' to balance exploration (param adaptation) and exploitation (greediness).
    3. Restart Mechanism: Re-initializes population (keeping the best) if diversity collapses, preventing stagnation in local optima.
    """
    
    # Start the timer
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Compromise between diversity and speed. 
    # Adaptive to dimension but clamped to [30, 100] to ensure iteration frequency.
    pop_size = int(np.clip(10 * dim, 30, 100))
    
    # jDE Adaptation Probabilities
    tau_F = 0.1
    tau_CR = 0.1
    
    # Convergence tolerance for restart (standard deviation of fitness)
    restart_tol = 1e-6

    # --- Initialization ---
    # Prepare bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize Adaptive Parameters (F and CR) for each individual
    # F starts around 0.5, CR around 0.9
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Fitness array
    fitness = np.full(pop_size, np.inf)
    
    # Track global best
    best_idx = -1
    best_val = np.inf
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Strict time check
        if (time.time() - start_time) >= max_time:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Optimization Loop ---
    while True:
        # Check time at the start of each generation
        if (time.time() - start_time) >= max_time:
            return best_val
            
        # 1. Restart Check
        # If population diversity is too low (collapsed), restart to find new basins.
        if np.std(fitness) < restart_tol:
            # Preserve the single best individual
            best_vec = pop[best_idx].copy()
            
            # Re-initialize the rest of the population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Reset parameters
            F = np.full(pop_size, 0.5)
            CR = np.full(pop_size, 0.9)
            
            # Restore best to index 0
            pop[0] = best_vec
            fitness[:] = np.inf
            fitness[0] = best_val
            best_idx = 0
            
            # Re-evaluate new individuals (skipping index 0)
            for k in range(1, pop_size):
                if (time.time() - start_time) >= max_time:
                    return best_val
                val = func(pop[k])
                fitness[k] = val
                if val < best_val:
                    best_val = val
                    best_idx = k
            
            # Continue to next iteration immediately
            continue

        # 2. Sort Population
        # Indices sorted by fitness (best to worst) for 'current-to-pbest' mutation
        sorted_indices = np.argsort(fitness)
        
        # Define p-best pool size (top 10% of population, minimum 2)
        num_pbest = max(2, int(pop_size * 0.1))
        
        # 3. Evolution Cycle
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
            
            target = pop[i]
            
            # -- Parameter Adaptation (jDE) --
            # Each individual generates candidate F and CR values
            new_F = F[i]
            new_CR = CR[i]
            
            if np.random.rand() < tau_F:
                new_F = 0.1 + 0.9 * np.random.rand() # F in [0.1, 1.0]
            if np.random.rand() < tau_CR:
                new_CR = np.random.rand() # CR in [0.0, 1.0]
            
            # -- Mutation: DE/current-to-pbest/1 --
            # Select a random individual from the top p% best
            p_idx = sorted_indices[np.random.randint(num_pbest)]
            x_pbest = pop[p_idx]
            
            # Select two random distinct individuals r1, r2
            r1 = np.random.randint(pop_size)
            r2 = np.random.randint(pop_size)
            while r1 == r2:
                r2 = np.random.randint(pop_size)
            
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Mutation vector
            # V = X_i + F*(X_pbest - X_i) + F*(X_r1 - X_r2)
            mutant = target + new_F * (x_pbest - target) + new_F * (x_r1 - x_r2)
            
            # -- Crossover --
            # Binary crossover
            cross_points = np.random.rand(dim) < new_CR
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            
            trial = np.where(cross_points, mutant, target)
            
            # -- Bound Handling --
            trial = np.clip(trial, min_b, max_b)
            
            # -- Selection --
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Accept trial vector
                pop[i] = trial
                fitness[i] = f_trial
                
                # Update individual's adaptive parameters on success
                F[i] = new_F
                CR[i] = new_CR
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i

    return best_val
