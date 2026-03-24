import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using Self-Adaptive Differential Evolution (jDE) 
    with a 'current-to-best/1/bin' mutation strategy and a Restart Mechanism.
    """
    # Initialize timer
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Preprocessing ---
    # Convert bounds to numpy arrays for efficient computation
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Configuration ---
    # Population size: clamped to ensure responsiveness on high dimensions
    # and sufficient diversity on low dimensions.
    pop_size = int(np.clip(10 * dim, 20, 70))
    
    # --- Initialization ---
    # Random initial population within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # jDE Adaptive Parameters (F and CR per individual)
    # Initialize F=0.5, CR=0.9
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Global Best Tracking
    best_val = float('inf')
    best_vec = None
    
    # Helper for time checking
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if check_timeout():
            # If timeout occurs during initialization, return best found or fallback
            return best_val if best_val != float('inf') else func(population[0])
            
        try:
            val = func(population[i])
        except Exception:
            val = float('inf')
            
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()
            
    # --- Main Loop ---
    # Variables for restart mechanism
    stag_limit = 30  # Generations allowed without significant improvement
    stag_count = 0
    last_best = best_val
    
    while not check_timeout():
        
        # 1. Restart Mechanism
        # Trigger if population diversity is lost (low std dev) or if stagnant
        pop_std = np.std(fitness)
        if pop_std < 1e-8 or stag_count > stag_limit:
            # Perform Restart: Keep best solution, randomize the rest
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_vec
            fitness[:] = float('inf')
            fitness[0] = best_val
            
            # Reset Adaptive Parameters
            F[:] = 0.5
            CR[:] = 0.9
            
            # Reset counters
            stag_count = 0
            last_best = best_val
            
            # Re-evaluate new population (skip index 0 which is best_vec)
            for i in range(1, pop_size):
                if check_timeout(): return best_val
                try:
                    val = func(population[i])
                except:
                    val = float('inf')
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = population[i].copy()
            continue

        # 2. Evolution Cycle (Iteration over population)
        for i in range(pop_size):
            if check_timeout(): return best_val
            
            # A. Parameter Adaptation (jDE)
            f_i = F[i]
            cr_i = CR[i]
            
            # Update F with probability 0.1
            if np.random.rand() < 0.1:
                f_i = 0.1 + 0.9 * np.random.rand()
                
            # Update CR with probability 0.1
            if np.random.rand() < 0.1:
                cr_i = np.random.rand()
            
            # B. Mutation: 'current-to-best/1'
            # Formula: V = X_i + F*(X_best - X_i) + F*(X_r1 - X_r2)
            # This strategy balances exploitation (to best) and exploration (diff).
            
            # Select distinct random indices r1, r2 != i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
                
            x_i = population[i]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            mutant = x_i + f_i * (best_vec - x_i) + f_i * (x_r1 - x_r2)
            
            # C. Crossover (Binomial)
            rand_vals = np.random.rand(dim)
            mask = rand_vals < cr_i
            # Ensure at least one parameter is changed
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, x_i)
            
            # D. Boundary Handling (Clip)
            trial = np.clip(trial, min_b, max_b)
            
            # E. Selection
            try:
                f_trial = func(trial)
            except:
                f_trial = float('inf')
                
            # Greedy selection
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                # Update adaptive parameters for this individual
                F[i] = f_i
                CR[i] = cr_i
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
        
        # 3. Check Stagnation
        if best_val < last_best - 1e-9:
            last_best = best_val
            stag_count = 0
        else:
            stag_count += 1
            
    return best_val
