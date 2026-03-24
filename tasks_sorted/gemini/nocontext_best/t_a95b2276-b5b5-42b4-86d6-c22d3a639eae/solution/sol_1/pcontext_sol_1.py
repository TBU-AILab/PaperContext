#Here is a fully operational implementation using a **Restart-Covariant Differential Evolution (DE)** strategy.
#
#### Algorithm Description
#This algorithm improves upon standard Differential Evolution by implementing:
#1.  **DE/current-to-best/1/bin Strategy**: A mutation scheme that directs candidate solutions toward the best found solution while maintaining population diversity. This generally converges faster than random search.
#2.  **Restart Mechanism**: If the population converges (variance drops) or fitness stagnates, the algorithm triggers a "Soft Restart." It reinjects the global best solution into a fresh random population, allowing the search to escape local optima without losing progress.
#3.  **Adaptive Parameters**: It dithers the scaling factor $F$ each generation to balance exploration and exploitation dynamically.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution with Restarts (DE/current-to-best/1).
    """
    t_start = time.time()
    
    # Helper to check strict time budget
    def time_is_up():
        # Buffer of 0.05s to ensure clean return
        return (time.time() - t_start) >= (max_time - 0.05)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population size logic: 
    # High dimensions need more individuals, but we cap at 100 to ensure 
    # enough generations run within the time limit.
    pop_size = int(15 * dim)
    pop_size = max(20, min(100, pop_size))
    
    global_best_val = float('inf')
    global_best_pos = None

    # --- Main Loop (Restarts) ---
    while not time_is_up():
        
        # 1. Initialize Population
        # Uniform random distribution within bounds
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Soft Restart / Elitism: 
        # If we already found a good solution, keep it in the new population 
        # to guide the search (exploitation) while the rest explores.
        if global_best_pos is not None:
            pop[0] = global_best_pos
            # Add slightly perturbed copies of the best to refine local area
            for k in range(1, 3): 
                if k < pop_size:
                    pert = global_best_pos + np.random.normal(0, 0.01, dim) * diff_b
                    pop[k] = np.clip(pert, min_b, max_b)

        fitness = np.full(pop_size, float('inf'))
        
        # 2. Evaluate Initial Population
        for i in range(pop_size):
            if time_is_up(): return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_pos = pop[i].copy()

        # Track best index in current population
        current_best_idx = np.argmin(fitness)
        prev_gen_best_val = fitness[current_best_idx]
        stall_count = 0

        # --- DE Optimization Loop ---
        while True:
            if time_is_up(): return global_best_val
            
            # 3. Parameters
            # F (Mutation Factor): Dithered between 0.5 and 0.9 to prevent stagnation
            F = 0.5 + 0.4 * np.random.rand()
            # CR (Crossover Rate): High probability to inherit structure from mutant
            CR = 0.9
            
            # 4. Mutation: DE/current-to-best/1
            # V = X_current + F * (X_best - X_current) + F * (X_r1 - X_r2)
            # This vector directs the search towards the best solution found so far.
            
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            x_best = pop[current_best_idx]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Vectorized mutation calculation
            mutant = pop + F * (x_best - pop) + F * (x_r1 - x_r2)
            
            # 5. Crossover
            rand_mat = np.random.rand(pop_size, dim)
            mask = rand_mat < CR
            
            # Construct trial population
            trial_pop = np.where(mask, mutant, pop)
            
            # 6. Bound Constraints
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 7. Selection
            # Evaluate trials and replace if better
            for i in range(pop_size):
                if time_is_up(): return global_best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_pos = trial_pop[i].copy()
            
            # 8. Check for Restart Conditions
            current_best_idx = np.argmin(fitness)
            current_gen_best_val = fitness[current_best_idx]
            
            # Calculate population variance to detect convergence
            if np.std(fitness) < 1e-8:
                break # Population converged, trigger restart
            
            # Check for stagnation (no improvement over generations)
            if current_gen_best_val < prev_gen_best_val:
                prev_gen_best_val = current_gen_best_val
                stall_count = 0
            else:
                stall_count += 1
                
            if stall_count > 30: # Stalled for too long, trigger restart
                break

    return global_best_val
