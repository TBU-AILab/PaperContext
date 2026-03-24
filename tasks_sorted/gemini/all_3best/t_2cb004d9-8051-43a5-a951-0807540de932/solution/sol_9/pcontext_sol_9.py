#The following Python code implements a **Vectorized jDE (Self-Adaptive Differential Evolution) with Opposition-Based Learning (OBL) Initialization and Robust Pattern Search**.
#
#### Improvements Explained:
#
#1.  **Robust Pattern Search (Coordinate Descent)**: The previous best algorithm (Code 2) aggressively reduced the step size even after finding an improvement. This prevented the local search from effectively "walking" down long valleys or slopes. This improved version utilizes a robust Pattern Search strategy that **maintains the step size** upon successful moves, allowing the algorithm to traverse the landscape efficiently until a local minimum is bracketed, and only then refines (shrinks) the search radius.
#2.  **Opposition-Based Learning (OBL) Initialization**: Instead of simple random initialization, the algorithm generates a random population *and* its opposite population ($x' = min + max - x$) within the bounds. It evaluates both sets and selects the fittest half to form the initial population. This simple mathematical trick improves the probability of starting in a promising basin of attraction.
#3.  **Refined Stagnation Detection**: The algorithm strictly monitors population variance. Upon stagnation, it triggers the Local Search to extract maximum precision from the current basin, then restarts the population to explore new areas, injecting the polished global best to ensure monotonic improvement (Elitism).
#4.  **Vectorized jDE (`rand/1`)**: The core global search engine remains the vectorized Self-Adaptive DE (`rand/1/bin`), which has proven to be the most robust strategy for this problem based on previous iterations, balancing diversity and convergence speed better than greedy strategies.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized jDE with OBL Initialization 
    and Robust Pattern Search (Coordinate Descent).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size logic
    # Use 15 * dim to ensure enough diversity for exploration.
    # Cap at 70 to allow for high generation count within the time limit.
    pop_size = int(max(20, 15 * dim))
    if pop_size > 70:
        pop_size = 70
        
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracker
    global_best_val = float('inf')
    global_best_x = None
    
    # jDE Constants (Adaptive Parameter Probabilities)
    tau_F = 0.1
    tau_CR = 0.1
    
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Local Search: Robust Pattern Search ---
    def local_search(start_x, start_val):
        """
        Performs a robust Pattern Search (Coordinate Descent).
        Unlike previous versions, this maintains step size on success 
        to traverse valleys efficiently before refining.
        """
        current_x = start_x.copy()
        current_val = start_val
        
        # Initial step size per dimension (Starts at 5% of domain)
        step_sizes = diff_b * 0.05
        min_step = 1e-9
        
        # Limit iterations to prevent time drainage
        max_iter = 100 
        
        for _ in range(max_iter):
            if check_time(): break
            
            # If step sizes are too small, we have converged locally
            if np.max(step_sizes) < min_step:
                break
            
            improved_pass = False
            # Randomize order to avoid bias
            dims_order = np.random.permutation(dim)
            
            for d in dims_order:
                if check_time(): break
                
                original_val_d = current_x[d]
                
                # 1. Try moving in negative direction
                move_neg = np.clip(original_val_d - step_sizes[d], min_b[d], max_b[d])
                current_x[d] = move_neg
                val_neg = func(current_x)
                
                if val_neg < current_val:
                    current_val = val_neg
                    improved_pass = True
                    # Keep the change
                else:
                    # 2. Try moving in positive direction
                    move_pos = np.clip(original_val_d + step_sizes[d], min_b[d], max_b[d])
                    current_x[d] = move_pos
                    val_pos = func(current_x)
                    
                    if val_pos < current_val:
                        current_val = val_pos
                        improved_pass = True
                        # Keep the change
                    else:
                        # Revert if no improvement in either direction
                        current_x[d] = original_val_d
            
            # Step size management
            if improved_pass:
                # If we improved, keep the step size to continue "walking" down the slope
                # This traverses valleys much faster than shrinking immediately
                pass
            else:
                # Only if NO improvement was found in any dimension, shrink the search radius
                step_sizes *= 0.5
                
        return current_x, current_val

    # --- Main Optimization Loop (Restarts) ---
    while True:
        if check_time(): return global_best_val
        
        # 1. OBL Initialization (Opposition-Based Learning)
        # Generate random population
        pop_rand = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Generate opposite population
        pop_opp = min_b + max_b - pop_rand
        pop_opp = np.clip(pop_opp, min_b, max_b)
        
        # Combine populations
        combined_pop = np.vstack((pop_rand, pop_opp))
        combined_fitness = np.full(2 * pop_size, float('inf'))
        
        # Evaluate all candidates
        for i in range(2 * pop_size):
            # Batch time check
            if i % 10 == 0 and check_time(): return global_best_val
            
            val = func(combined_pop[i])
            combined_fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_x = combined_pop[i].copy()
                
        # Select the best N individuals to start the DE
        best_indices = np.argsort(combined_fitness)[:pop_size]
        population = combined_pop[best_indices]
        fitness = combined_fitness[best_indices]
        
        # Elitism: Inject global best from previous restarts if it's better
        if global_best_x is not None and global_best_val < fitness[0]:
            population[0] = global_best_x.copy()
            fitness[0] = global_best_val
            
        # Initialize jDE Parameters (F and CR)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # Evolution Loop
        while True:
            if check_time(): return global_best_val
            
            # Check Stagnation (Convergence)
            # If population variance is negligible, we are stuck in a basin.
            if np.std(fitness) < 1e-6 or (np.max(fitness) - np.min(fitness)) < 1e-6:
                # Trigger Local Search on the best candidate to polish it
                if global_best_x is not None:
                    refined_x, refined_val = local_search(global_best_x, global_best_val)
                    if refined_val < global_best_val:
                        global_best_val = refined_val
                        global_best_x = refined_x
                # Break inner loop to Trigger Restart
                break
                
            # --- Vectorized DE Generation ---
            
            # 1. Update Parameters (jDE Logic)
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            F[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F)) # F in [0.1, 1.0]
            CR[mask_CR] = np.random.rand(np.sum(mask_CR))          # CR in [0.0, 1.0]
            
            # 2. Mutation (rand/1/bin)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            r3 = np.random.randint(0, pop_size, pop_size)
            
            mutant = population[r1] + F[:, None] * (population[r2] - population[r3])
            
            # 3. Crossover (Binomial)
            mask_cross = np.random.rand(pop_size, dim) < CR[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            # Ensure at least one dimension comes from mutant
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask_cross, mutant, population)
            
            # 4. Bound Handling (Reflection)
            # Reflect out-of-bound values back into the domain
            mask_l = trial_pop < min_b
            if np.any(mask_l):
                trial_pop[mask_l] = 2 * min_b[np.where(mask_l)[1]] - trial_pop[mask_l]
            mask_u = trial_pop > max_b
            if np.any(mask_u):
                trial_pop[mask_u] = 2 * max_b[np.where(mask_u)[1]] - trial_pop[mask_u]
                
            # Final safety clip
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 5. Selection
            for i in range(pop_size):
                if i % 10 == 0 and check_time(): return global_best_val
                
                t_val = func(trial_pop[i])
                
                if t_val <= fitness[i]:
                    population[i] = trial_pop[i]
                    fitness[i] = t_val
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_x = trial_pop[i].copy()
                        
    return global_best_val
