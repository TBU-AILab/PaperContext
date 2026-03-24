#The following Python code implements a **Vectorized jDE (Self-Adaptive Differential Evolution) with Restarts and Accelerated Coordinate Descent Local Search**.
#
###### Improvements Explained:
#1.  **Accelerated Coordinate Descent (Line Search)**: The previous best algorithm used a simple step-based pattern search. This version introduces an **accelerated line search**. When a descent direction is found for a variable, the algorithm accelerates movement in that direction (increasing step size geometrically: 1x, 2x, 4x...) until no further improvement is found. This allows the local search to traverse valleys significantly faster and deeper than fixed-step methods.
#2.  **Vectorized jDE (rand/1)**: Maintains the `rand/1` mutation strategy which proved most robust (#1 in the leaderboard), as it preserves population diversity better than greedy strategies, preventing premature convergence before the basin of attraction is identified.
#3.  **Boundary Reflection**: The Global Search (DE) uses boundary reflection to preserve the statistical distribution of the population near edges, while the Local Search uses clipping to ensure stability during aggressive line searches.
#4.  **Optimized Restarts**: The algorithm restarts upon population stagnation to explore new basins, but injects the polished global best solution to ensure monotonic improvement.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized jDE (rand/1) with Restarts 
    and Accelerated Coordinate Descent Local Search.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 
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

    # --- Local Search: Accelerated Coordinate Descent ---
    def local_search_accelerated(start_x, start_val):
        """
        Performs coordinate descent with line search acceleration.
        Scans each dimension, finds descent direction, and accelerates 
        step size to traverse the valley quickly.
        """
        current_x = start_x.copy()
        current_val = start_val
        
        # Initial step size per dimension (Starts at 5% of domain)
        # We allow this to shrink as we refine.
        step_sizes = diff_b * 0.05
        min_step = 1e-9
        
        # Passes over all dimensions
        # Limited to 5 passes to ensure we don't drain the time budget
        max_passes = 5
        
        for _ in range(max_passes):
            if check_time(): break
            
            improved_pass = False
            # Randomize order to avoid bias
            dims_order = np.random.permutation(dim)
            
            for d in dims_order:
                if check_time(): break
                
                # Check neighbors (Pattern Search)
                x_minus = current_x.copy()
                x_minus[d] = np.clip(current_x[d] - step_sizes[d], min_b[d], max_b[d])
                val_minus = func(x_minus)
                
                x_plus = current_x.copy()
                x_plus[d] = np.clip(current_x[d] + step_sizes[d], min_b[d], max_b[d])
                val_plus = func(x_plus)
                
                # Determine descent direction
                best_neighbor_val = current_val
                direction = 0
                
                if val_minus < best_neighbor_val:
                    best_neighbor_val = val_minus
                    direction = -1
                
                if val_plus < best_neighbor_val:
                    best_neighbor_val = val_plus
                    direction = 1
                
                # If improvement found, perform Accelerated Line Search in that direction
                if direction != 0:
                    improved_pass = True
                    
                    # Adopt the first step
                    if direction == -1:
                        current_x[d] = x_minus[d]
                        current_val = val_minus
                    else:
                        current_x[d] = x_plus[d]
                        current_val = val_plus
                        
                    # Line Search Loop: Accelerate (1x, 2x, 4x...)
                    acc = 2.0
                    while True:
                        if check_time(): break
                        
                        # Calculate accelerated step
                        move = direction * step_sizes[d] * acc
                        next_val_d = current_x[d] + move
                        
                        # Bound check: Break if we hit the wall
                        if next_val_d < min_b[d] or next_val_d > max_b[d]:
                            break
                        
                        # Evaluate
                        test_x = current_x.copy()
                        test_x[d] = next_val_d
                        test_val = func(test_x)
                        
                        if test_val < current_val:
                            # Keep moving
                            current_val = test_val
                            current_x[d] = next_val_d
                            acc *= 2.0 # Accelerate
                        else:
                            # Stop line search, previous point was better
                            break
                            
                else:
                    # No improvement in either direction, shrink step for this dim
                    step_sizes[d] *= 0.5
            
            # If a pass completed with no improvement, we are close to optimum or stuck
            if not improved_pass:
                # If steps are tiny, quit early
                if np.max(step_sizes) < min_step:
                    break
        
        return current_x, current_val

    # --- Main Optimization Loop (Restarts) ---
    while True:
        if check_time(): return global_best_val
        
        # 1. Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best from previous runs to maintain progress
        if global_best_x is not None:
            population[0] = global_best_x.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            # Periodically check time
            if i % 10 == 0 and check_time(): return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_x = population[i].copy()
                
        # Initialize jDE Adaptive Parameters
        # Each individual has its own F and CR
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # Evolution Loop
        while True:
            if check_time(): return global_best_val
            
            # Check Stagnation (Convergence)
            # If population variance is low, we are trapped in a basin.
            if np.std(fitness) < 1e-6 or (np.max(fitness) - np.min(fitness)) < 1e-6:
                # Trigger Local Search on the best candidate to polish it
                if global_best_x is not None:
                    ls_x, ls_val = local_search_accelerated(global_best_x, global_best_val)
                    if ls_val < global_best_val:
                        global_best_val = ls_val
                        global_best_x = ls_x
                # Break inner loop to Trigger Restart
                break
                
            # --- Vectorized DE Generation ---
            
            # 1. Update Parameters (jDE Logic)
            # Probabilistically update F and CR
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            F[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F)) # F in [0.1, 1.0]
            CR[mask_CR] = np.random.rand(np.sum(mask_CR))          # CR in [0.0, 1.0]
            
            # 2. Mutation (rand/1/bin)
            # Select random indices (vectorized)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            r3 = np.random.randint(0, pop_size, pop_size)
            
            # Compute difference vectors
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
