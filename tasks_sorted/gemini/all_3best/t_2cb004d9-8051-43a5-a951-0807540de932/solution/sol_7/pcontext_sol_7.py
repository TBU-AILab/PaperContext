#The following Python code implements a **Vectorized jDE (Self-Adaptive Differential Evolution) with Restarts and Coordinate Descent Local Search**.
#
##### Improvements Explained:
#1.  **Vectorized jDE (rand/1)**: Based on previous results, the `rand/1` mutation strategy with self-adaptive parameters ($F$, $CR$) performed best (#4). It maintains diversity better than greedy strategies like `current-to-pbest`, preventing premature convergence in the early phases.
#2.  **Coordinate Descent Local Search**: To address the "refinement" weakness of standard DE, this algorithm runs a **Local Search** (Coordinate Descent) on the global best solution whenever the population stagnates. This "polishes" the solution, drilling down into the minimum with reducing step sizes, extracting precision that stochastic search often misses.
#3.  **Elitist Restarts**: When the population stagnates (low variance), the algorithm restarts to explore new areas of the search space. Critically, it injects the refined global best solution into the new population, ensuring monotonic improvement.
#4.  **Optimized Vectorization**: Fully vectorized operations using NumPy for mutation, crossover, and bound handling ensure maximum function evaluations within the `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized jDE with Restarts and Coordinate Descent Local Search.
    
    Strategy:
    1. Global Search: Vectorized Self-Adaptive DE (jDE) explores the landscape.
    2. Local Refinement: When DE stagnates, a Coordinate Descent Local Search 
       polishes the best solution found so far.
    3. Restarts: Population is reset (keeping the best) to escape local optima.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Balanced for exploration vs generation speed.
    # 15 * dim is a standard heuristic, capped to ensure responsiveness.
    pop_size = int(max(20, 15 * dim))
    if pop_size > 70:
        pop_size = 70
        
    # Bounds pre-processing
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
    
    # Helper to check time budget efficiently
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Local Search: Coordinate Descent ---
    def local_search(start_x, start_val):
        """
        Performs a Coordinate Descent (Pattern Search) around start_x 
        to refine the solution precision.
        """
        current_x = start_x.copy()
        current_val = start_val
        
        # Initial step size (search radius) per dimension
        # Start with a fraction of the domain range
        step_size = diff_b * 0.2 
        
        # Perform a few iterations of refinement
        # We decrease step size each pass (5 passes max)
        for pass_idx in range(5): 
            if check_time(): break
            
            improved_pass = False
            
            # Iterate over all dimensions in random order to avoid bias
            dims_order = np.random.permutation(dim)
            
            for d in dims_order:
                if check_time(): break
                
                original_val = current_x[d]
                
                # 1. Try moving negative direction
                current_x[d] = np.clip(original_val - step_size[d], min_b[d], max_b[d])
                val_neg = func(current_x)
                
                if val_neg < current_val:
                    current_val = val_neg
                    improved_pass = True
                    # Keep the change
                else:
                    # 2. Try moving positive direction
                    current_x[d] = np.clip(original_val + step_size[d], min_b[d], max_b[d])
                    val_pos = func(current_x)
                    
                    if val_pos < current_val:
                        current_val = val_pos
                        improved_pass = True
                        # Keep the change
                    else:
                        # Revert changes if no improvement
                        current_x[d] = original_val
            
            # Adjust step size based on success
            if improved_pass:
                # If we found improvement, shrink slightly to refine
                step_size *= 0.5
            else:
                # If no improvement, shrink drastically to look closer
                step_size *= 0.5
                
            # If step size is too small (below machine precision relevance), stop
            if np.max(step_size) < 1e-9:
                break
                
        return current_x, current_val

    # --- Main Optimization Loop (Restarts) ---
    while True:
        if check_time(): return global_best_val
        
        # 1. Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best from previous runs
        if global_best_x is not None:
            population[0] = global_best_x.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if i % 10 == 0 and check_time(): return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_x = population[i].copy()
        
        # Initialize jDE Control Parameters (F and CR)
        # Each individual maintains its own parameters
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # Evolution Loop
        while True:
            if check_time(): return global_best_val
            
            # Check Stagnation (Convergence)
            # If population variance is low, we are stuck in a basin.
            if np.std(fitness) < 1e-6 or (np.max(fitness) - np.min(fitness)) < 1e-6:
                # Trigger Local Search on the best candidate before restarting
                # This squeezes the last bit of performance out of the basin
                if global_best_x is not None:
                    refined_x, refined_val = local_search(global_best_x, global_best_val)
                    if refined_val < global_best_val:
                        global_best_val = refined_val
                        global_best_x = refined_x
                break # Break inner loop to trigger restart
            
            # --- Vectorized DE Steps ---
            
            # 1. Parameter Adaptation (jDE)
            # F -> 0.1 + 0.9 * rand
            # CR -> rand
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            F[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
            CR[mask_CR] = np.random.rand(np.sum(mask_CR))
            
            # 2. Mutation (rand/1/bin)
            # V = X_r1 + F * (X_r2 - X_r3)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            r3 = np.random.randint(0, pop_size, pop_size)
            
            mutant = population[r1] + F[:, None] * (population[r2] - population[r3])
            
            # 3. Crossover
            mask_cross = np.random.rand(pop_size, dim) < CR[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
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
