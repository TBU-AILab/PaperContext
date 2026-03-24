#The output value of the last generated algorithm is: 12.61657793417723.
#The following improved algorithm implements a **Hybrid Differential Evolution (DE) with Restart and Coordinate Descent Polishing**.
#
#**Key Improvements:**
#1.  **Greedy Strategy (current-to-best/1/bin):** This mutation strategy was identified as the most effective in the previous iterations (Algorithm 1). It converges faster than random variants by utilizing the best solution found so far in the current population.
#2.  **Coordinate Descent "Polishing":** After the DE population converges (low variance) or stagnates, the algorithm switches to a **Pattern Search (Coordinate Descent)** on the global best solution. This deterministic local search refines the parameters with high precision, squeezing out the last bit of optimality that stochastic DE might miss.
#3.  **Restart Mechanism:** If the population converges and the polishing is done, the algorithm restarts with a fresh random population to explore other basins of attraction within the remaining time.
#4.  **Robust Time Management:** A custom `TimeOutException` is used to break out of nested loops (DE or Polish) immediately when `max_time` is exceeded, ensuring the best result is returned without overrunning the limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using a Hybrid approach: 
    Differential Evolution (current-to-best) for exploration 
    + Coordinate Descent for precise exploitation (Polishing),
    wrapped in a Restart mechanism.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    lower_bound = bounds_arr[:, 0]
    upper_bound = bounds_arr[:, 1]
    bound_diff = upper_bound - lower_bound
    
    # Container for the Global Best to be accessed by nested functions
    # 'pos' stores the input array, 'val' stores the output value
    best_holder = {'val': float('inf'), 'pos': None}
    
    # Custom exception for clean exit upon timeout
    class TimeOutException(Exception):
        pass

    def evaluate(p):
        """Wrapper to evaluate function, track global best, and check time."""
        # Check time budget
        if time.time() - start_time >= max_time:
            raise TimeOutException
        
        val = func(p)
        
        # Update global best
        if val < best_holder['val']:
            best_holder['val'] = val
            best_holder['pos'] = p.copy()
        return val

    def polish(start_pos, initial_step):
        """
        Coordinate Descent (Pattern Search) to refine a solution.
        Explores neighbors in each dimension to find a local minimum.
        """
        step = initial_step
        current_pos = start_pos.copy()
        current_val = best_holder['val']
        
        # Iteratively reduce step size until precision limit
        while step > 1e-9:
            improved = False
            # Randomize order of dimensions to avoid bias
            indices = np.random.permutation(dim)
            
            for i in indices:
                # Check time inside the loop
                if time.time() - start_time >= max_time:
                    raise TimeOutException
                
                original_val = current_pos[i]
                
                # Try moving in positive direction
                current_pos[i] = np.clip(original_val + step, lower_bound[i], upper_bound[i])
                val = evaluate(current_pos)
                
                if val < current_val:
                    current_val = val
                    improved = True
                    continue # Move accepted, go to next dimension
                
                # Try moving in negative direction
                current_pos[i] = np.clip(original_val - step, lower_bound[i], upper_bound[i])
                val = evaluate(current_pos)
                
                if val < current_val:
                    current_val = val
                    improved = True
                else:
                    # Revert if no improvement
                    current_pos[i] = original_val
            
            # If no improvement found in any direction with current step, reduce step size
            if not improved:
                step *= 0.5

    # --- Main Loop (Restarts) ---
    try:
        while True:
            # Stop if time is up
            if time.time() - start_time >= max_time:
                break
                
            # --- DE Configuration ---
            # Population size scaled by dimension, clamped for efficiency
            pop_size = int(max(20, min(100, 15 * dim)))
            
            # Initialize Population
            population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
            fitnesses = np.full(pop_size, float('inf'))
            
            # Initial evaluation
            for i in range(pop_size):
                fitnesses[i] = evaluate(population[i])
            
            # Track local population best
            best_idx = np.argmin(fitnesses)
            pop_best_val = fitnesses[best_idx]
            
            stagnation_counter = 0
            
            # --- DE Evolution Phase ---
            while True:
                # Dithered F for diversity [0.5, 1.0]
                F = 0.5 + 0.5 * np.random.rand()
                # High CR often works well for inseparable functions
                CR = 0.9
                
                # Vectorized Strategy: DE/current-to-best/1/bin
                # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
                
                r1 = np.random.randint(0, pop_size, pop_size)
                r2 = np.random.randint(0, pop_size, pop_size)
                
                x_best = population[best_idx] # Broadcasts current best
                x_r1 = population[r1]
                x_r2 = population[r2]
                
                # Mutation
                mutant = population + F * (x_best - population) + F * (x_r1 - x_r2)
                
                # Crossover
                rand_cr = np.random.rand(pop_size, dim)
                cross_mask = rand_cr < CR
                # Ensure at least one dimension is changed
                j_rand = np.random.randint(0, dim, pop_size)
                cross_mask[np.arange(pop_size), j_rand] = True
                
                trial_pop = np.where(cross_mask, mutant, population)
                trial_pop = np.clip(trial_pop, lower_bound, upper_bound)
                
                # Selection
                improved_in_gen = False
                for i in range(pop_size):
                    val = evaluate(trial_pop[i])
                    
                    if val <= fitnesses[i]:
                        fitnesses[i] = val
                        population[i] = trial_pop[i]
                        
                        if val < pop_best_val:
                            pop_best_val = val
                            best_idx = i
                            improved_in_gen = True
                
                # Check for Stagnation
                if not improved_in_gen:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                # Convergence Criteria for Restart
                # 1. Variance is very low (population converged)
                # 2. Stagnation for too many generations (stuck in local optimum)
                if np.std(fitnesses) < 1e-6 or stagnation_counter > 25:
                    break
            
            # --- Polish Phase ---
            # After DE converges, take the best found solution and refine it
            if best_holder['pos'] is not None:
                # Initial step: 5% of average domain width
                polish_step = np.mean(bound_diff) * 0.05
                polish(best_holder['pos'], polish_step)

    except TimeOutException:
        pass

    return best_holder['val']
