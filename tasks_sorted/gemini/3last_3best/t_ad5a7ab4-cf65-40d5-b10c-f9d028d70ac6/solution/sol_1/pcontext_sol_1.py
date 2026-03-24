#The following algorithm implements a **Restarted "Current-to-Best" Differential Evolution (DE)** strategy.
#
#### Algorithm Description
#This approach improves upon standard Differential Evolution by incorporating three key mechanisms to minimize the output value more effectively within the time limit:
#
#1.  **"Current-to-Best" Mutation Strategy**: Instead of random exploration, the population is guided towards the best solution found so far while maintaining diversity through random perturbations ($v = x + F(x_{best} - x) + F(x_{r1} - x_{r2})$). This typically converges faster than standard DE.
#2.  **Adaptive Parameters (Dithering)**: The mutation factor $F$ and crossover rate $CR$ are randomized slightly in each generation to prevent the algorithm from getting stuck in search patterns and to improve robustness across different fitness landscapes.
#3.  **Restarts with Memory**: If the population converges (low variance) or stagnates (no improvement), the algorithm restarts with a fresh population but injects the global best solution found so far. This allows the algorithm to escape local optima and continually refine the best solution until time runs out.
#
#### Python Implementation
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarted Differential Evolution with 
    current-to-best mutation and adaptive parameters.
    """
    start_time = datetime.now()
    # Use 95% of max_time to ensure safe return before timeout
    time_limit = timedelta(seconds=max_time * 0.95)

    # --- Configuration ---
    # Population size: Balance between diversity and speed
    # Cap at 200 to ensure progress in high dimensions within short time limits
    pop_size = min(200, max(30, 10 * dim))
    
    # Pre-process bounds for efficient vectorized operations
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # Track global best
    best_val = float('inf')
    best_vec = None
    
    def check_timeout():
        return datetime.now() - start_time >= time_limit

    # --- Main Restart Loop ---
    # Runs continuously until time is almost up
    while True:
        if check_timeout(): return best_val
        
        # 1. Initialize Population
        pop = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
        fitness = np.full(pop_size, float('inf'))
        
        # 2. Evaluate Initial Population
        for i in range(pop_size):
            if check_timeout(): return best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_vec = pop[i].copy()
        
        # 3. Inject Global Best (Memory)
        # If we have a previous best, inject it into the new population
        # to focus the search around promising areas (exploitation).
        if best_vec is not None:
            pop[0] = best_vec
            fitness[0] = best_val

        # Variables for stagnation detection
        stall_count = 0
        current_best_fit = np.min(fitness)

        # --- Optimization Loop (Generations) ---
        while True:
            if check_timeout(): return best_val

            # 4. Adaptive Parameters (Dither)
            # Randomize F (0.5 to 0.9) and CR (0.8 to 1.0) to maintain diversity
            F = 0.5 + 0.4 * np.random.rand()
            CR = 0.8 + 0.2 * np.random.rand()

            # 5. Mutation: DE/current-to-best/1
            # Equation: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            
            # Find local best in current population
            idx_best = np.argmin(fitness)
            x_best = pop[idx_best]
            
            # Select random indices r1, r2
            # Note: Approximated vectorization (collisions rare enough to ignore)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Vectorized mutation step
            diff_to_best = x_best - pop
            diff_rand = pop[r1] - pop[r2]
            mutant = pop + F * diff_to_best + F * diff_rand
            
            # 6. Crossover
            # Generate mask
            mask = np.random.rand(pop_size, dim) < CR
            # Ensure at least one parameter is changed for every individual
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            # Create trial vectors
            trial = np.where(mask, mutant, pop)
            
            # 7. Bound Constraints (Clipping)
            trial = np.clip(trial, lower_bounds, upper_bounds)
            
            # 8. Selection
            # Evaluate trial vectors one by one
            gen_improved = False
            for i in range(pop_size):
                if check_timeout(): return best_val
                
                f_trial = func(trial[i])
                
                # Greedy selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    # Update global best
                    if f_trial < best_val:
                        best_val = f_trial
                        best_vec = trial[i].copy()
                        gen_improved = True

            # 9. Convergence Check
            # Check for stagnation (no improvement in best fitness)
            new_best_fit = np.min(fitness)
            if new_best_fit < current_best_fit:
                current_best_fit = new_best_fit
                stall_count = 0
            else:
                stall_count += 1
            
            # Check for population convergence (variance close to zero)
            # If population has converged or stalled, break to trigger restart
            if np.var(fitness) < 1e-8 or stall_count > 30:
                break 

    return best_val
