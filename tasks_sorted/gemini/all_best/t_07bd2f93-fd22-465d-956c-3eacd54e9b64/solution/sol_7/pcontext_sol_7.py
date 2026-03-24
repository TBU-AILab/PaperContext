#The following output presents an optimized **Vectorized Differential Evolution (DE)** algorithm.
#
#**Key Improvements:**
#1.  **Vectorized Operations**: Unlike the previous best algorithm (which was loop-based), this implementation utilizes NumPy's vectorization for mutation and crossover. This significantly reduces interpreter overhead, allowing for more generations and function evaluations within the limited `max_time`.
#2.  **Strategy (DE/current-to-best/1)**: It retains the successful greedy strategy of the previous best algorithm (`current-to-best`), which pulls the population rapidly towards the best-known solution.
#3.  **Restarts with Elitism**: To prevent stagnation in local minima (a risk with greedy strategies), the algorithm monitors population variance. If the population converges, it triggers a restart, initializing a fresh population but injecting the global best solution (Elitism) to ensure monotonic improvement.
#4.  **Gaussian Polish**: Before restarting, the algorithm executes a lightweight local search (Gaussian walk with shrinking variance) around the global best. This exploits the remaining time to refine the solution precision, often achieving the final few decimal points of improvement that global searchers miss.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Differential Evolution (DE)
    with 'current-to-best' strategy, Restarts, and Gaussian Polishing.
    """
    # --- 1. Setup & Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- 2. Initialization Helpers ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size Heuristic
    # A bit larger than minimal to utilize vectorization speed while ensuring diversity
    pop_size = max(30, 15 * dim)
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_vec = None

    # --- 3. Main Optimization Loop (Restarts) ---
    while not check_timeout():
        
        # A. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject the best solution found so far into the new population
        # This allows the new population to explore around the best known area immediately
        if global_best_vec is not None:
            pop[0] = global_best_vec
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_timeout(): return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
        
        # B. Evolutionary Cycle
        # Continue until timeout or convergence
        while not check_timeout():
            
            # --- Vectorized Mutation (DE/current-to-best/1) ---
            # Strategy: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            
            # Identify current best in population
            best_idx = np.argmin(fitness)
            x_best = pop[best_idx]
            
            # Generate Parameters
            # F: Uniform random [0.5, 0.9] per individual to maintain diversity
            F = np.random.uniform(0.5, 0.9, (pop_size, 1))
            
            # CR: Fixed high crossover rate 0.9 to preserve good genetic blocks
            CR = 0.9
            
            # Select random indices r1, r2
            # We use random selection with replacement for vectorization speed.
            # Collisions have negligible impact on performance for standard pop sizes.
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Compute Mutant Vectors (Vectorized)
            # Broadcasting x_best to match pop shape
            mutant = pop + F * (x_best - pop) + F * (pop[r1] - pop[r2])
            
            # --- Vectorized Crossover (Binomial) ---
            rand_uniform = np.random.rand(pop_size, dim)
            mask = rand_uniform < CR
            
            # Ensure at least one parameter is mutated for each individual
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            # Create Trial Population
            trial_pop = np.where(mask, mutant, pop)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection & Evaluation ---
            epoch_improved = False
            
            for i in range(pop_size):
                if check_timeout(): return global_best_val
                
                t_val = func(trial_pop[i])
                
                # Greedy Selection: Child replaces parent if better or equal
                if t_val <= fitness[i]:
                    fitness[i] = t_val
                    pop[i] = trial_pop[i]
                    
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_vec = trial_pop[i].copy()
                        epoch_improved = True
            
            # --- Convergence & Stagnation Check ---
            # If standard deviation of fitness is very low, the population has converged.
            std_fit = np.std(fitness)
            
            if std_fit < 1e-6:
                
                # --- Gaussian Polish Phase ---
                # Before restarting, perform a local search around the global best
                # to squeeze out final precision.
                if global_best_vec is not None:
                    curr_vec = global_best_vec.copy()
                    curr_val = global_best_val
                    
                    # Search radius starts at 5% of domain size and shrinks
                    sigma = np.max(diff_b) * 0.05 
                    polish_steps = 50 
                    
                    for _ in range(polish_steps):
                        if check_timeout(): return global_best_val
                        
                        perturb = np.random.normal(0, sigma, dim)
                        neighbor = np.clip(curr_vec + perturb, min_b, max_b)
                        n_val = func(neighbor)
                        
                        if n_val < curr_val:
                            curr_val = n_val
                            curr_vec = neighbor
                            if n_val < global_best_val:
                                global_best_val = n_val
                                global_best_vec = neighbor.copy()
                        else:
                            # Decay sigma if no improvement
                            sigma *= 0.8
                            
                # Break inner loop to trigger full restart
                break

    return global_best_val
