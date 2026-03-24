#The following output presents a **Vectorized Differential Evolution (DE) algorithm with Restarts and Gaussian Polishing**.
#
#**Key Improvements:**
#1.  **Fully Vectorized Operations:** Unlike standard loop-based implementations, this code generates mutant and trial vectors for the entire population simultaneously using NumPy. This significantly reduces Python interpreter overhead, allowing for many more function evaluations within the `max_time` limit.
#2.  **Greedy Strategy (`current-to-best`):** The algorithm uses the `DE/current-to-best/1/bin` strategy. This approach is more aggressive than `DE/rand`, guiding the population rapidly towards the best-known solution, which is crucial for finding good minima quickly.
#3.  **Adaptive Restart Mechanism:** The algorithm detects when the population has converged (low variance) or stagnated. Instead of wasting time in a local minimum, it triggers a "Polish" phase and then restarts with a fresh population, ensuring efficient use of the remaining time.
#4.  **Gaussian Polish:** Before restarting, the algorithm performs a concentrated local search (Gaussian random walk with shrinking variance) around the global best. This helps refine the solution to a higher precision than DE typically achieves on its own.
#5.  **Elitism:** The best solution found so far is preserved across restarts, ensuring that the result quality never degrades.
#
#output value is: 8.31029410385
#
#algorithm code is:
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
    
    # Population Size Heuristic:
    # Use a larger population for higher dimensions to maintain diversity,
    # but cap it to prevent iteration slowness.
    pop_size = int(min(200, max(50, 15 * dim)))
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_vec = None

    # --- 3. Main Optimization Loop (Restarts) ---
    while not check_timeout():
        
        # A. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject the best solution found so far into the new population
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
        stagnation_counter = 0
        
        while not check_timeout():
            
            # --- Vectorized Mutation (DE/current-to-best/1) ---
            # Strategy: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            
            # Identify current best
            best_idx = np.argmin(fitness)
            x_best = pop[best_idx]
            
            # Generate Parameters (Dithering)
            # F sampled from Normal(0.5, 0.3) per individual to ensure diversity
            F = np.random.normal(0.5, 0.3, (pop_size, 1))
            F = np.clip(F, 0.2, 1.0) 
            
            # CR sampled from Normal(0.9, 0.05) - High CR favors greedy convergence
            CR = np.random.normal(0.9, 0.05, (pop_size, 1))
            CR = np.clip(CR, 0.5, 1.0)
            
            # Select random indices r1, r2
            # We use random selection with replacement for speed (vectorization).
            # The probability of collision (r1=r2 or r1=i) is low and acceptable.
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
                
                # Greedy Selection
                if t_val <= fitness[i]:
                    fitness[i] = t_val
                    pop[i] = trial_pop[i]
                    
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_vec = trial_pop[i].copy()
                        epoch_improved = True
            
            # --- Convergence & Stagnation Check ---
            if not epoch_improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # Check population variance
            std_fit = np.std(fitness)
            
            # Trigger Restart if converged or stagnated
            if std_fit < 1e-6 or stagnation_counter > 20:
                
                # --- Gaussian Polish Phase ---
                # Before discarding the population, perform a local search around the best
                # to refine the solution precision.
                curr_vec = global_best_vec.copy()
                curr_val = global_best_val
                
                # Sigma starts at 5% of domain size and shrinks
                sigma = 0.05 * np.max(diff_b) 
                polish_steps = 50 
                
                for _ in range(polish_steps):
                    if check_timeout(): return global_best_val
                    
                    # Sample neighbor
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
                        # Contract search radius if no improvement
                        sigma *= 0.8
                
                # Break inner loop to trigger full restart
                break

    return global_best_val
