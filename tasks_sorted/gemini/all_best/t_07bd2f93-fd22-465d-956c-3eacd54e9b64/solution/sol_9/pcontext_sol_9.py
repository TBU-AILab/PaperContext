#The following output presents an optimized **Vectorized Differential Evolution (DE) algorithm with Restarts and Gaussian Polishing**.
#
#**Key Improvements:**
#1.  **Fully Vectorized Operations:** The algorithm replaces loop-based mutation and crossover (from Algorithm 8) with NumPy vectorization. This minimizes interpreter overhead, allowing significantly more function evaluations within the `max_time`, which is crucial for finding better minima.
#2.  **Strategy (DE/current-to-best/1/bin):** It utilizes the greedy `current-to-best` mutation strategy. This approach converges faster than standard `DE/rand` by guiding the population toward the best-known solution, while the difference vector `(r1 - r2)` maintains diversity.
#3.  **Restart Mechanism with Elitism:** To prevent the greedy strategy from getting stuck in local minima, the algorithm monitors population variance. If the population converges (low standard deviation) or stagnates, it restarts with a fresh population but injects the global best solution (Elitism) to ensure monotonic improvement.
#4.  **Gaussian Polish:** Before restarting, a lightweight local search (Gaussian walk with shrinking variance) is performed around the global best. This exploits the final moments of a convergence phase to refine the solution precision, often securing the last few decimal points of improvement.
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
    
    # Population Size Heuristic:
    # Large enough for diversity, small enough for speed in high dimensions.
    # 15*dim is a standard robust choice.
    pop_size = max(20, 15 * dim)
    
    # Global Best Tracking (Persists across restarts)
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
            
            # 1. Identify current best in population
            best_idx = np.argmin(fitness)
            x_best = pop[best_idx]
            
            # 2. Generate Parameters (Dithering)
            # F: Uniform random [0.5, 0.9] per individual to maintain diversity
            F = np.random.uniform(0.5, 0.9, (pop_size, 1))
            
            # CR: Uniform random [0.8, 1.0] - High crossover favors convergence
            CR = np.random.uniform(0.8, 1.0, (pop_size, 1))
            
            # 3. Select random indices r1, r2 != i
            # Vectorized selection excluding i
            r1 = np.random.randint(0, pop_size - 1, pop_size)
            r1 += (r1 >= np.arange(pop_size)) # Shift indices >= i to ensure r1 != i
            
            # Select r2
            # For speed, we just ensure r2 != i. DE is robust to occasional r1==r2.
            r2 = np.random.randint(0, pop_size - 1, pop_size)
            r2 += (r2 >= np.arange(pop_size))
            
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # 4. Compute Mutant Vectors (Vectorized)
            # Broadcasting x_best to match pop shape
            mutant = pop + F * (x_best - pop) + F * (x_r1 - x_r2)
            
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
            if not epoch_improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # Calculate population diversity
            std_fit = np.std(fitness)
            
            # Trigger Restart if converged or stagnated
            if std_fit < 1e-6 or stagnation_counter > 20:
                
                # --- Gaussian Polish Phase ---
                # Before discarding the population, perform a local search around the best
                # to refine the solution precision.
                if global_best_vec is not None:
                    curr_vec = global_best_vec.copy()
                    curr_val = global_best_val
                    
                    # Search radius starts at 5% of domain size and shrinks
                    sigma = np.max(diff_b) * 0.05 
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
