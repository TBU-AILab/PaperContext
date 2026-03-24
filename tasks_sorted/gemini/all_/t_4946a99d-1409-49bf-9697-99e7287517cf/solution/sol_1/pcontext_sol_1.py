#The improved algorithm implements a **Self-Adaptive Differential Evolution (JADE-like) with Restarts**. 
#
#**Key Improvements:**
#1.  **Self-Adaptation:** Instead of fixed `F` and `CR` parameters (which can be suboptimal), the algorithm adaptively learns these values during execution (`mu_cr`, `mu_f`) based on successful mutations.
#2.  **Guided Mutation (`current-to-pbest`):** This strategy uses the top $p\%$ of the population to guide mutations, offering significantly faster convergence than standard random mutation.
#3.  **Restarts:** To prevent getting stuck in local optima (a likely cause for the value ~94 in the previous attempt), the algorithm detects stagnation (low variance or no improvement) and restarts the population while preserving the global best found. This is crucial for multimodal functions.
#4.  **Vectorization:** Most operations are vectorized using NumPy to maximize the number of function evaluations within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using a Restarting Self-Adaptive Differential Evolution (JADE-variant).
    Features:
    - Adaptive parameter control (F, CR) for robustness.
    - Current-to-pbest mutation for faster convergence.
    - Restart mechanism to escape local optima.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)

    def is_time_up():
        return (datetime.now() - start_time) >= limit

    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Algorithm Hyperparameters
    # Population size: larger allows diversity, but constrained for speed.
    # 40-100 is a robust range for moderate dimensions.
    pop_size = int(np.clip(10 * dim, 40, 100))
    
    global_best_val = float('inf')

    # --- Restart Loop ---
    # Allows exploring multiple basins of attraction within the time limit
    while not is_time_up():
        
        # 1. Initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if is_time_up(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Adaptive Parameter Memory (JADE logic)
        mu_cr = 0.5
        mu_f = 0.5
        learn_rate = 0.1
        
        # Stagnation counters
        stall_count = 0
        current_run_best = np.min(fitness)
        
        # --- Evolution Loop ---
        while not is_time_up():
            
            # 2. Parameter Generation
            # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1). 
            # Approx using standard_cauchy: F = mu + 0.1 * Cauchy
            # F is clamped to [0.1, 1.0] to prevent degeneration or explosion
            f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            f = np.clip(f, 0.1, 1.0)

            # 3. Mutation: current-to-pbest/1
            # Sort population to find top p% (pbest)
            sorted_idx = np.argsort(fitness)
            p_count = max(1, int(0.1 * pop_size)) # top 10%
            top_p_idx = sorted_idx[:p_count]
            
            # Select pbest, r1, r2
            pbest_indices = np.random.choice(top_p_idx, pop_size)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            r2_indices = np.random.randint(0, pop_size, pop_size)
            
            # Vectorized mutation
            x_i = pop
            x_pbest = pop[pbest_indices]
            x_r1 = pop[r1_indices]
            x_r2 = pop[r2_indices]
            
            F_col = f[:, np.newaxis]
            
            # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            v = x_i + F_col * (x_pbest - x_i) + F_col * (x_r1 - x_r2)
            v = np.clip(v, min_b, max_b)
            
            # 4. Crossover (Binomial)
            # Ensure at least one variable is mutated per vector
            j_rand = np.random.randint(0, dim, pop_size)
            rand_uni = np.random.rand(pop_size, dim)
            mask = rand_uni < cr[:, np.newaxis]
            mask[np.arange(pop_size), j_rand] = True
            
            u = np.where(mask, v, pop)
            
            # 5. Selection and Adaptation
            succ_f = []
            succ_cr = []
            
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                f_trial = func(u[i])
                
                if f_trial <= fitness[i]:
                    pop[i] = u[i]
                    fitness[i] = f_trial
                    
                    # Record successful parameters
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
            
            # Update Adaptive Means (Lehmer mean for F, Arithmetic for CR)
            if succ_f:
                mean_cr = np.mean(succ_cr)
                mu_cr = (1 - learn_rate) * mu_cr + learn_rate * mean_cr
                
                sf_arr = np.array(succ_f)
                mean_f = np.sum(sf_arr**2) / np.sum(sf_arr)
                mu_f = (1 - learn_rate) * mu_f + learn_rate * mean_f
            
            # 6. Check for Restart (Convergence or Stagnation)
            new_run_best = np.min(fitness)
            if new_run_best < current_run_best:
                current_run_best = new_run_best
                stall_count = 0
            else:
                stall_count += 1
            
            std_dev = np.std(fitness)
            
            # Restart if variance is tiny or no progress for ~40 gens
            if std_dev < 1e-8 or stall_count > 40:
                break

    return global_best_val
