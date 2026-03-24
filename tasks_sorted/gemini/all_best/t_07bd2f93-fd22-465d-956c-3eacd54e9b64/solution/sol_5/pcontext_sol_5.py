#The following output presents a highly optimized algorithm based on **JADE (Self-Adaptive Differential Evolution)** enhanced with a **Gaussian Polishing** phase and **Restarts**.
#
#**Key Improvements:**
#1.  **Self-Adaptation (JADE):** The algorithm automatically adapts the mutation factor ($F$) and crossover rate ($CR$) based on successful solutions in previous generations. This removes the need for manual hyperparameter tuning and fits the strategy to the specific function landscape.
#2.  **Vectorized Operations:** Mutation, crossover, and boundary handling are implemented using vectorized NumPy operations. This significantly reduces the Python interpretation overhead, allowing for more function evaluations within the time limit.
#3.  **Current-to-pBest Mutation:** This strategy drives the population towards the top $p\%$ of best individuals. It is more robust than "current-to-best" (which can get stuck in local minima) and faster than "rand". It also utilizes an **external archive** of historically good solutions to maintain diversity.
#4.  **Gaussian Polish:** When the population converges (low variance), a lightweight local search (Gaussian walk with shrinking variance) is triggered on the best solution to refine the result to high precision before restarting.
#5.  **Restart with Elitism:** If the search stagnates, the population is reset to explore new areas, but the global best solution is preserved (elitism) to ensure monotonic improvement.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized JADE (Self-Adaptive Differential Evolution)
    with 'current-to-pbest' mutation, External Archive, and Gaussian Polishing 
    upon convergence.
    """
    
    # --- 1. Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    def is_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- 2. Hyperparameters ---
    # Population size: Standard heuristic is 10*dim, but we cap it 
    # to ensure convergence within time limits for high dimensions.
    pop_size = min(100, max(30, 10 * dim))
    
    # JADE Adaptive parameters
    mu_cr = 0.5      # Mean Crossover Rate
    mu_f = 0.5       # Mean Mutation Factor
    c_adapt = 0.1    # Adaptation rate
    p_best_rate = 0.10 # Top 10% for mutation guidance
    
    # Archive parameters
    archive_size = pop_size 
    
    # --- 3. Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    global_best_vec = None

    # --- 4. Main Optimization Loop (Restarts) ---
    while not is_timeout():
        
        # A. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if is_timeout(): return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
                
        # Elitism: Inject global best into new population to preserve progress
        if global_best_vec is not None:
            worst_idx = np.argmax(fitness)
            pop[worst_idx] = global_best_vec
            fitness[worst_idx] = global_best_val
            
        # Initialize Archive
        archive = []
        
        # B. Evolutionary Loop
        stagnation_counter = 0
        
        while not is_timeout():
            
            # --- 1. Parameter Adaptation (Vectorized) ---
            # CR ~ Normal(mu_cr, 0.1)
            cr_g = np.random.normal(mu_cr, 0.1, pop_size)
            cr_g = np.clip(cr_g, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            # Cauchy distribution allows for occasional large jumps (exploration)
            f_g = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            f_g = np.clip(f_g, 0.1, 1.0) # Clip to valid range [0.1, 1.0]
            f_g[f_g <= 0] = 0.1 # Safety floor
            
            # --- 2. Mutation: DE/current-to-pbest/1 ---
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            
            # Sort population to find top p%
            sorted_indices = np.argsort(fitness)
            num_pbest = max(1, int(p_best_rate * pop_size))
            pbest_indices = sorted_indices[:num_pbest]
            
            # Select x_pbest randomly for each individual
            pbest_choices = np.random.choice(pbest_indices, pop_size)
            x_pbest = pop[pbest_choices]
            
            # Select x_r1: random from population
            r1 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1]
            
            # Select x_r2: random from (population U archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                pop_archive = np.vstack((pop, archive_np))
            else:
                pop_archive = pop
            
            r2 = np.random.randint(0, len(pop_archive), pop_size)
            x_r2 = pop_archive[r2]
            
            # Compute Mutant Vectors (Vectorized)
            f_col = f_g[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- 3. Crossover: Binomial ---
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            
            # Mask: True if crossover occurs
            mask = rand_vals < cr_g[:, np.newaxis]
            # Ensure at least one dimension is taken from mutant
            mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask, mutant, pop)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- 4. Selection and Evaluation ---
            succ_f = []
            succ_cr = []
            
            for i in range(pop_size):
                if is_timeout(): return global_best_val
                
                t_val = func(trial_pop[i])
                
                if t_val <= fitness[i]:
                    # Success: Update Population
                    
                    # Add replaced parent to archive
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace random archive member
                        archive[np.random.randint(0, len(archive))] = pop[i].copy()
                    
                    pop[i] = trial_pop[i]
                    fitness[i] = t_val
                    
                    # Record successful parameters
                    succ_f.append(f_g[i])
                    succ_cr.append(cr_g[i])
                    
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_vec = trial_pop[i].copy()
            
            # --- 5. Update Adaptive Parameters (JADE Logic) ---
            if len(succ_f) > 0:
                succ_f = np.array(succ_f)
                succ_cr = np.array(succ_cr)
                
                # Lehmer Mean for F
                mean_lehmer = np.sum(succ_f**2) / (np.sum(succ_f) + 1e-10)
                mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer
                
                # Arithmetic Mean for CR
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(succ_cr)
                
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # --- 6. Convergence Check & Polish ---
            # If population variance is low or stagnation is high, we polish and restart
            fit_std = np.std(fitness)
            if fit_std < 1e-6 or stagnation_counter > 30:
                
                # --- Gaussian Polish Phase ---
                # A lightweight local search around the global best
                curr_polish = global_best_vec.copy()
                curr_val = global_best_val
                
                # Start sigma at 10% of domain size
                sigma = np.max(diff_b) * 0.1 
                
                # Limited budget for polishing
                polish_iters = 5 * dim 
                
                for _ in range(polish_iters):
                    if is_timeout(): return global_best_val
                    
                    # Generate neighbor
                    perturb = np.random.normal(0, sigma, dim)
                    neighbor = np.clip(curr_polish + perturb, min_b, max_b)
                    
                    val_n = func(neighbor)
                    
                    if val_n < curr_val:
                        curr_val = val_n
                        curr_polish = neighbor
                        if val_n < global_best_val:
                            global_best_val = val_n
                            global_best_vec = neighbor.copy()
                    else:
                        # Contract step size if no improvement
                        sigma *= 0.6
                        if sigma < 1e-9:
                            break
                
                # Break inner loop to trigger restart
                break 

    return global_best_val
