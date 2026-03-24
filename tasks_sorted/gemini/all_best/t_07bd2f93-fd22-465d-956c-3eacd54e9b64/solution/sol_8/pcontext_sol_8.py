#The following algorithm implements a **Vectorized Self-Adaptive Differential Evolution (JADE)** with **Restarts** and **Gaussian Polishing**.
#
#**Key Improvements:**
#1.  **JADE (Self-Adaptation):** Instead of fixed `F` and `CR` parameters, the algorithm adapts them based on successful mutations in previous generations. This allows it to automatically tune itself to the specific function landscape.
#2.  **Vectorization:** Heavy computations (mutation, crossover, boundary handling) are vectorized using NumPy, significantly reducing interpreter overhead compared to loop-based approaches.
#3.  **current-to-pbest Mutation:** This strategy balances greediness (convergence speed) with diversity. It guides individuals toward the top $p\%$ of the population rather than just the single best, preventing premature convergence to local minima.
#4.  **Gaussian Polishing:** When the population converges (low variance), the algorithm switches to a lightweight local search (Gaussian walk with shrinking variance) around the global best. This refines the solution to high precision (exploiting the "last mile").
#5.  **Restart with Elitism:** To handle multimodal functions, the algorithm restarts the population when stagnation is detected, but it injects the best-found solution (Elitism) into the new population to ensure monotonic improvement.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized JADE (Self-Adaptive Differential Evolution)
    with 'current-to-pbest' mutation, Archive, Gaussian Polishing, and Restarts.
    """
    # --- 1. Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    def is_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- 2. Hyperparameters ---
    # Population size: Balance between vectorization efficiency and iteration speed.
    # 10*dim to 20*dim is standard. We use a lower bound to ensure speed.
    pop_size = max(20, 10 * dim)
    
    # JADE Adaptation settings
    c_adapt = 0.1      # Adaptation rate for mu_F and mu_CR
    p_best_rate = 0.05 # Top 5% for current-to-pbest mutation
    
    # Archive settings
    max_archive_size = pop_size
    
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
        
        # Elitism: Inject global best to guide this restart (if available)
        if global_best_vec is not None:
            pop[0] = global_best_vec
            
        # Evaluate initial population
        for i in range(pop_size):
            if is_timeout(): return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
                
        # B. Epoch State
        mu_cr = 0.5
        mu_f = 0.5
        archive = []
        stagnation_counter = 0
        
        # C. Evolutionary Cycle
        while not is_timeout():
            
            # --- 1. Parameter Adaptation (JADE) ---
            # CR ~ Normal(mu_cr, 0.1)
            cr_g = np.random.normal(mu_cr, 0.1, pop_size)
            cr_g = np.clip(cr_g, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            # Standard Cauchy + location; clipped to [0.1, 1.0]
            f_g = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            f_g = np.clip(f_g, 0.1, 1.0)
            f_g[f_g <= 0] = 0.1 # Correction for negative values
            
            # --- 2. Mutation: current-to-pbest/1 ---
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # Identify X_pbest
            sorted_indices = np.argsort(fitness)
            num_pbest = max(1, int(p_best_rate * pop_size))
            pbest_indices = sorted_indices[:num_pbest]
            
            # Randomly select pbest for each individual
            pbest_choices = np.random.choice(pbest_indices, pop_size)
            x_pbest = pop[pbest_choices]
            
            # Select X_r1 (distinct from i)
            # Fast vectorized selection ensuring r1 != i
            r1 = np.random.randint(0, pop_size - 1, pop_size)
            # Shift indices >= i to skip i
            r1 += (r1 >= np.arange(pop_size)) 
            x_r1 = pop[r1]
            
            # Select X_r2 (from Pop U Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                pop_archive = np.vstack((pop, archive_np))
            else:
                pop_archive = pop
            
            # We don't strictly enforce r2 != r1 != i here for speed, 
            # as the archive makes the pool large enough.
            r2 = np.random.randint(0, len(pop_archive), pop_size)
            x_r2 = pop_archive[r2]
            
            # Compute Mutant
            f_col = f_g[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- 3. Crossover ---
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            
            # Binomial crossover mask
            mask = rand_vals < cr_g[:, np.newaxis]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # --- 4. Selection & Evaluation ---
            succ_f = []
            succ_cr = []
            epoch_improved = False
            
            for i in range(pop_size):
                if is_timeout(): return global_best_val
                
                t_val = func(trial[i])
                
                if t_val <= fitness[i]:
                    # Archive management: add replaced parent
                    if len(archive) < max_archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace random archive member
                        archive[np.random.randint(0, len(archive))] = pop[i].copy()
                    
                    # Update population
                    pop[i] = trial[i]
                    fitness[i] = t_val
                    
                    # Store successful parameters
                    succ_f.append(f_g[i])
                    succ_cr.append(cr_g[i])
                    
                    # Update Global Best
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_vec = trial[i].copy()
                        epoch_improved = True
            
            # --- 5. Adapt Parameters ---
            if len(succ_f) > 0:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                
                # Lehmer Mean for F
                mean_lehmer = np.sum(s_f**2) / (np.sum(s_f) + 1e-10)
                mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer
                
                # Arithmetic Mean for CR
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(s_cr)
                
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # --- 6. Convergence Check & Polish ---
            # If population is very flat (low variance) or stagnant
            std_dev = np.std(fitness)
            
            if std_dev < 1e-6 or stagnation_counter > 20:
                
                # --- Gaussian Polish Phase ---
                # Before restarting, perform a local search around the global best
                if global_best_vec is not None:
                    curr_polish = global_best_vec.copy()
                    curr_val = global_best_val
                    
                    # Shrinking sigma strategy
                    sigma = 0.05 * np.max(diff_b)
                    polish_steps = 50 
                    
                    for _ in range(polish_steps):
                        if is_timeout(): return global_best_val
                        
                        perturb = np.random.normal(0, sigma, dim)
                        cand = np.clip(curr_polish + perturb, min_b, max_b)
                        cand_val = func(cand)
                        
                        if cand_val < curr_val:
                            curr_val = cand_val
                            curr_polish = cand
                            if cand_val < global_best_val:
                                global_best_val = cand_val
                                global_best_vec = cand.copy()
                        else:
                            # Reduce step size if no improvement
                            sigma *= 0.6
                
                # Break inner loop to trigger Restart
                break
                
    return global_best_val
