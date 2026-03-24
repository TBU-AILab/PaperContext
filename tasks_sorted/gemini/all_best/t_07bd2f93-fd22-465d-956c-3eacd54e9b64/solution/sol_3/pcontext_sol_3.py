#The following algorithm implements a **Self-Adaptive Differential Evolution (JADE)** strategy with **Restart**.
#
#**Improvements:**
#1.  **Adaptive Parameters (JADE):** Instead of fixed or randomized hyperparameters, this algorithm adapts the mutation factor ($F$) and crossover rate ($CR$) based on the success history of the population. It uses a **Lehmer mean** for $F$ and arithmetic mean for $CR$ to guide future generations towards successful parameter regions.
#2.  **Current-to-pBest Mutation:** Instead of `current-to-best` (which can be too greedy) or `rand` (which is too slow), it uses `current-to-pbest`. It guides individuals towards a random solution within the top $p\%$ (e.g., top 5%) of the population. This balances convergence speed with robustness against local minima.
#3.  **External Archive:** It maintains an archive of inferior solutions recently replaced. These archived solutions are used in the mutation step (`x_r2`) to preserve diversity and slow down the loss of population variance.
#4.  **Smart Restart:** The algorithm monitors population variance and stagnation. If the search stagnates or converges prematurely, it triggers a restart, re-initializing the population while preserving the single best solution found so far (Elitism).
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using JADE (Adaptive Differential Evolution) with 
    Archive and Restarts.
    """
    # --- Time Management ---
    start_time = datetime.now()
    timeout = timedelta(seconds=max_time)
    
    def check_timeout():
        return (datetime.now() - start_time) >= timeout

    # --- Hyperparameters ---
    # Population size: Robust choice is 10*dim to 20*dim. 
    # We use a dynamic size, bounded to keep iterations fast.
    pop_size = max(20, 10 * dim)
    
    # JADE Adaptation settings
    c_adapt = 0.1      # Parameter adaptation rate
    p_best_rate = 0.05 # Top 5% for current-to-pbest mutation
    
    # Archive settings
    max_archive_size = pop_size
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    global_best_vec = None
    
    # --- Main Optimization Loop (Restarts) ---
    while not check_timeout():
        
        # 1. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best to guide this restart (if available)
        if global_best_vec is not None:
            pop[0] = global_best_vec
            
        # Evaluate initial population
        fitness = np.array([float('inf')] * pop_size)
        
        # Safe evaluation loop
        for i in range(pop_size):
            if check_timeout():
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
                
        # 2. Algorithm State for this epoch
        mu_cr = 0.5
        mu_f = 0.5
        archive = [] # Archive is cleared on restart to allow fresh exploration
        stall_counter = 0
        
        # 3. Epoch Loop
        while not check_timeout():
            
            # --- Parameter Generation (JADE) ---
            # CR ~ Normal(mu_cr, 0.1)
            crs = np.random.normal(mu_cr, 0.1, pop_size)
            crs = np.clip(crs, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            # Generate Cauchy: standard_cauchy() * gamma + x0
            fs = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
            # Clip F to [0.1, 1.0] to prevent degeneration or explosion
            fs = np.clip(fs, 0.1, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # Sort population to find top p% individuals
            sorted_indices = np.argsort(fitness)
            sorted_pop = pop[sorted_indices]
            
            # Select x_pbest randomly from top p%
            num_pbest = max(1, int(p_best_rate * pop_size))
            pbest_indices = np.random.randint(0, num_pbest, pop_size)
            x_pbest = sorted_pop[pbest_indices]
            
            # Select r1: Random from population
            r1_indices = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_indices]
            
            # Select r2: Random from Union(Pop, Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                # Stack pop and archive for selection
                pop_archive = np.vstack((pop, archive_np))
            else:
                pop_archive = pop
                
            r2_indices = np.random.randint(0, len(pop_archive), pop_size)
            x_r2 = pop_archive[r2_indices]
            
            # Compute Mutant Vectors
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            F_col = fs[:, np.newaxis]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            mask = rand_vals < crs[:, np.newaxis]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection & Evaluation ---
            successful_fs = []
            successful_crs = []
            
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            epoch_improved = False
            
            for i in range(pop_size):
                if check_timeout():
                    return global_best_val
                
                f_trial = func(trial[i])
                
                if f_trial < fitness[i]:
                    # Survivor Selection: Trial beats Parent
                    new_pop[i] = trial[i]
                    new_fitness[i] = f_trial
                    
                    # Add parent to archive (for diversity in future mutations)
                    archive.append(pop[i].copy())
                    
                    # Record successful parameters
                    successful_fs.append(fs[i])
                    successful_crs.append(crs[i])
                    
                    # Update Global Best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_vec = trial[i].copy()
                        epoch_improved = True
            
            # Manage Archive Size
            while len(archive) > max_archive_size:
                # Randomly remove elements to keep archive size constant
                archive.pop(np.random.randint(0, len(archive)))
                
            pop = new_pop
            fitness = new_fitness
            
            # --- Adapt Parameters (JADE Logic) ---
            if len(successful_fs) > 0:
                s_f = np.array(successful_fs)
                s_cr = np.array(successful_crs)
                
                # Lehmer Mean for F: sum(f^2) / sum(f)
                mean_lehmer = np.sum(s_f**2) / (np.sum(s_f) + 1e-10)
                mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer
                
                # Arithmetic Mean for CR
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(s_cr)
                
            # --- Convergence Detection ---
            # If population variance is extremely low, we are likely stuck in a local minimum
            std_dev = np.std(fitness)
            if std_dev < 1e-9:
                break
            
            # Stagnation counter: if no improvement for many generations
            if epoch_improved:
                stall_counter = 0
            else:
                stall_counter += 1
                
            # Restart if stagnated for too long (heuristic based on dim)
            if stall_counter > 50 + dim:
                break
                
    return global_best_val
