#The output value is: 55.32187
#
#The algorithm code is:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using a Restarting Self-Adaptive Differential Evolution (JADE) algorithm.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # ----------------------------------------------------
    # Configuration
    # ----------------------------------------------------
    # Population size: Adaptive based on dimension.
    # JADE typically benefits from a population size of around 10*dim to 20*dim.
    # We clip it to ensure speed on expensive functions within limited time.
    pop_size = int(np.clip(20 * dim, 50, 100))
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')

    # ----------------------------------------------------
    # Restart Loop
    # ----------------------------------------------------
    # Restarts help escape local optima if the population converges prematurely.
    while True:
        # Check remaining time before starting a new run
        if datetime.now() - start_time >= time_limit:
            return global_best_val
            
        # Initialize Population (Uniform Random)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # JADE Adaptive Parameters Initialization
        mu_cr = 0.5
        mu_f = 0.5
        c = 0.1   # Adaptation rate
        p = 0.05  # Top percentage for p-best selection
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val

        # ------------------------------------------------
        # Evolutionary Loop (JADE Strategy)
        # ------------------------------------------------
        while True:
            # Time Check
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            # Sort population by fitness
            # This simplifies finding the top-p individuals
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Convergence Check
            # If the population fitness range is negligible, restart.
            if fitness[-1] - fitness[0] < 1e-8:
                break
            
            # --------------------------------------------
            # Generate Adaptive Parameters
            # --------------------------------------------
            # CR ~ Normal(mu_cr, 0.1), clipped to [0, 1]
            cr_i = np.random.normal(mu_cr, 0.1, pop_size)
            cr_i = np.clip(cr_i, 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1)
            # Approximated using tan(pi * (rand - 0.5))
            rand_c = np.random.rand(pop_size)
            f_i = mu_f + 0.1 * np.tan(np.pi * (rand_c - 0.5))
            
            # Handle F bounds: if <= 0 set to 0.1, if > 1 set to 1.0
            f_i = np.where(f_i <= 0, 0.1, f_i)
            f_i = np.minimum(f_i, 1.0)
            
            # --------------------------------------------
            # Mutation: DE/current-to-pbest/1
            # V = X_i + F_i * (X_pbest - X_i) + F_i * (X_r1 - X_r2)
            # --------------------------------------------
            
            # Select X_pbest from top p% individuals
            top_cnt = max(1, int(pop_size * p))
            pbest_indices = np.random.randint(0, top_cnt, pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select distinct random indices r1, r2
            # For speed, we use random draw and ignore rare collision probability
            idxs = np.random.randint(0, pop_size, (pop_size, 2))
            x_r1 = pop[idxs[:, 0]]
            x_r2 = pop[idxs[:, 1]]
            
            # Compute Mutant Vectors (Vectorized)
            f_col = f_i[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --------------------------------------------
            # Crossover (Binomial)
            # --------------------------------------------
            rand_vals = np.random.rand(pop_size, dim)
            mask = rand_vals < cr_i[:, None]
            
            # Guarantee at least one parameter changes
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            # Create Trial Population
            trial_pop = np.where(mask, mutant, pop)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --------------------------------------------
            # Selection and Parameter Update
            # --------------------------------------------
            succ_f = []
            succ_cr = []
            
            # Evaluate trial vectors
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    pop[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    # Store successful parameters
                    succ_f.append(f_i[i])
                    succ_cr.append(cr_i[i])
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
            
            # Update Adaptive Means (Lehmer Mean for F, Arithmetic Mean for CR)
            if len(succ_f) > 0:
                sf = np.array(succ_f)
                scr = np.array(succ_cr)
                
                mean_lehmer = np.sum(sf**2) / np.sum(sf)
                mu_f = (1 - c) * mu_f + c * mean_lehmer
                mu_cr = (1 - c) * mu_cr + c * np.mean(scr)
                
    return global_best_val
