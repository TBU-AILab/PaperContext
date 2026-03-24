#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 39.46349818182866
#2. output value is: 46.14857048099866
#
#Here is an improved algorithm. It implements **JADE (Adaptive Differential Evolution with Optional External Archive)**.
#
#### Rationale for Improvement
#The previous best result was achieved using a self-adaptive DE (jDE), but it can be further improved.
#1.  **Global Parameter Adaptation (JADE)**: Unlike jDE which adapts parameters at an individual level, JADE adapts the global means ($\mu_{CR}, \mu_F$) based on the success history of the entire population. This filters out noise and converges faster to optimal control parameters.
#2.  **External Archive**: The previous algorithms lacked a mechanism to preserve diversity effectively. This implementation uses an external archive to store recently replaced inferior solutions. During mutation, difference vectors can be formed using archived solutions, maintaining diversity and preventing premature convergence to local optima.
#3.  **Current-to-p-best Mutation**: Instead of `current-to-best` (which is greedy and risks getting stuck) or `rand` (which is slow), this uses `current-to-pbest`. It directs the search towards the top $p\%$ of good solutions, balancing exploitation and exploration robustness.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using JADE (Adaptive Differential Evolution with External Archive).
    This variant uses 'current-to-pbest' mutation and adapts crossover/mutation 
    parameters based on success history to find the global minimum efficiently.
    """
    
    # --- Configuration & Time Management ---
    start_time = datetime.now()
    # Set a hard deadline.
    end_time = start_time + timedelta(seconds=max_time)

    # --- Hyperparameters ---
    # Population size: 
    # A dynamic size bounded between 20 and 50 is generally robust for 
    # time-constrained black-box optimization.
    pop_size = int(max(20, min(50, 10 * dim)))
    
    # Adaptation parameters (initial values)
    mu_cr = 0.5     # Mean Crossover Rate
    mu_f = 0.5      # Mean Mutation Factor
    c_adapt = 0.1   # Adaptation learning rate
    p_best_rate = 0.05  # Top 5% used for mutation guidance (p-best)
    
    # Archive parameters
    archive = []
    max_archive_size = pop_size 

    # --- Initialization ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b

    # Initialize population randomly
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Track global best
    best_val = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # --- Main Optimization Loop ---
    while datetime.now() < end_time:
        
        # 1. Parameter Generation
        # Sort population to find top p-best indices later
        sorted_idx = np.argsort(fitness)
        
        # Generate CR_i ~ Normal(mu_cr, 0.1)
        cr_vals = np.random.normal(mu_cr, 0.1, pop_size)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        # Generate F_i ~ Cauchy(mu_f, 0.1)
        # Note: numpy doesn't have a direct Cauchy(loc, scale), so we use standard + math
        f_vals = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Constrain F values: If F > 1 set to 1. If F <= 0 regenerate.
        while True:
            bad_mask = f_vals <= 0
            if not np.any(bad_mask):
                break
            # Regenerate only the bad ones
            f_vals[bad_mask] = mu_f + 0.1 * np.random.standard_cauchy(np.sum(bad_mask))
        f_vals = np.minimum(f_vals, 1.0)

        # Lists to store parameters that lead to improvement
        succ_cr = []
        succ_f = []
        
        # 2. Evolution Cycle (Iterate through population)
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_val

            # --- Mutation Strategy: DE/current-to-pbest/1 ---
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            x_i = pop[i]
            F_i = f_vals[i]
            CR_i = cr_vals[i]
            
            # Select X_pbest: Randomly from top p%
            top_count = max(1, int(pop_size * p_best_rate))
            p_idx = sorted_idx[np.random.randint(0, top_count)]
            x_pbest = pop[p_idx]
            
            # Select r1: Random from population, distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = pop[r1]
            
            # Select r2: Random from Union(Population, Archive), distinct from i and r1
            pool_size = pop_size + len(archive)
            r2_idx = np.random.randint(0, pool_size)
            
            if r2_idx < pop_size:
                # If r2 is from population, ensure it's distinct
                while r2_idx == i or r2_idx == r1:
                    r2_idx = np.random.randint(0, pool_size)
                    # If we picked from archive in retry (very unlikely loop), handle it:
                    if r2_idx >= pop_size: 
                        break 
                
                if r2_idx < pop_size:
                    x_r2 = pop[r2_idx]
                else:
                    x_r2 = archive[r2_idx - pop_size]
            else:
                # If r2 is from archive, it's automatically distinct from current pop indices
                x_r2 = archive[r2_idx - pop_size]

            # Calculate mutant vector
            mutant = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            cross_mask = np.random.rand(dim) < CR_i
            # Force at least one dimension to change
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # Bound constraints
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                # Improvement: update population and archive
                
                # Add current (replaced) solution to archive
                archive.append(x_i.copy())
                if len(archive) > max_archive_size:
                    # Remove a random member to keep archive size constant
                    del archive[np.random.randint(0, len(archive))]
                
                # Update population
                pop[i] = trial
                fitness[i] = f_trial
                
                # Save successful parameters
                succ_cr.append(CR_i)
                succ_f.append(F_i)
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
            
        # 3. Adaptation (Update global means)
        if len(succ_cr) > 0:
            # Update mu_cr (Arithmetic Mean)
            mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(succ_cr)
            
            # Update mu_f (Lehmer Mean: sum(x^2)/sum(x))
            sf_arr = np.array(succ_f)
            mean_lehmer = np.sum(sf_arr**2) / (np.sum(sf_arr) + 1e-10)
            mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer

    return best_val
