#Here is a fully operational, self-contained Python solution.
#
#It implements a **JADE (Joint Adaptive Differential Evolution) with Restarts** algorithm. This approach significantly outperforms standard DE by:
#1.  **Adaptive Parameters**: Automatically adjusting the mutation factor ($F$) and crossover rate ($CR$) based on successful updates in previous generations, removing the need for manual tuning.
#2.  **Current-to-pBest Mutation**: Guiding the search towards the top $p\%$ of best solutions found so far, which accelerates convergence compared to random mutation.
#3.  **Restarts**: Detecting population stagnation (low variance) and restarting with a fresh population to escape local minima, while preserving the global best.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using JADE (Adaptive Differential Evolution) with Restarts.
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Balance between exploration (large) and speed (small).
    # Adaptive DE works well with smaller populations than standard DE.
    pop_size = int(10 * dim)
    pop_size = np.clip(pop_size, 20, 60) # Cap to ensure enough generations run
    
    # JADE Adaptive Parameters
    mu_cr = 0.5       # Initial mean Crossover Rate
    mu_f = 0.5        # Initial mean Mutation Factor
    c_adapt = 0.1     # Adaptation learning rate
    p_best_rate = 0.1 # Top 10% used for mutation guidance
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    best_fitness = float('inf')
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Time check before starting new population
        if datetime.now() >= end_time: return best_fitness
        
        # 1. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_fitness
            val = func(pop[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
        
        # 2. Evolution Loop
        while True:
            # Check time
            if datetime.now() >= end_time: return best_fitness
            
            # Check for stagnation/convergence to trigger restart
            if np.std(fitness) < 1e-8:
                break 
            
            # --- Parameter Generation (JADE) ---
            # Sort population indices by fitness to find top p%
            sorted_idx = np.argsort(fitness)
            top_cut = max(1, int(pop_size * p_best_rate))
            top_indices = sorted_idx[:top_cut]
            
            # Generate CR: Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)
            
            # Generate F: Cauchy(mu_f, 0.1)
            # Cauchy random number ~ mu + scale * tan(pi * (rand - 0.5))
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            # Truncate F: if > 1 clamp to 1, if <= 0 clamp to 0.1 (approx resampling)
            f = np.where(f >= 1.0, 1.0, f)
            f = np.where(f <= 0.0, 0.1, f)
            
            # --- Mutation: current-to-pbest/1 ---
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Select random pbest for each individual
            pbest_idx = np.random.choice(top_indices, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1, r2 (random indices from population)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Calculate mutant vectors (Vectorized)
            f_col = f[:, np.newaxis] # Reshape for broadcasting
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, np.newaxis]
            
            # Forced Gene: Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            # Create trial population and clip to bounds
            trial_pop = np.where(cross_mask, mutant, pop)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection and Adaptation ---
            successful_f = []
            successful_cr = []
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_fitness
                
                f_trial = func(trial_pop[i])
                
                # Greedy selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # Record successful parameters
                    successful_f.append(f[i])
                    successful_cr.append(cr[i])
                    
                    if f_trial < best_fitness:
                        best_fitness = f_trial
            
            # --- Update Adaptive Means ---
            if successful_f:
                sf = np.array(successful_f)
                scr = np.array(successful_cr)
                
                # Update mu_f using Lehmer Mean (gives more weight to larger F)
                mean_lehmer = np.sum(sf**2) / np.sum(sf)
                mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer
                
                # Update mu_cr using Arithmetic Mean
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(scr)

    return best_fitness
