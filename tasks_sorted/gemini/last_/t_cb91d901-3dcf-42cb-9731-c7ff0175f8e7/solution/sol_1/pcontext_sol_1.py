#Here is a fully operational and self-contained Python algorithm implementing **Restarting Self-Adaptive Differential Evolution (JADE variant)**.
#
#This algorithm improves upon standard Differential Evolution by:
#1.  **Self-Adaptation**: Automatically tuning mutation factor ($F$) and crossover rate ($CR$) during the run, allowing it to adapt to the function's landscape.
#2.  **Greedy Mutation Strategy**: Using `current-to-pbest` mutation to guide the search towards the best solutions found so far while maintaining diversity.
#3.  **Restarts**: Detecting convergence (stagnation) and restarting the population to escape local optima and explore new areas of the search space within the remaining time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a black-box function using Restarting Self-Adaptive Differential Evolution (JADE).
    
    Key features:
    - Adaptive F and CR parameters.
    - DE/current-to-pbest/1 mutation strategy.
    - Automatic restart mechanism upon convergence.
    - Strict time management.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: JADE is robust with ~100, but we scale it 
    # slightly with dimension while capping to ensure iteration speed.
    pop_size = int(max(20, min(100, 10 * dim)))
    
    # Adaptive Parameter Initial Means
    mu_cr = 0.5
    mu_f = 0.5
    adapt_c = 0.1  # Learning rate for parameter updates
    p_share = 0.05 # Top 5% for current-to-pbest mutation
    
    # Precompute bounds for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found
    best_val = float('inf')
    
    # --- Main Loop (Restarts) ---
    while True:
        # Check overall time before starting a new run/restart
        if datetime.now() - start_time >= limit:
            return best_val
            
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Initial Evaluation of the Population
        for i in range(pop_size):
            if datetime.now() - start_time >= limit:
                return best_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
        
        # --- Evolution Loop ---
        while True:
            # Time check per generation
            if datetime.now() - start_time >= limit:
                return best_val
            
            # Convergence Check: If population fitness variance is negligible, restart
            if np.std(fitness) < 1e-8:
                break 
            
            # Sort population to facilitate p-best selection
            # We sort arrays so pop[0] is the best
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            # --- Parameter Adaptation ---
            # Generate F using Cauchy distribution: loc=mu_f, scale=0.1
            # Generate CR using Normal distribution: mean=mu_cr, std=0.1
            
            # Note: standard_cauchy returns samples from Cauchy(0, 1)
            F = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Handle F bounds: if > 1 clamp to 1, if <= 0 clamp to 0.1 (simplified regeneration)
            F = np.where(F > 1.0, 1.0, F)
            F = np.where(F <= 0.0, 0.1, F)
            
            CR = np.random.normal(mu_cr, 0.1, pop_size)
            CR = np.clip(CR, 0.0, 1.0)
            
            # --- Mutation: DE/current-to-pbest/1 ---
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # Indices for mutation components
            # pbest: randomly selected from the top p% individuals
            p_top = max(1, int(pop_size * p_share))
            idx_pbest = np.random.randint(0, p_top, pop_size)
            
            # r1, r2: randomly selected from population
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            idx_r2 = np.random.randint(0, pop_size, pop_size)
            
            # Vectorized selection of parents
            x = pop
            xp = pop[idx_pbest]
            xr1 = pop[idx_r1]
            xr2 = pop[idx_r2]
            
            # Reshape F for broadcasting: (pop_size, 1)
            F_v = F[:, np.newaxis]
            
            # Compute Mutant Vectors
            mutant = x + F_v * (xp - x) + F_v * (xr1 - xr2)
            
            # --- Crossover: Binomial ---
            mask_rand = np.random.rand(pop_size, dim)
            CR_v = CR[:, np.newaxis]
            cross_mask = mask_rand < CR_v
            
            # Ensure at least one dimension is inherited from mutant (standard DE requirement)
            j_rand = np.random.randint(0, dim, pop_size)
            j_mask = np.zeros((pop_size, dim), dtype=bool)
            j_mask[np.arange(pop_size), j_rand] = True
            cross_mask = np.logical_or(cross_mask, j_mask)
            
            # Create Trial Population
            trial_pop = np.where(cross_mask, mutant, x)
            
            # Bound Handling: Clip to boundaries
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection and Adaptation Stats Collection ---
            succ_F = []
            succ_CR = []
            
            # Evaluate Trial Vectors
            for i in range(pop_size):
                if datetime.now() - start_time >= limit:
                    return best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    succ_F.append(F[i])
                    succ_CR.append(CR[i])
                    
                    if f_trial < best_val:
                        best_val = f_trial
            
            # --- Update Adaptive Means ---
            # If there were successful mutations, update mu_f and mu_cr
            if len(succ_F) > 0:
                sf = np.array(succ_F)
                scr = np.array(succ_CR)
                
                # Lehmer Mean for F (biases towards larger successful F)
                mean_lehmer = np.sum(sf**2) / (np.sum(sf) + 1e-9)
                mu_f = (1 - adapt_c) * mu_f + adapt_c * mean_lehmer
                
                # Arithmetic Mean for CR
                mu_cr = (1 - adapt_c) * mu_cr + adapt_c * np.mean(scr)
                
    return best_val
