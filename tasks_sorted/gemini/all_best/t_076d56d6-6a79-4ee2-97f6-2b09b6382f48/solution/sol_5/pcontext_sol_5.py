#The output value of the best generated algorithm is: 4.589970908775712
#The best generated algorithm code:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the function `func` using Restart-based JADE (Joint Adaptive Differential Evolution)
    with IPOP (Increasing Population Size) and an external Archive.
    
    Key Improvements:
    1.  **JADE Adaptation**: Automatically adapts the mean mutation factor (mu_f) and crossover rate (mu_cr) 
        based on successful updates, removing the need for static parameter tuning.
    2.  **External Archive**: Maintains a history of decent inferior solutions to preserve diversity in 
        the `current-to-pbest` mutation strategy, significantly improving performance on multimodal landscapes.
    3.  **IPOP Strategy**: Restarts with exponentially increasing population size when convergence is detected.
    4.  **Midpoint Bound Handling**: Instead of clipping to the edge, out-of-bound solutions are reset to 
        the midpoint between the parent and the bound, improving search near boundaries.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracking ---
    best_global_val = float('inf')
    best_global_vec = None
    
    # --- IPOP Settings ---
    # Start with a robust small population.
    base_pop_size = 20 + 4 * dim
    restart_count = 0
    
    # --- Main Restart Loop ---
    while True:
        # Check time before starting a new restart
        if datetime.now() - start_time >= time_limit:
            return best_global_val
            
        # IPOP: Increase population size exponentially
        pop_size = int(base_pop_size * (1.5 ** restart_count))
        
        # Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject best solution found so far into the new population
        start_idx = 0
        if best_global_vec is not None:
            population[0] = best_global_vec.copy()
            fitness[0] = best_global_val
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_global_val
            val = func(population[i])
            fitness[i] = val
            if val < best_global_val:
                best_global_val = val
                best_global_vec = population[i].copy()
                
        # --- JADE Adaptive Parameters ---
        # Initial means for F and CR
        mu_f = 0.5
        mu_cr = 0.5
        c_adapt = 0.1  # Adaptation learning rate
        
        # Archive to store successful but replaced solutions
        archive = []
        
        # --- Evolution Loop ---
        while True:
            # Strict Time Check
            if datetime.now() - start_time >= time_limit:
                return best_global_val
            
            # --- Convergence Check ---
            # If the population is clustered extremely tightly, we are likely stuck.
            if np.max(fitness) - np.min(fitness) < 1e-8:
                break
                
            # --- 1. Parameter Generation (Vectorized) ---
            # Generate CR from Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F from Cauchy(mu_f, 0.1)
            # Cauchy random: location + scale * tan(pi * (uniform - 0.5))
            u = np.random.rand(pop_size)
            f = mu_f + 0.1 * np.tan(np.pi * (u - 0.5))
            
            # Handle F constraints: F > 0 (retry), F <= 1 (clip)
            retry_mask = f <= 0
            while np.any(retry_mask):
                u_retry = np.random.rand(np.sum(retry_mask))
                f[retry_mask] = mu_f + 0.1 * np.tan(np.pi * (u_retry - 0.5))
                retry_mask = f <= 0
            f = np.clip(f, 0.0, 1.0)
            
            # --- 2. Mutation: DE/current-to-pbest/1 ---
            # Strategy: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            
            # Select p-best (top 11%)
            sorted_idx = np.argsort(fitness)
            num_top = max(2, int(0.11 * pop_size))
            top_indices = sorted_idx[:num_top]
            
            pbest_idx = np.random.choice(top_indices, pop_size)
            x_pbest = population[pbest_idx]
            
            # Select r1 (random from population, try to avoid self)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            col_r1 = (r1_idx == np.arange(pop_size))
            r1_idx[col_r1] = (r1_idx[col_r1] + 1) % pop_size
            x_r1 = population[r1_idx]
            
            # Select r2 (random from Population U Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                candidates = np.vstack((population, archive_np))
            else:
                candidates = population
            
            r2_idx = np.random.randint(0, len(candidates), pop_size)
            x_r2 = candidates[r2_idx]
            
            # Compute Mutant Vector
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # --- 3. Crossover (Binomial) ---
            mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask, mutant, population)
            
            # --- 4. Bound Handling (Midpoint) ---
            # If solution violates bounds, place it between parent and bound.
            # This preserves evolutionary direction better than clipping.
            # Lower bounds
            mask_l = trial_pop < min_b
            trial_pop[mask_l] = (population[mask_l] + min_b[mask_l]) / 2.0
            # Upper bounds
            mask_u = trial_pop > max_b
            trial_pop[mask_u] = (population[mask_u] + max_b[mask_u]) / 2.0
            
            # --- 5. Selection & Adaptation ---
            succ_f = []
            succ_cr = []
            
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_global_val
                    
                val_trial = func(trial_pop[i])
                
                if val_trial < fitness[i]:
                    # Successful Update
                    
                    # Add parent to archive before overwriting
                    if len(archive) < pop_size:
                        archive.append(population[i].copy())
                    else:
                        # Replace random archive member
                        rem_idx = np.random.randint(0, len(archive))
                        archive[rem_idx] = population[i].copy()
                        
                    fitness[i] = val_trial
                    population[i] = trial_pop[i]
                    
                    # Record successful parameters
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    
                    # Update Global Best
                    if val_trial < best_global_val:
                        best_global_val = val_trial
                        best_global_vec = trial_pop[i].copy()
            
            # --- 6. Parameter Adaptation (Lehmer Mean) ---
            if len(succ_f) > 0:
                sf = np.array(succ_f)
                scr = np.array(succ_cr)
                
                # Update mu_cr (Arithmetic Mean)
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(scr)
                
                # Update mu_f (Lehmer Mean)
                denom = np.sum(sf)
                if denom > 0:
                    lehmer_mean = np.sum(sf**2) / denom
                    mu_f = (1 - c_adapt) * mu_f + c_adapt * lehmer_mean
                else:
                    mu_f = (1 - c_adapt) * mu_f
                    
        # Loop ends due to convergence -> Restart with larger population
        restart_count += 1
