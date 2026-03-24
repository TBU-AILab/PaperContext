#The following Python algorithm implements **JADE (Adaptive Differential Evolution with Optional External Archive)** combined with a **Restart Mechanism**.
#
#### Key Improvements:
#1.  **JADE Strategy**: Uses the `DE/current-to-pbest/1` mutation strategy with an optional external archive. This balances exploitation (by guiding search towards the best solutions found so far) and exploration (by maintaining diversity via the archive).
#2.  **Parameter Adaptation**: Automatically tunes the mutation factor ($F$) and crossover rate ($CR$) for each individual based on success history (Lehmer mean), removing the need for manual hyperparameter tuning.
#3.  **Vectorized Operations**: The generation of candidate solutions is fully vectorized using NumPy, significantly reducing overhead and allowing for more generations within the time limit compared to loop-based implementations.
#4.  **Restart Mechanism**: Detects convergence (when population variance drops) and restarts the search while preserving the best solution found. This prevents stagnation in local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using JADE (Adaptive Differential Evolution) with Restart.
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Safety buffer to ensure we return before the strict timeout
    end_time = start_time + timedelta(seconds=max_time - 0.05)
    
    # --- Configuration ---
    # Population size: Adaptive to dimension, clamped for efficiency
    pop_size = int(np.clip(10 * dim, 30, 100))
    
    # JADE Hyperparameters
    p = 0.05            # Top 5% for pbest selection
    c = 0.1             # Learning rate for parameter update
    archive_factor = 1.0 # Max archive size relative to population
    
    # Adaptive Parameter Means (Initial)
    mu_cr = 0.5
    mu_f = 0.5
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Archive for diversity maintenance
    archive = []
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_fitness
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while datetime.now() < end_time:
        
        # 1. Sort Population (Required for p-best selection)
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        
        # 2. Restart Mechanism (Convergence Detection)
        # If population diversity is lost, restart to explore new areas
        if np.std(fitness) < 1e-6 and np.std(pop) < 1e-6:
            # Preserve the elite (Global Best)
            elite = best_sol.copy()
            elite_fit = best_fitness
            
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Inject elite at index 0
            pop[0] = elite
            fitness[:] = float('inf')
            fitness[0] = elite_fit
            
            # Reset JADE adaptive parameters
            mu_cr = 0.5
            mu_f = 0.5
            archive = []
            
            # Re-evaluate new population (skipping elite)
            for i in range(1, pop_size):
                if datetime.now() >= end_time: return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            
            # Start fresh with new population
            continue

        # 3. Generate Adaptive Parameters (Vectorized)
        # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr_vals = np.random.normal(mu_cr, 0.1, pop_size)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        # F ~ Cauchy(mu_f, 0.1)
        # Approximation: location + scale * tan(pi * (rand - 0.5))
        f_vals = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        # Clip F: if > 1 set to 1, if <= 0 set to 0.1 (to ensure mutation)
        f_vals = np.where(f_vals > 1.0, 1.0, f_vals)
        f_vals = np.where(f_vals <= 0.0, 0.1, f_vals)
        
        # 4. Mutation: DE/current-to-pbest/1
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # Select p-best indices (top p%)
        top_p_count = max(1, int(pop_size * p))
        pbest_indices = np.random.randint(0, top_p_count, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 indices (random from pop, != current)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Fix collision where r1 == current_index
        conflict_mask = (r1_indices == np.arange(pop_size))
        r1_indices[conflict_mask] = (r1_indices[conflict_mask] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 indices (random from Pop U Archive)
        if len(archive) > 0:
            pop_union = np.vstack((pop, np.array(archive)))
        else:
            pop_union = pop
            
        r2_indices = np.random.randint(0, len(pop_union), pop_size)
        x_r2 = pop_union[r2_indices]
        
        # Compute Mutation Vectors
        # Reshape F for broadcasting
        f_col = f_vals[:, np.newaxis]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = rand_vals < cr_vals[:, np.newaxis]
        # Ensure at least one parameter comes from mutant
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 6. Bound Constraints
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 7. Selection and Archive Update
        successful_f = []
        successful_cr = []
        
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_fitness
            
            # Evaluate Candidate
            t_val = func(trial_pop[i])
            
            # Greedy Selection
            if t_val < fitness[i]:
                # Add old solution to archive
                archive.append(pop[i].copy())
                
                # Accept new solution
                fitness[i] = t_val
                pop[i] = trial_pop[i]
                
                # Record successful parameters
                successful_f.append(f_vals[i])
                successful_cr.append(cr_vals[i])
                
                # Update Global Best
                if t_val < best_fitness:
                    best_fitness = t_val
                    best_sol = trial_pop[i].copy()
        
        # 8. Manage Archive Size
        max_archive_size = int(pop_size * archive_factor)
        while len(archive) > max_archive_size:
            # Remove random element
            archive.pop(np.random.randint(0, len(archive)))
            
        # 9. Update Adaptive Parameter Means
        if len(successful_f) > 0:
            sf = np.array(successful_f)
            scr = np.array(successful_cr)
            
            # Lehmer Mean for F
            mean_lehmer = np.sum(sf**2) / np.sum(sf)
            mu_f = (1 - c) * mu_f + c * mean_lehmer
            
            # Arithmetic Mean for CR
            mu_cr = (1 - c) * mu_cr + c * np.mean(scr)

    return best_fitness
