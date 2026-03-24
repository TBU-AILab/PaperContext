#The following Python code implements a **JADE (Adaptive Differential Evolution) with Archive and Restart** algorithm.
#
#**Key Improvements over previous algorithms:**
#1.  **Adaptive Parameters (JADE mechanism)**: Instead of fixed `F` and `CR` or the individual-based logic of jDE, this algorithm adapts the distributions of mutation (`F`) and crossover (`CR`) rates using success-based history (Lehmer mean). This converges to optimal parameters for the specific function landscape automatically.
#2.  **`current-to-pbest/1` Strategy**: This mutation strategy guides the search towards the top $p\%$ of best solutions (exploitation) while using difference vectors from the population (exploration). It converges significantly faster than `rand/1`.
#3.  **External Archive**: A historical archive of inferior solutions is maintained to preserve diversity. This allows the difference vectors in the mutation step to reference older positions, preventing premature convergence and enhancing the global search capability.
#4.  **Robust Restart**: The algorithm monitors population variance. If it drops below a threshold (stagnation), it triggers a restart while keeping the best found solution (Elitism). This effectively handles multimodal functions by exploring different basins of attraction within the time limit.
#5.  **Optimized Vectorization**: The implementation uses extensive NumPy vectorization for mutation, crossover, and boundary handling, minimizing the Python interpreter overhead which is critical for the "limited time" constraint.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using JADE (Adaptive Differential Evolution) 
    with External Archive and Restart mechanism.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: clamped to ensure speed on high dimensions while maintaining diversity
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # JADE specific parameters
    c = 0.1             # Learning rate for parameter adaptation
    top_p_ratio = 0.05  # Percentage of top individuals for p-best selection
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    global_best_sol = None

    # Helper for time checking
    def is_time_up():
        return (datetime.now() - start_time) >= limit

    # --- Main Optimization Loop (Restarts) ---
    while not is_time_up():
        
        # 1. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best solution into the new population (if exists)
        start_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_idx = 1
            
        # Evaluate initial population
        for i in range(start_idx, pop_size):
            if is_time_up(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()

        # 2. Initialize Adaptive Parameters & Archive
        mu_cr = 0.5
        mu_f = 0.5
        
        # Archive: stores decent solutions that were replaced
        archive = np.zeros((pop_size, dim))
        num_arc = 0  # Current number of items in archive
        
        no_improve_count = 0
        
        # --- Evolution Loop ---
        while not is_time_up():
            
            # --- A. Parameter Generation ---
            # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(mu_f, 0.1), clipped [0.1, 1.0]
            # Cauchy: loc + scale * tan(pi * (rand - 0.5))
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f = np.clip(f, 0.1, 1.0)
            
            # --- B. Mutation (current-to-pbest/1) ---
            # Sort population to find p-best
            sorted_indices = np.argsort(fitness)
            num_pbest = max(2, int(top_p_ratio * pop_size))
            pbest_indices = sorted_indices[:num_pbest]
            
            # Select random p-best for each individual
            r_pbest = np.random.choice(pbest_indices, pop_size)
            
            # Select r1 (distinct from i)
            # Method: (i + random_shift) % N
            r1 = (np.arange(pop_size) + np.random.randint(1, pop_size, pop_size)) % pop_size
            
            # Select r2 (from Population Union Archive, distinct from i, r1)
            # We construct a virtual pool of vectors
            if num_arc > 0:
                # Combine pop and current archive
                # Note: vstack is relatively expensive, done once per generation
                pop_and_archive = np.vstack((pop, archive[:num_arc]))
                size_union = pop_size + num_arc
                r2 = np.random.randint(0, size_union, pop_size)
                
                # Loose collision handling for speed (relying on large pool size)
                # x_r2 will be selected from this pool
                x_r2 = pop_and_archive[r2]
            else:
                # If archive is empty, r2 comes from pop (distinct from i and r1)
                r2 = (r1 + np.random.randint(1, pop_size, pop_size)) % pop_size
                x_r2 = pop[r2]

            x = pop
            x_pbest = pop[r_pbest]
            x_r1 = pop[r1]
            
            # Calculate Mutant Vector: v = x + F*(pbest - x) + F*(r1 - r2)
            f_col = f[:, np.newaxis]
            mutant = x + f_col * (x_pbest - x) + f_col * (x_r1 - x_r2)
            
            # --- C. Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, np.newaxis]
            
            # Force at least one dimension from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, x)
            
            # --- D. Bound Handling (Reflection) ---
            # Lower bounds
            mask_l = trial < min_b
            if np.any(mask_l):
                # Reflect: 2*min - x
                trial[mask_l] = 2 * min_b[np.where(mask_l)[1]] - trial[mask_l]
                trial = np.maximum(trial, min_b) # Safety clamp
            
            # Upper bounds
            mask_u = trial > max_b
            if np.any(mask_u):
                # Reflect: 2*max - x
                trial[mask_u] = 2 * max_b[np.where(mask_u)[1]] - trial[mask_u]
                trial = np.minimum(trial, max_b) # Safety clamp

            # --- E. Selection & Adaptation ---
            successful_f = []
            successful_cr = []
            improved = False
            
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                trial_val = func(trial[i])
                
                if trial_val < fitness[i]:
                    # Solution improved
                    improved = True
                    
                    # 1. Update Archive
                    # Add parent to archive (before replacing it)
                    if num_arc < pop_size:
                        archive[num_arc] = pop[i]
                        num_arc += 1
                    else:
                        # Randomly replace an existing archive member
                        rand_idx = np.random.randint(0, pop_size)
                        archive[rand_idx] = pop[i]
                    
                    # 2. Update Population
                    pop[i] = trial[i]
                    fitness[i] = trial_val
                    
                    # 3. Record success for parameter adaptation
                    successful_f.append(f[i])
                    successful_cr.append(cr[i])
                    
                    # 4. Update Global Best
                    if trial_val < global_best_val:
                        global_best_val = trial_val
                        global_best_sol = trial[i].copy()
                        no_improve_count = 0
            
            if not improved:
                no_improve_count += 1
                
            # --- F. Update Adaptive Parameters (Lehmer Mean) ---
            if len(successful_f) > 0:
                s_f = np.array(successful_f)
                s_cr = np.array(successful_cr)
                
                # Update mu_cr (Arithmetic Mean)
                mu_cr = (1 - c) * mu_cr + c * np.mean(s_cr)
                
                # Update mu_f (Lehmer Mean)
                sum_f = np.sum(s_f)
                if sum_f > 0:
                    mean_lehmer = np.sum(s_f**2) / sum_f
                    mu_f = (1 - c) * mu_f + c * mean_lehmer
            
            # --- G. Convergence Check / Restart ---
            # If population variance is negligible or no improvement for many generations
            # Stagnation count threshold (30) is heuristic for small populations
            std_dev = np.std(fitness)
            if std_dev < 1e-8 or no_improve_count > 30:
                break # Break inner loop to restart with new population

    return global_best_val
