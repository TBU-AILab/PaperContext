#An algorithm to find the set of input parameter values that lead to the minimum output value.
#
#This solution implements a **Simplified SHADE (Success-History based Adaptive Differential Evolution) with Archive and Restart Strategy**.
#
#Key improvements over standard algorithms include:
#1.  **Current-to-pbest Mutation**: Moves individuals towards the best solutions found so far while maintaining diversity, converging faster than standard random mutation.
#2.  **Adaptive Parameters (F & CR)**: Automatically tunes the mutation factor and crossover probability based on successful evaluations, adapting to the specific function landscape.
#3.  **Archive**: Stores recent inferior solutions to preserve diversity and prevent premature convergence, a critical component of state-of-the-art DE variants like JADE/SHADE.
#4.  **Restart Mechanism**: Detects stagnation (low population variance) and restarts the population while preserving the global best, ensuring the algorithm uses the full time budget to escape local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using a Simplified SHADE (Adaptive Differential Evolution)
    algorithm with an external archive and restart strategy.
    """
    # --- Initialization & Timing ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # Helper to check remaining time
    def has_time():
        return (datetime.now() - start_time) < time_limit

    # --- Configuration ---
    # Population size: Balance between exploration (high) and speed (low).
    # We use a dynamic size based on dimension, capped to ensure efficiency.
    pop_size = min(60, max(20, 10 * dim))
    
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Archive for historical vectors (preserves diversity)
    archive = []
    
    # Adaptive Parameter Memory (History length H)
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0  # Memory index
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if not has_time():
            return best_fitness if best_fitness != float('inf') else func(pop[i])
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()

    # --- Main Optimization Loop ---
    while has_time():
        
        # 1. Parameter Adaptation
        # Assign each individual a parameter set from history memory
        r_idx = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idx]
        mu_f = mem_f[r_idx]
        
        # Generate CR ~ Normal(mu, 0.1), clipped to [0, 1]
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F ~ Cauchy(mu, 0.1), clipped to [0.1, 1.0]
        # Approximation using standard_cauchy
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        f = np.clip(f, 0.1, 1.0)
        
        # 2. Mutation Strategy: current-to-pbest/1
        # Select pbest from top p% (greedy component)
        p = 0.1
        top_p_cnt = max(1, int(p * pop_size))
        sorted_indices = np.argsort(fitness)
        pbest_indices = sorted_indices[:top_p_cnt]
        
        pbest_choice = np.random.choice(pbest_indices, pop_size)
        x_pbest = pop[pbest_choice]
        
        # Select r1 distinct from i (random component)
        idxs = np.arange(pop_size)
        r1 = (idxs + np.random.randint(1, pop_size, pop_size)) % pop_size
        x_r1 = pop[r1]
        
        # Select r2 from Union(Pop, Archive) distinct from i, r1 (diversity component)
        if len(archive) > 0:
            archive_np = np.array(archive)
            pop_all = np.vstack((pop, archive_np))
        else:
            pop_all = pop
            
        # Randomly select r2
        r2 = np.random.randint(0, len(pop_all), pop_size)
        x_r2 = pop_all[r2]
        
        # Calculate Mutant Vector V
        # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
        f_broad = f[:, np.newaxis]
        mutant = pop + f_broad * (x_pbest - pop) + f_broad * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, np.newaxis]
        
        # Ensure at least one parameter is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[idxs, j_rand] = True 
        
        trial = np.where(cross_mask, mutant, pop)
        
        # Bound Constraints (Clipping)
        trial = np.clip(trial, min_b, max_b)
        
        # 4. Selection & Evaluation
        success_f = []
        success_cr = []
        
        for i in range(pop_size):
            if not has_time():
                return best_fitness
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # If strictly better, move parent to archive
                if f_trial < fitness[i]:
                    archive.append(pop[i].copy())
                    # Limit archive size to population size
                    if len(archive) > pop_size:
                        archive.pop(np.random.randint(0, len(archive)))
                        
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                success_f.append(f[i])
                success_cr.append(cr[i])
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
                    
        # 5. Update Adaptive Memory
        if len(success_f) > 0:
            sf = np.array(success_f)
            scr = np.array(success_cr)
            
            # Lehmer Mean for F (biases towards larger successful F)
            mean_f = np.sum(sf**2) / np.sum(sf)
            # Arithmetic Mean for CR
            mean_cr = np.mean(scr)
            
            mem_f[k_mem] = mean_f
            mem_cr[k_mem] = mean_cr
            k_mem = (k_mem + 1) % H
            
        # 6. Restart Strategy
        # If population variance is extremely low, we are converged.
        # Restart to use remaining time to find potential better optima.
        if np.std(fitness) < 1e-6:
             # Re-initialize population with random values
             pop = min_b + np.random.rand(pop_size, dim) * diff_b
             # Inject the best solution found so far (Elitism)
             pop[0] = best_sol
             fitness[:] = float('inf')
             fitness[0] = best_fitness
             # Clear archive
             archive = []
             
    return best_fitness
