#The following is an improved algorithm implementing **SHADE (Success-History based Adaptive Differential Evolution)**.
#
#### Improvements Explanation:
#1.  **Adaptive Parameter History**: Unlike the previous algorithm (jDE) which adapted parameters per individual, SHADE maintains a history memory ($M_{CR}, M_F$) of successful parameter values. This allows the algorithm to learn the best mutation factor ($F$) and crossover rate ($CR$) for the specific function landscape over time.
#2.  **Current-to-pBest Mutation**: This strategy ($V = X_i + F(X_{pbest} - X_i) + F(X_{r1} - X_{r2})$) guides the search towards the top $p\%$ of best individuals rather than just the single best. This prevents premature convergence to local optima while maintaining selection pressure.
#3.  **External Archive**: An archive stores recently discarded inferior solutions. These are used in the mutation step (as $X_{r2}$) to maintain diversity in the difference vectors, ensuring the population doesn't lose exploration power even as it converges.
#4.  **Stagnation Detection & Restart**: Retains the safety net of restarting the population (keeping only the best solution) if the fitness variance drops near zero, ensuring the algorithm doesn't waste time in a local optimum.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    with Restart mechanism for robust global optimization within a time limit.
    """
    # --- Timing Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Standard for SHADE is roughly 18*dim, but we cap it 
    # to ensure sufficient generations run within max_time.
    pop_size = 18 * dim
    if pop_size > 100: pop_size = 100
    if pop_size < 30: pop_size = 30
    
    # SHADE Parameters
    H = 5               # History memory size
    M_cr = np.full(H, 0.5) # Memory for Crossover Rate (init at 0.5)
    M_f = np.full(H, 0.5)  # Memory for Scaling Factor (init at 0.5)
    k_mem = 0           # Memory index pointer
    p_best_rate = 0.11  # Top 11% individuals used for 'current-to-pbest'
    
    # Archive settings (stores inferior solutions to maintain diversity)
    archive = []
    archive_size = int(pop_size * 1.4)
    
    # Bounds processing
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_sol = None
    
    # Evaluate initial population
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_val if best_val != float('inf') else float('inf')
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Strict time check at start of generation
        if (datetime.now() - start_time) >= time_limit:
            return best_val
            
        # 1. Stagnation Check / Restart Mechanism
        # If population variance is negligible, we are likely stuck in a local optimum.
        # Action: Keep global best, randomize the rest, reset history and archive.
        if np.std(fitness) < 1e-9:
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol # Elitism: Keep the best found so far
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Reset SHADE adaptive components
            archive = []
            M_cr = np.full(H, 0.5)
            M_f = np.full(H, 0.5)
            k_mem = 0
            
            # Re-evaluate the new random population (skipping index 0)
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = pop[i].copy()
            continue # Restart generation loop immediately

        # 2. Parameter Generation based on Memory
        # Assign a random memory index to each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = M_cr[r_idx]
        m_f = M_f[r_idx]
        
        # Generate CR (Normal dist, clipped [0, 1])
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy dist)
        # If F > 1 -> 1. If F <= 0 -> regenerate.
        f = np.zeros(pop_size)
        for i in range(pop_size):
            while True:
                val = m_f[i] + 0.1 * np.random.standard_cauchy()
                if val > 0:
                    if val > 1: val = 1.0
                    f[i] = val
                    break
                    
        # 3. Evolution Step
        # Identify top p-best individuals
        sorted_indices = np.argsort(fitness)
        num_p_best = max(2, int(pop_size * p_best_rate))
        top_p_indices = sorted_indices[:num_p_best]
        
        new_pop = np.zeros_like(pop)
        new_fitness = np.zeros(pop_size)
        
        # Track successful parameters for memory update
        scr = [] # Successful CR
        sf = []  # Successful F
        df = []  # Fitness improvement magnitude
        
        for i in range(pop_size):
            # Check time before every evaluation
            if (datetime.now() - start_time) >= time_limit:
                return best_val
                
            # --- Mutation: current-to-pbest/1 ---
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Select p-best index
            p_idx = np.random.choice(top_p_indices)
            x_pbest = pop[p_idx]
            
            # Select r1 distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = pop[r1]
            
            # Select r2 distinct from i and r1
            # r2 is drawn from Union(Population, Archive)
            len_combined = pop_size + len(archive)
            r2 = np.random.randint(0, len_combined)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, len_combined)
                
            if r2 < pop_size:
                x_r2 = pop[r2]
            else:
                x_r2 = archive[r2 - pop_size]
                
            # Create mutant vector
            mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < cr[i]
            mask[j_rand] = True # Ensure at least one parameter comes from mutant
            
            trial = np.where(mask, mutant, pop[i])
            
            # --- Boundary Constraints ---
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                # Successful update
                new_pop[i] = trial
                new_fitness[i] = f_trial
                
                # Record success info
                scr.append(cr[i])
                sf.append(f[i])
                df.append(fitness[i] - f_trial)
                
                # Add old parent to archive
                archive.append(pop[i].copy())
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_sol = trial.copy()
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]
        
        # Update population arrays
        pop = new_pop
        fitness = new_fitness
        
        # 4. Archive Maintenance
        # If archive exceeds size, remove random elements
        if len(archive) > archive_size:
            num_to_remove = len(archive) - archive_size
            idxs_to_remove = np.random.choice(len(archive), num_to_remove, replace=False)
            # Sort descending to delete by index correctly
            idxs_to_remove.sort()
            for idx in idxs_to_remove[::-1]:
                del archive[idx]
                
        # 5. Memory Update (Lehmer Mean)
        # If there were successful mutations, update M_cr and M_f
        if len(scr) > 0:
            scr = np.array(scr)
            sf = np.array(sf)
            df = np.array(df)
            
            # Weights based on fitness improvement
            if np.sum(df) > 0:
                weights = df / np.sum(df)
                
                # Weighted Lehmer Mean for F: sum(w * f^2) / sum(w * f)
                mean_sf = np.sum(weights * (sf ** 2)) / np.sum(weights * sf)
                
                # Weighted Arithmetic Mean for CR
                mean_scr = np.sum(weights * scr)
                
                # Update memory slot
                M_cr[k_mem] = mean_scr
                M_f[k_mem] = mean_sf
                
                # Increment memory index
                k_mem = (k_mem + 1) % H

    return best_val
