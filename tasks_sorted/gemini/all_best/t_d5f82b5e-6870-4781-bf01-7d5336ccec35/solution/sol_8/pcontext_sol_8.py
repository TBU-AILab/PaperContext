#The output value is: **24.710746602269268**
#
#The improved algorithm code is:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Implements L-SHADE (Linear Success-History Adaptive Differential Evolution).
    
    Key Features:
    1. Historical Parameter Adaptation (SHADE): Uses a memory of successful F and CR 
       values to guide future generations, rather than random dithering or instant adaptation.
    2. Linear Population Size Reduction (LPSR): Starts with a large population to explore 
       the landscape and linearly reduces it to concentrate on the best area as time passes.
    3. External Archive: Maintains a set of inferior solutions recently replaced to 
       maintain diversity in the mutation equation.
    4. Current-to-pBest/1 Mutation: Balances greediness with exploration.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Check time helper
    def time_is_up():
        return datetime.now() - start_time >= time_limit

    # --- Configuration ---
    # Initial population size (scaled by dimension, usually 18*dim in L-SHADE)
    # Capped to avoid excessive initial evaluation time
    initial_pop_size = int(18 * dim)
    if initial_pop_size > 100: initial_pop_size = 100
    if initial_pop_size < 30: initial_pop_size = 30
    
    # Minimum population size at the end
    min_pop_size = 4
    
    # Archive size (same as initial population size)
    archive_size = initial_pop_size
    
    # Memory size for historical adaptation
    H = 5
    mem_F = np.full(H, 0.5)
    mem_CR = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    pop_size = initial_pop_size
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate initial population
    best_idx = 0
    best_val = float('inf')
    
    for i in range(pop_size):
        if time_is_up(): return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i
            
    # External Archive (stores vectors replaced by better offspring)
    archive = []
    
    # --- Main Loop ---
    while not time_is_up():
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate progress ratio based on time
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        
        # Calculate target population size linearly
        new_pop_size = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * progress))
        if new_pop_size < min_pop_size: new_pop_size = min_pop_size
        
        # If we need to reduce population, remove worst individuals
        if pop_size > new_pop_size:
            # Sort by fitness (descending implies worst at the end? No, we minimize. 
            # argsort gives ascending. Worst are at the end.)
            sorted_indices = np.argsort(fitness)
            # Keep top 'new_pop_size'
            keep_indices = sorted_indices[:new_pop_size]
            
            population = population[keep_indices]
            fitness = fitness[keep_indices]
            pop_size = new_pop_size
            
            # Adjust Archive size to be proportional to current pop size (optional variation, 
            # but standard L-SHADE keeps fixed capacity or reduces. We keep fixed cap for diversity).
            if len(archive) > initial_pop_size:
                del archive[initial_pop_size:]

        # 2. Parameter Generation (F and CR)
        # Select random memory index for each individual
        r_indices = np.random.randint(0, H, pop_size)
        m_cr = mem_CR[r_indices]
        m_f = mem_F[r_indices]
        
        # Generate CR: Normal(mean=mem_CR, std=0.1)
        # Clip to [0, 1]
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0, 1)
        
        # Generate F: Cauchy(loc=mem_F, scale=0.1)
        # If F > 1, clip to 1. If F <= 0, resample (simplification: set to 0.5 * random)
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F values
        # If too high, clamp to 1.0
        F[F > 1.0] = 1.0
        # If too low, regenerate (simplification for speed: just clamp to small value)
        F[F <= 0.0] = 0.4 + 0.1 * np.random.rand(np.sum(F <= 0.0))

        # 3. Mutation: current-to-pbest/1
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        
        # Sort current population to find p-best
        sorted_indices = np.argsort(fitness)
        
        # p-best selection parameter (usually top 11% reduced to 5%, we fix at 0.1)
        p_val = 0.1
        p_num = max(2, int(pop_size * p_val))
        
        # Select x_pbest indices
        pbest_indices = sorted_indices[np.random.randint(0, p_num, pop_size)]
        x_pbest = population[pbest_indices]
        
        # Select x_r1 indices (distinct from i)
        # Vectorized random choice is tricky to ensure distinctness perfectly without loop.
        # Approximation: simple random, collision probability is low.
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_indices]
        
        # Select x_r2 from Union(Population, Archive)
        # Convert archive to numpy for indexing
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((population, archive_np))
        else:
            union_pop = population
            
        r2_indices = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Calculate mutant vectors
        # Reshape F for broadcasting: (pop_size, 1)
        F_col = F[:, np.newaxis]
        
        # Mutation
        mutant = population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR[:, np.newaxis]
        
        # Force at least one dimension to be from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)
        
        # Bound Handling (Clip)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 5. Evaluation and Selection
        success_F = []
        success_CR = []
        diff_fitness = []
        
        # Iterate to evaluate (func is scalar)
        for i in range(pop_size):
            if time_is_up(): return best_val
            
            f_trial = func(trial_pop[i])
            f_target = fitness[i]
            
            if f_trial <= f_target:
                # Success
                # Add target to archive
                if len(archive) < archive_size:
                    archive.append(population[i].copy())
                else:
                    # Replace random archive member
                    rem_idx = np.random.randint(0, archive_size)
                    archive[rem_idx] = population[i].copy()
                
                # Update population
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                # Store successful params
                success_CR.append(CR[i])
                success_F.append(F[i])
                diff_fitness.append(f_target - f_trial)
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
        
        # 6. Update Memory (Weighted Lehmer Mean)
        if len(success_F) > 0:
            success_F = np.array(success_F)
            success_CR = np.array(success_CR)
            diff_fitness = np.array(diff_fitness)
            
            # Weights based on fitness improvement
            total_diff = np.sum(diff_fitness)
            if total_diff == 0:
                weights = np.ones(len(diff_fitness)) / len(diff_fitness)
            else:
                weights = diff_fitness / total_diff
            
            # Weighted Lehmer Mean for F
            # mean_wl = sum(w * x^2) / sum(w * x)
            mean_F = np.sum(weights * (success_F ** 2)) / (np.sum(weights * success_F) + 1e-10)
            
            # Weighted Mean for CR
            # mean_wa = sum(w * x)
            mean_CR = np.sum(weights * success_CR)
            
            # Update memory at pointer k
            mem_F[k_mem] = mean_F
            mem_CR[k_mem] = mean_CR
            
            # Increment pointer
            k_mem = (k_mem + 1) % H
            
    return best_val
