#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution) with Restarts**. 
#
#**Why this is an improvement:**
#1.  **History-Based Adaptation**: Unlike simple jDE (which resets parameters randomly), SHADE learns the optimal control parameters ($F$ and $CR$) from recent successful updates using a historical memory ($M_{CR}, M_F$). This guides the search much more efficiently.
#2.  **Archive Mechanism**: It maintains an archive of inferior solutions recently replaced by better ones. This preserves diversity and provides "good" directions for mutation (`current-to-pbest` with archive), preventing premature convergence.
#3.  **Robust Restart Strategy**: It detects stagnation (low population variance) and restarts the search while preserving the global best found so far. This is crucial for "limited time" scenarios to escape local optima.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    with a Restart mechanism.
    """
    start_time = datetime.now()
    # Use 98% of max_time to ensure we return before timeout
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    # --- Pre-process Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Tracking ---
    best_val = np.inf
    best_vec = None

    # --- SHADE Hyperparameters ---
    H = 5  # Size of the historical memory
    
    # --- Main Loop (Restarts) ---
    while True:
        # Check if we have enough time to start a meaningful run (e.g., > 5% or > 0.5s)
        remaining = (end_time - datetime.now()).total_seconds()
        if remaining < max(0.5, max_time * 0.05):
            return best_val

        # 1. Initialization
        # Population size: adaptive to dimension, capped for speed
        pop_size = min(120, max(30, 15 * dim))
        
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, np.inf)
        
        # Initial Evaluation
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_vec = population[i].copy()

        # Initialize Memory (F and CR)
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0  # Memory index pointer
        
        # Archive to store inferior solutions (for diversity)
        archive = []
        
        # 2. Evolution Loop
        while True:
            # Time Check
            if datetime.now() >= end_time:
                return best_val
            
            # Convergence Check (Trigger Restart)
            # If population fitness variance is negligible, we are stuck
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break
                
            # --- Parameter Generation ---
            # Pick random index from memory for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate CR ~ Normal(mean=m_cr, std=0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(loc=m_f, scale=0.1)
            # Cauchy random variable = loc + scale * tan(pi * (rand - 0.5))
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Handle F constraints
            # If F > 1 -> 1. If F <= 0 -> Regenerate
            retry_mask = f <= 0
            while np.any(retry_mask):
                f[retry_mask] = m_f[retry_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(retry_mask)) - 0.5))
                retry_mask = f <= 0
            f = np.minimum(f, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # Identify p-best (top p%)
            sorted_indices = np.argsort(fitness)
            # p is random in [2/pop, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_count = max(2, int(pop_size * p_val))
            top_indices = sorted_indices[:top_count]
            
            # Select pbest, r1, r2
            pbest_ind = np.random.choice(top_indices, pop_size)
            x_pbest = population[pbest_ind]
            
            # Select r1 (distinct from current i)
            r1_ind = np.random.randint(0, pop_size, pop_size)
            # Fix collisions
            cols = (r1_ind == np.arange(pop_size))
            while np.any(cols):
                r1_ind[cols] = np.random.randint(0, pop_size, np.sum(cols))
                cols = (r1_ind == np.arange(pop_size))
            x_r1 = population[r1_ind]
            
            # Select r2 (distinct from i and r1) from Population U Archive
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((population, archive_np))
            else:
                union_pop = population
            
            union_size = len(union_pop)
            r2_ind = np.random.randint(0, union_size, pop_size)
            
            # Fix collisions for r2
            # Collision is only possible if r2 index points to current population range
            cols = (r2_ind == np.arange(pop_size)) | (r2_ind == r1_ind)
            while np.any(cols):
                r2_ind[cols] = np.random.randint(0, union_size, np.sum(cols))
                cols = (r2_ind == np.arange(pop_size)) | (r2_ind == r1_ind)
            x_r2 = union_pop[r2_ind]
            
            # Compute Mutant Vectors
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            cross_mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection and Update ---
            success_f = []
            success_cr = []
            diff_f = []
            
            for i in range(pop_size):
                if datetime.now() >= end_time:
                    return best_val
                
                new_val = func(trial_pop[i])
                
                if new_val < fitness[i]:
                    # Solution improved
                    # Move old solution to archive
                    archive.append(population[i].copy())
                    
                    # Store success data
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_f.append(fitness[i] - new_val)
                    
                    # Update population
                    population[i] = trial_pop[i]
                    fitness[i] = new_val
                    
                    if new_val < best_val:
                        best_val = new_val
                        best_vec = trial_pop[i].copy()
            
            # Maintain Archive Size (Limit to pop_size)
            if len(archive) > pop_size:
                random.shuffle(archive)
                archive = archive[:pop_size]
            
            # --- Memory Update (Historical Learning) ---
            if len(diff_f) > 0:
                weights = np.array(diff_f)
                total_w = np.sum(weights)
                if total_w > 0:
                    weights /= total_w
                    
                    sf = np.array(success_f)
                    scr = np.array(success_cr)
                    
                    # Weighted Lehmer Mean for F
                    mean_f = np.sum(weights * (sf ** 2)) / np.sum(weights * sf)
                    
                    # Weighted Arithmetic Mean for CR
                    mean_cr = np.sum(weights * scr)
                    
                    # Update Memory
                    mem_f[k_mem] = mean_f
                    mem_cr[k_mem] = mean_cr
                    k_mem = (k_mem + 1) % H

    return best_val
