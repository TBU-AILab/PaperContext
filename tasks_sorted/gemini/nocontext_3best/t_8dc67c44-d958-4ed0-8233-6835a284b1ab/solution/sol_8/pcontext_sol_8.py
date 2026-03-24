#The following algorithm implements **L-SHADE-R (Linear Population Size Reduction SHADE with Restarts) enriched with Memetic Local Search**.
#
#**Why this is an improvement:**
#1.  **Time-Based Linear Population Reduction (LPSR)**: Unlike fixed-population algorithms, this method reduces the population size linearly based on the global elapsed time. It starts with a diverse swarm for exploration and shrinks to a small set for aggressive exploitation as the deadline approaches.
#2.  **Memetic Local Search**: Standard DE struggles with the "last mile" of optimization (getting from 3.08 to 0.0). This algorithm integrates a lightweight **Coordinate Descent** step. Whenever a new global best is found, it performs a quick gradient probe on random dimensions. This allows the algorithm to "slide" down the basin of attraction much faster than mutation alone.
#3.  **Adaptive Greediness**: The parameter `p` (controlling selection pressure in `current-to-pbest`) adapts over time. It starts high (diversity) and decreases (greediness), aligning with the population reduction schedule.
#4.  **Resilient Restarts**: If the population converges (stagnates) before time is up, it triggers a restart. Crucially, the restart preserves the global best (elitism) and adapts the population size to the remaining time window.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Restarts and Memetic Local Search.
    
    Key Components:
    - SHADE (Success-History Adaptive Differential Evolution)
    - LPSR (Linear Population Size Reduction) based on time
    - Coordinate Descent (Local Search) on best solution updates
    - Restart mechanism for stagnation
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Use 98% of available time to ensure safe return
    end_time = start_time + timedelta(seconds=max_time * 0.98)
    
    # --- Pre-process Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global State ---
    best_val = np.inf
    best_vec = None
    
    # Helper for time check
    def check_timeout():
        return datetime.now() >= end_time

    # --- Main Optimization Loop (Restarts) ---
    while True:
        if check_timeout(): return best_val
        
        # 1. Initialize Population
        # Adaptive size: larger for high dim, capped for speed
        # We use this N_init as the baseline for reduction
        N_init = int(min(150, max(30, 18 * dim)))
        
        pop_size = N_init
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best from previous runs
        if best_vec is not None:
            population[0] = best_vec.copy()
            
        fitness = np.full(pop_size, np.inf)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_timeout(): return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_vec = population[i].copy()
                
        # --- SHADE Parameters ---
        H = 5  # Memory size
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Local Search Step Size (Coordinate Descent)
        # Start with 5% of domain width
        ls_step = diff_b * 0.05
        
        # --- Evolution Loop ---
        while True:
            if check_timeout(): return best_val
            
            # 2. Time-Based Linear Population Reduction (LPSR)
            # Calculate global progress (0.0 to 1.0)
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = elapsed / max_time
            if progress > 1.0: progress = 1.0
            
            # Calculate target population size for this moment in time
            # Reduces from N_init to 4
            min_pop = 4
            target_pop = int(round(N_init + (min_pop - N_init) * progress))
            target_pop = max(min_pop, target_pop)
            
            # If current size > target, remove worst individuals
            if pop_size > target_pop:
                # Sort by fitness
                sort_indices = np.argsort(fitness)
                # Keep top 'target_pop'
                keep_indices = sort_indices[:target_pop]
                
                population = population[keep_indices]
                fitness = fitness[keep_indices]
                pop_size = target_pop
                
                # Resize archive to match new pop_size
                if len(archive) > pop_size:
                    random.shuffle(archive)
                    archive = archive[:pop_size]

            # 3. Stagnation Check (Trigger Restart)
            # If population variance is negligible, we are stuck. 
            # Restart to explore new areas (keeping best_vec).
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break 

            # 4. Parameter Generation
            # Select random memory slot
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate CR ~ Normal
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            # Handle F <= 0 (regenerate)
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # 5. Mutation: current-to-pbest/1
            # Adaptive p: Starts at 0.2 (diverse), drops to 0.05 (greedy) based on progress
            p_val = 0.2 * (1.0 - progress) + 0.05
            p_count = max(2, int(pop_size * p_val))
            
            # Sort population for p-best selection
            sorted_idx = np.argsort(fitness)
            top_indices = sorted_idx[:p_count]
            
            pbest_idx = np.random.choice(top_indices, pop_size)
            x_pbest = population[pbest_idx]
            
            # Select r1 (distinct from i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Simple collision fix
            cols = (r1_idx == np.arange(pop_size))
            while np.any(cols):
                r1_idx[cols] = np.random.randint(0, pop_size, np.sum(cols))
                cols = (r1_idx == np.arange(pop_size))
            x_r1 = population[r1_idx]
            
            # Select r2 (distinct from i, r1) from Union(Population, Archive)
            if len(archive) > 0:
                union_pop = np.vstack((population, np.array(archive)))
            else:
                union_pop = population
            
            r2_idx = np.random.randint(0, len(union_pop), pop_size)
            cols = (r2_idx == np.arange(pop_size)) | (r2_idx == r1_idx)
            while np.any(cols):
                r2_idx[cols] = np.random.randint(0, len(union_pop), np.sum(cols))
                cols = (r2_idx == np.arange(pop_size)) | (r2_idx == r1_idx)
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # 6. Crossover (Binomial)
            cross_mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, population)
            trial = np.clip(trial, min_b, max_b)
            
            # 7. Selection & Memetic Local Search
            success_f = []
            success_cr = []
            diff_f = []
            
            for i in range(pop_size):
                if check_timeout(): return best_val
                
                new_val = func(trial[i])
                
                if new_val < fitness[i]:
                    # Improved
                    diff = fitness[i] - new_val
                    archive.append(population[i].copy())
                    
                    fitness[i] = new_val
                    population[i] = trial[i]
                    
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_f.append(diff)
                    
                    if new_val < best_val:
                        best_val = new_val
                        best_vec = trial[i].copy()
                        
                        # --- Memetic Step: Coordinate Descent ---
                        # Exploits the gradient at the new best location.
                        # We try moving slightly in one random dimension.
                        # Step size decays with time.
                        
                        # Pick random dimension
                        d_ls = np.random.randint(0, dim)
                        step = ls_step[d_ls] * (1.0 - progress) # Decay step
                        
                        origin = best_vec[d_ls]
                        
                        # Try Positive Move
                        best_vec[d_ls] = np.clip(origin + step, min_b[d_ls], max_b[d_ls])
                        val_ls = func(best_vec)
                        
                        if val_ls < best_val:
                            # Accepted
                            best_val = val_ls
                            fitness[i] = best_val
                            population[i] = best_vec.copy()
                        else:
                            # Try Negative Move
                            best_vec[d_ls] = np.clip(origin - step, min_b[d_ls], max_b[d_ls])
                            val_ls = func(best_vec)
                            
                            if val_ls < best_val:
                                # Accepted
                                best_val = val_ls
                                fitness[i] = best_val
                                population[i] = best_vec.copy()
                            else:
                                # Revert
                                best_vec[d_ls] = origin

            # Cap Archive Size
            while len(archive) > pop_size:
                archive.pop(random.randint(0, len(archive)-1))
            
            # 8. Update Historical Memory (Weighted Lehmer Mean)
            if len(diff_f) > 0:
                w = np.array(diff_f)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w /= w_sum
                    
                    sf = np.array(success_f)
                    scr = np.array(success_cr)
                    
                    mean_f = np.sum(w * (sf**2)) / np.sum(w * sf)
                    mean_cr = np.sum(w * scr)
                    
                    mem_f[k_mem] = mean_f
                    mem_cr[k_mem] = mean_cr
                    k_mem = (k_mem + 1) % H

    return best_val
