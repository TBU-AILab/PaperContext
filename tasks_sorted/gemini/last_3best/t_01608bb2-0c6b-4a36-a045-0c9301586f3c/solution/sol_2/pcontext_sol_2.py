#Here is the improved algorithm. It implements **L-SHADE (Success-History based Adaptive Differential Evolution with Linear Population Size Reduction)**, which is a state-of-the-art evolutionary algorithm. 
#
#**Key Improvements:**
#1.  **L-SHADE Strategy**: Utilizes historical memory to adapt mutation ($F$) and crossover ($CR$) parameters specifically for the problem landscape, rather than random or fixed values.
#2.  **Linear Population Size Reduction (LPSR)**: The population size linearly decreases as time progresses. This encourages exploration in the beginning and exploitation (convergence) towards the end of the time budget.
#3.  **Archive Mechanism**: Maintains a history of "good" inferior solutions to maintain diversity and prevent premature convergence.
#4.  **Restart Mechanism**: If the population converges early (variance becomes negligible), the algorithm automatically restarts with a fresh population while preserving the global best, ensuring the full `max_time` is utilized to find the global optimum.
#5.  **Robust Time Management**: Checks time constraints strictly to ensure a result is returned within `max_time`.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration for L-SHADE ---
    # Initial population size: High diversity to start
    # Capped at 200 to ensure responsiveness on slower functions
    pop_size_init = max(20, min(200, 18 * dim))
    pop_size_min = 4 # Minimum population size for mutation operators
    
    # Memory size for parameter adaptation
    H = 5
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Helper Functions ---
    def init_population(n, d):
        return min_b + np.random.rand(n, d) * diff_b

    # Global Best Tracking
    best_val = float('inf')
    
    # --- Main Loop (Restart Mechanism) ---
    # Loop allows restarting if convergence occurs before time limit
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Initialization for new Run
        pop_size = pop_size_init
        population = init_population(pop_size, dim)
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            # Ensure we get at least one valid result before checking time
            if i > 0 and (datetime.now() - start_time) >= time_limit:
                return best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
        
        # Initialize Memory (Historical Success) to 0.5
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k_mem = 0 # Memory index pointer
        
        # Archive to store improved-upon solutions
        archive = []
        
        # --- Evolutionary Cycle ---
        while True:
            # Global Time Check
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            if elapsed_seconds >= max_time:
                return best_val
            
            # 2. Linear Population Size Reduction (LPSR) based on Time
            # Linearly reduce pop_size from init to min based on elapsed time
            progress = elapsed_seconds / max_time
            target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Remove worst individuals to shrink population
                sort_ind = np.argsort(fitness)
                keep_ind = sort_ind[:target_size]
                population = population[keep_ind]
                fitness = fitness[keep_ind]
                pop_size = target_size
                
                # Shrink archive if necessary
                if len(archive) > pop_size:
                    random.shuffle(archive)
                    archive = archive[:pop_size]

            # 3. Parameter Generation (Adaptive)
            # Pick random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            
            # Generate CR ~ Normal(M_CR, 0.1)
            cr = np.random.normal(M_CR[r_idx], 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(M_F, 0.1)
            f = M_F[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Constraint handling for F
            # If F <= 0, regenerate. If F > 1, clip to 1.
            bad_f = f <= 0
            while np.any(bad_f):
                f[bad_f] = M_F[r_idx[bad_f]] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_f)) - 0.5))
                bad_f = f <= 0
            f = np.clip(f, 0, 1)

            # 4. Mutation: current-to-pbest/1
            # Sort to find p-best
            sorted_indices = np.argsort(fitness)
            
            # Select p-best individuals (top p%, where p is random in [2/N, 0.2])
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_count = int(max(2, round(p_val * pop_size)))
            top_indices = sorted_indices[:top_count]
            
            # Indices for mutation vectors
            idx_pbest = np.random.choice(top_indices, pop_size)
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            
            # Fix r1 collisions (r1 != i)
            hit = idx_r1 == np.arange(pop_size)
            idx_r1[hit] = (idx_r1[hit] + 1) % pop_size
            
            # Select r2 from Population U Archive
            if len(archive) > 0:
                union_pop = np.vstack((population, np.array(archive)))
            else:
                union_pop = population
            
            idx_r2 = np.random.randint(0, len(union_pop), pop_size)
            
            # Simple collision fix for r2 (r2 != i and r2 != r1)
            # (Loop is acceptable here as pop_size is small/reducing)
            for k in range(pop_size):
                while idx_r2[k] == k or idx_r2[k] == idx_r1[k]:
                    idx_r2[k] = np.random.randint(0, len(union_pop))
            
            # Vector Calculation
            x = population
            xp = population[idx_pbest]
            xr1 = population[idx_r1]
            xr2 = union_pop[idx_r2]
            
            F_col = f[:, None]
            mutant = x + F_col * (xp - x) + F_col * (xr1 - xr2)
            
            # 5. Crossover (Binomial)
            mask = np.random.rand(pop_size, dim) < cr[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, x)
            
            # Boundary Constraint (Midpoint method often works better than clipping)
            low_viol = trial < min_b
            trial[low_viol] = (x[low_viol] + min_b[np.where(low_viol)[1]]) / 2.0
            high_viol = trial > max_b
            trial[high_viol] = (x[high_viol] + max_b[np.where(high_viol)[1]]) / 2.0
            
            # 6. Selection
            success_diff = []
            success_F = []
            success_CR = []
            
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                
                f_tri = func(trial[i])
                
                if f_tri <= fitness[i]:
                    diff = fitness[i] - f_tri
                    
                    # Add parent to archive before replacement if strictly better
                    if f_tri < fitness[i]:
                        archive.append(population[i].copy())
                    
                    # Update population
                    population[i] = trial[i]
                    fitness[i] = f_tri
                    
                    # Record success for parameter adaptation
                    if diff > 0:
                        success_diff.append(diff)
                        success_F.append(f[i])
                        success_CR.append(cr[i])
                    
                    if f_tri < best_val:
                        best_val = f_tri
            
            # Maintain Archive Size <= Pop Size
            if len(archive) > pop_size:
                random.shuffle(archive)
                archive = archive[:pop_size]
            
            # 7. Memory Update (Weighted Lehmer Mean)
            if len(success_diff) > 0:
                s_diff = np.array(success_diff)
                s_F = np.array(success_F)
                s_CR = np.array(success_CR)
                
                w = s_diff / np.sum(s_diff)
                
                mean_F = np.sum(w * (s_F ** 2)) / np.sum(w * s_F)
                mean_CR = np.sum(w * s_CR)
                
                M_F[k_mem] = mean_F
                M_CR[k_mem] = mean_CR
                k_mem = (k_mem + 1) % H
            
            # 8. Convergence Check (Restart Trigger)
            # If standard deviation is extremely low, restart to escape local optima
            if np.std(fitness) < 1e-8:
                break 
                
    return best_val
