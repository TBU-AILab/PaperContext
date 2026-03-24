#The goal is to provide a robust, high-performance optimization algorithm that fits within the self-contained template. Based on the previous best results, **SHADE (Success-History based Adaptive Differential Evolution) with a Restart Strategy** is the most promising approach. It combines the rapid convergence of the adaptive `current-to-pbest` mutation strategy with the global search capability ensured by restarts when the population stagnates.
#
#Here is the fully operational Python code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History Adaptive Differential Evolution)
    with a Restart Strategy to maximize the use of the time budget.
    """
    start_time = datetime.now()
    # Use 98% of max_time to ensure safe return before external timeout
    time_limit = timedelta(seconds=max_time * 0.98)
    
    # --- Pre-processing Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    best_fitness = float('inf')
    
    # --- Algorithm Parameters ---
    # Memory size for historical adaptation
    H = 10 
    
    # --- Main Restart Loop ---
    # Restarts the population if it converges/stagnates, preventing it 
    # from wasting time in a local optimum.
    while True:
        # Check time budget at the start of each run
        if datetime.now() - start_time >= time_limit:
            return best_fitness
            
        # Population Size
        # 18 * dim is a robust heuristic for SHADE, but we cap it to ensure speed
        # in higher dimensions within the limited time.
        pop_size = int(np.clip(18 * dim, 30, 150))
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_fitness
            try:
                val = func(pop[i])
            except:
                val = float('inf')
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val

        # SHADE Memory Initialization
        # M_F and M_CR store mean values for successful F and CR parameters
        mem_M_F = np.full(H, 0.5)
        mem_M_CR = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive (stores inferior solutions to preserve diversity)
        archive = []
        
        # --- Generation Loop ---
        while True:
            # Time check
            if datetime.now() - start_time >= time_limit:
                return best_fitness
            
            # 1. Parameter Generation
            # Pick random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_M_F[r_idx]
            m_cr = mem_M_CR[r_idx]
            
            # Generate CR: Normal distribution around memory, clipped [0, 1]
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # Generate F: Cauchy distribution around memory
            # F must be positive. If <= 0, retry. If > 1, clamp to 1.
            F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Vectorized repair for F <= 0
            retry_mask = F <= 0
            retry_count = 0
            while np.any(retry_mask) and retry_count < 5:
                F[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
                retry_mask = F <= 0
                retry_count += 1
            F[F <= 0] = 0.05 # Fallback for stubborn negatives
            F = np.minimum(F, 1.0)
            
            # 2. Mutation: current-to-pbest/1
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Sort population to find p-best
            sorted_indices = np.argsort(fitness)
            
            # Random p in range [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_cnt = int(max(2, p_val * pop_size))
            top_indices = sorted_indices[:top_cnt]
            
            # Select pbest
            pbest_idxs = np.random.choice(top_indices, pop_size)
            x_pbest = pop[pbest_idxs]
            
            # Select r1 (distinct from i)
            idxs = np.arange(pop_size)
            shift = np.random.randint(1, pop_size, pop_size)
            r1_idxs = (idxs + shift) % pop_size
            x_r1 = pop[r1_idxs]
            
            # Select r2 (distinct from i and r1, from Union(Pop, Archive))
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((pop, archive_np))
            else:
                union_pop = pop
            
            union_size = len(union_pop)
            r2_idxs = np.random.randint(0, union_size, pop_size)
            
            # Fix collisions for r2
            # Case r2 == i
            conflict_i = (r2_idxs == idxs)
            r2_idxs[conflict_i] = (r2_idxs[conflict_i] + 1) % union_size
            
            # Case r2 == r1 (only check if r2 is within population bounds)
            in_pop_mask = r2_idxs < pop_size
            conflict_r1 = in_pop_mask & (r2_idxs == r1_idxs)
            r2_idxs[conflict_r1] = (r2_idxs[conflict_r1] + 1) % union_size
            
            x_r2 = union_pop[r2_idxs]
            
            # Calculate Mutant Vector
            F_col = F[:, np.newaxis]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 3. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_jrand = np.zeros((pop_size, dim), dtype=bool)
            mask_jrand[np.arange(pop_size), j_rand] = True
            
            mask_cr = np.random.rand(pop_size, dim) < CR[:, np.newaxis]
            mask_trial = mask_cr | mask_jrand
            
            trial_pop = np.where(mask_trial, mutant, pop)
            
            # 4. Selection and Evaluation
            success_F = []
            success_CR = []
            diff_f = []
            
            for i in range(pop_size):
                # Strict time check inside loop to handle slow functions
                if datetime.now() - start_time >= time_limit:
                    return best_fitness
                
                try:
                    f_trial = func(trial_pop[i])
                except:
                    f_trial = float('inf')
                
                # Greedy selection
                if f_trial <= fitness[i]:
                    # If strictly better, record success info
                    if f_trial < fitness[i]:
                        diff_f.append(fitness[i] - f_trial)
                        success_F.append(F[i])
                        success_CR.append(CR[i])
                        
                        # Move parent to archive
                        archive.append(pop[i].copy())
                    
                    # Update Population
                    pop[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
            
            # 5. Archive Maintenance
            # Maintain archive size <= pop_size
            while len(archive) > pop_size:
                rm_idx = np.random.randint(0, len(archive))
                archive.pop(rm_idx)
                
            # 6. Memory Update (Weighted Lehmer Mean)
            if len(diff_f) > 0:
                diff_f = np.array(diff_f)
                success_F = np.array(success_F)
                success_CR = np.array(success_CR)
                
                total_diff = np.sum(diff_f)
                weights = diff_f / total_diff
                
                # Update Mean F
                sum_sq_f = np.sum(weights * (success_F ** 2))
                sum_f = np.sum(weights * success_F)
                mean_f = sum_sq_f / sum_f if sum_f > 0 else 0.5
                
                # Update Mean CR
                mean_cr = np.sum(weights * success_CR)
                
                mem_M_F[k_mem] = 0.5 * mem_M_F[k_mem] + 0.5 * mean_f
                mem_M_CR[k_mem] = 0.5 * mem_M_CR[k_mem] + 0.5 * mean_cr
                
                k_mem = (k_mem + 1) % H
                
            # 7. Convergence Check
            # If population fitness variance is extremely low, restart to explore elsewhere
            if np.std(fitness) < 1e-6:
                break
                
    return best_fitness
