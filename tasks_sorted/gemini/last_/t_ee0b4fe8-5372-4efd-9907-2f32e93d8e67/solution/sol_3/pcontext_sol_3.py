#Here is a fully operational, self-contained Python implementation of the **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution)** algorithm.
#
#### Improvements over the previous version:
#1.  **Linear Population Reduction (LPR):** Instead of a fixed population size, this algorithm starts with a large population to explore the global space and linearly reduces the population size as time progresses. This forces the algorithm to shift from exploration to exploitation naturally, fitting the convergence curve to the allowed time.
#2.  **Dynamic Budget Estimation:** It estimates the maximum number of function evaluations (`MaxFES`) possible within `max_time` based on the timing of the initial population. This allows the LPR mechanism to work correctly with a time constraint rather than a fixed generation count.
#3.  **Improved Bound Handling:** Instead of simple clipping (which causes populations to stick to the edges), it uses a midpoint target between the parent and the bound, preserving diversity.
#4.  **Weighted Lehmer Mean:** A more mathematically robust method for updating the memory of successful control parameters ($F$ and $Cr$).
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History based Adaptive Differential Evolution).
    Adapted for time-constrained optimization.
    """
    # --- Initialization Timer ---
    start_time = time.time()
    
    # --- 1. Parameter Setup ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    
    # L-SHADE Configuration
    # Initial population size: Start large for exploration (e.g., 18 * dim)
    # but cap it to avoid excessive initial overhead on high dims
    N_init = int(round(max(20, min(300, 18 * dim))))
    N_min = 4  # Minimum population size allowed
    
    # Current Population Size
    N = N_init 
    
    # External Archive size (A) matches initial population size
    # In L-SHADE, archive size linearly decreases, but keeping it fixed relative to N 
    # or just fixed capacity is a common variation. We use capacity = N_init.
    archive_capacity = N_init 
    archive = [] # Stores inferior solutions
    
    # Memory for adaptive parameters (History length H)
    H = 6 
    mem_M_F = np.full(H, 0.5)  # Memory for Mutation Factor F
    mem_M_CR = np.full(H, 0.5) # Memory for Crossover Rate CR
    k_mem = 0                  # Memory index pointer

    # --- 2. Initialization ---
    pop = np.zeros((N, dim))
    # Random initialization within bounds
    for d in range(dim):
        pop[:, d] = np.random.uniform(min_b[d], max_b[d], N)
        
    fitness = np.zeros(N)
    
    # Initial Evaluation and Timer Calibration
    # We evaluate the initial population and measure how long it takes 
    # to estimate the total budget (MaxFES).
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    eval_start = time.time()
    for i in range(N):
        # Time Check
        if (time.time() - start_time) >= max_time:
            return best_val if best_val != float('inf') else func(pop[i])
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    eval_duration = time.time() - eval_start
    
    # Current Function Evaluations (FES)
    fes = N 
    
    # Estimate MaxFES based on available time
    # Safety margin: 95% of time to account for logic overhead
    avg_time_per_eval = eval_duration / N
    if avg_time_per_eval <= 0: avg_time_per_eval = 1e-9
    
    estimated_max_fes = int((max_time * 0.95) / avg_time_per_eval)
    max_fes = max(estimated_max_fes, N + 100) # Ensure at least some iterations run

    # --- 3. Main Loop ---
    while True:
        # Strict Time Check
        current_time = time.time()
        if (current_time - start_time) >= max_time:
            return best_val

        # --- Linear Population Size Reduction (LPSR) ---
        # Calculate allowed population size based on FES progress
        # N_next = round(( (N_min - N_init) / max_fes ) * fes + N_init )
        
        # We use time-based progress or FES-based progress. 
        # FES is safer for L-SHADE logic.
        progress = fes / max_fes
        N_next = int(round( (N_min - N_init) * progress + N_init ))
        N_next = max(N_min, N_next)
        
        # If reduction is needed
        if N > N_next:
            # Sort by fitness (worst at the end)
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Truncate population (remove worst)
            remove_count = N - N_next
            pop = pop[:N_next]
            fitness = fitness[:N_next]
            
            # Archive resizing: L-SHADE usually resizes archive to match pop size 
            # or a fixed ratio. Here we maintain capacity but could prune if needed.
            if len(archive) > archive_capacity:
                # Remove random elements if archive exceeds fixed capacity
                del_indices = np.random.choice(len(archive), len(archive) - archive_capacity, replace=False)
                # List comprehension is safer for deletion
                archive = [archive[i] for i in range(len(archive)) if i not in del_indices]
            
            N = N_next

        # --- Parameter Generation ---
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, N)
        mu_F = mem_M_F[r_idx]
        mu_CR = mem_M_CR[r_idx]
        
        # Generate CR: Normal(mu_CR, 0.1), clipped [0, 1]
        # If memory is -1, it means 'unused', but we init at 0.5.
        CR = np.random.normal(mu_CR, 0.1)
        CR = np.clip(CR, 0, 1)
        # Ensure CR is not negative (if normal tail goes <0, clip handles it)
        
        # Generate F: Cauchy(mu_F, 0.1)
        # If F > 1 -> 1. If F <= 0 -> Regenerate.
        F = np.zeros(N)
        for i in range(N):
            while True:
                f_val = mu_F[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                if f_val <= 0:
                    continue
                if f_val > 1:
                    f_val = 1.0
                F[i] = f_val
                break

        # --- Mutation: current-to-pbest/1 ---
        # Sort indices to find p-best
        sorted_indices = np.argsort(fitness)
        
        # p-best selection (top p%)
        # p is randomized between 2/N and 0.2
        # Standard L-SHADE often uses fixed top 11% or random range. 
        # We use a robust default of top 11% (p=0.11)
        p = 0.11
        num_pbest = max(2, int(round(N * p)))
        pbest_indices = sorted_indices[:num_pbest]
        
        # Vectors needed
        pbest_idx = np.random.choice(pbest_indices, N)
        x_pbest = pop[pbest_idx]
        
        # r1: Random distinct from i
        r1_indices = np.random.randint(0, N, N)
        for i in range(N):
            if r1_indices[i] == i:
                r1_indices[i] = (r1_indices[i] + 1) % N
        x_r1 = pop[r1_indices]
        
        # r2: Random distinct from i and r1, chosen from Union(Pop, Archive)
        if len(archive) > 0:
            arr_archive = np.array(archive)
            pop_all = np.vstack((pop, arr_archive))
        else:
            pop_all = pop
            
        len_all = len(pop_all)
        r2_indices = np.random.randint(0, len_all, N)
        
        # Correction for r2 indices to ensure distinctness
        for i in range(N):
            while (r2_indices[i] < N and r2_indices[i] == i) or r2_indices[i] == r1_indices[i]:
                r2_indices[i] = np.random.randint(0, len_all)
                
        x_r2 = pop_all[r2_indices]
        
        # Calculate Mutant
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        diff_pbest = x_pbest - pop
        diff_r1_r2 = x_r1 - x_r2
        F_col = F.reshape(-1, 1)
        
        mutant = pop + F_col * diff_pbest + F_col * diff_r1_r2
        
        # --- Crossover (Binomial) ---
        mask_rand = np.random.rand(N, dim)
        mask_cr = mask_rand <= CR.reshape(-1, 1)
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, N)
        for i in range(N):
            mask_cr[i, j_rand[i]] = True
            
        trial = np.where(mask_cr, mutant, pop)
        
        # --- Bound Handling (Midpoint Back) ---
        # Instead of clipping to bound, set value to (bound + parent)/2
        # This preserves diversity better than sticking to the wall.
        
        # Check Lower
        mask_lower = trial < min_b
        # Use safe arithmetic to avoid shape issues
        if np.any(mask_lower):
            trial[mask_lower] = (pop[mask_lower] + min_b[np.tile(min_b, (N, 1))[mask_lower]]) / 2.0
            
        # Check Upper
        mask_upper = trial > max_b
        if np.any(mask_upper):
            trial[mask_upper] = (pop[mask_upper] + max_b[np.tile(max_b, (N, 1))[mask_upper]]) / 2.0
            
        # Final clip just in case midpoint calculation failed due to precision
        trial = np.clip(trial, min_b, max_b)

        # --- Selection & Memory Update Preparations ---
        winning_S_F = []
        winning_S_CR = []
        winning_diff = []
        
        # Evaluation
        for i in range(N):
            # Granular time check inside loop
            if (time.time() - start_time) >= max_time:
                return best_val
            
            f_trial = func(trial[i])
            fes += 1
            
            if f_trial <= fitness[i]:
                # Improvement or equal
                if f_trial < fitness[i]:
                    # Archive Update: Add parent to archive
                    archive.append(pop[i].copy())
                    
                    # Store successful parameters
                    winning_S_F.append(F[i])
                    winning_S_CR.append(CR[i])
                    winning_diff.append(fitness[i] - f_trial)
                
                # Selection
                pop[i] = trial[i]
                fitness[i] = f_trial
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial[i].copy()
                    
        # Maintain Archive Size (Random removal)
        while len(archive) > archive_capacity:
            idx_rem = np.random.randint(0, len(archive))
            archive.pop(idx_rem)
            
        # --- Memory Update (Weighted Lehmer Mean) ---
        if len(winning_diff) > 0:
            w_diff = np.array(winning_diff)
            w_F = np.array(winning_S_F)
            w_CR = np.array(winning_S_CR)
            
            # Normalize weights
            total_diff = np.sum(w_diff)
            weights = w_diff / total_diff
            
            # Update mean CR (Weighted Arithmetic Mean)
            # If max(CR) is 0 (unlikely), handle gracefully
            mean_CR = np.sum(weights * w_CR)
            # Clamp to memory just in case
            mem_M_CR[k_mem] = np.clip(mean_CR, 0, 1)
            
            # Update mean F (Weighted Lehmer Mean)
            # mean = sum(w * F^2) / sum(w * F)
            numer_F = np.sum(weights * (w_F ** 2))
            denom_F = np.sum(weights * w_F)
            if denom_F > 0:
                mean_F = numer_F / denom_F
                mem_M_F[k_mem] = np.clip(mean_F, 0, 1)
            else:
                mem_M_F[k_mem] = 0.5
                
            # Increment memory pointer
            k_mem = (k_mem + 1) % H
            
        # --- Local Search / Restart Logic (Optional Safety) ---
        # If population has converged extremely (std dev ~ 0) but time remains,
        # perform a small perturbation (chaotic local search) on the best
        # to squeeze out remaining precision.
        std_fit = np.std(fitness)
        if std_fit < 1e-12 and N > 5:
             # Perturb random dimensions of the best vector
             # This prevents the algorithm from idling doing nothing when converged
             idx_perturb = np.random.randint(0, N)
             # Don't overwrite the actual best index in population if it holds the global best
             if fitness[idx_perturb] != best_val:
                 scale = 1e-5 * (max_b - min_b)
                 perturbation = np.random.normal(0, 1, dim) * scale
                 new_candidate = best_vec + perturbation
                 new_candidate = np.clip(new_candidate, min_b, max_b)
                 
                 # Eval
                 if (time.time() - start_time) < max_time:
                     f_new = func(new_candidate)
                     fes += 1
                     if f_new < best_val:
                         best_val = f_new
                         best_vec = new_candidate
                         # Insert into pop
                         pop[idx_perturb] = new_candidate
                         fitness[idx_perturb] = f_new

    return best_val
