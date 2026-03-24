#Here is the fully operational and self-contained Python algorithm implementing **L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution)** with a restart mechanism.
#
#This algorithm improves upon standard DE and JADE by:
#1.  **Linear Population Size Reduction (LPSR)**: Starts with a larger population to explore the search space and linearly reduces it over time to exploit the best areas, adapting to the `max_time` budget.
#2.  **History-Based Adaptation**: Remembers successful control parameters ($F$ and $CR$) in a history memory to guide future parameter generation.
#3.  **External Archive**: Maintains diversity by storing superior solutions that were replaced, utilizing them in the mutation strategy.
#4.  **Time-Aware Restarts**: If the population converges (stagnates) before time runs out, it restarts the search to explore new basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Time-based Population Reduction and Restarts.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population Sizing
    # L-SHADE suggests N_init = 18 * dim. We clamp it for safety within time limits.
    N_init = int(18 * dim)
    N_init = max(30, min(N_init, 500)) 
    N_min = 6  # Minimum population size
    
    # Memory and Archive
    H = 6           # History memory size
    arc_rate = 2.0  # Archive size relative to population
    
    # Bounds preprocessing
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    best_fitness = float('inf')

    # Helper: Check if time limit is reached
    def time_check():
        return datetime.now() - start_time >= limit

    # --- Main Loop (Restarts) ---
    while not time_check():
        
        # --- Initialization per Restart ---
        pop_size = N_init
        
        # Initialize Population (Uniform Random)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if time_check(): return best_fitness
            val = func(pop[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
        
        # Sort population by fitness
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        
        # Initialize Memory (M_cr, M_f) and Archive
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        archive = [] # Stores solution vectors
        
        # --- L-SHADE Evolution Loop ---
        while not time_check():
            
            # 1. Linear Population Size Reduction (LPSR) based on Time
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = elapsed / max_time
            if progress > 1.0: progress = 1.0
            
            # Calculate target population size
            target_size = int(round((N_min - N_init) * progress + N_init))
            target_size = max(N_min, target_size)
            
            # Reduce Population if needed
            if pop_size > target_size:
                # Pop is sorted at the start of loop/after selection. 
                # Keep the best 'target_size' individuals.
                pop = pop[:target_size]
                fitness = fitness[:target_size]
                pop_size = target_size
                
                # Resize Archive
                max_arc_size = int(pop_size * arc_rate)
                if len(archive) > max_arc_size:
                    # Randomly remove elements
                    keep_idxs = np.random.choice(len(archive), max_arc_size, replace=False)
                    archive = [archive[i] for i in keep_idxs]
            
            # 2. Convergence Check (Trigger Restart)
            # If population variance is negligible or we reached min size and stagnation
            if np.max(fitness) - np.min(fitness) < 1e-8:
                break # Break inner loop to restart
                
            # 3. Generate Parameters (F, CR)
            # Pick random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F: Cauchy(m_f, 0.1)
            f = np.zeros(pop_size)
            for i in range(pop_size):
                while True:
                    # Cauchy distribution
                    val = m_f[i] + 0.1 * np.random.standard_cauchy()
                    if val > 0: # Check lower bound
                        if val > 1: val = 1.0 # Check upper bound
                        f[i] = val
                        break
            
            # 4. Mutation: current-to-pbest/1 with Archive
            # Sort pop to find p-best
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            # Select p-best (top p% individuals)
            p_val = 0.11
            p_best_count = max(2, int(p_val * pop_size))
            pbest_idxs = np.random.randint(0, p_best_count, pop_size)
            xp = pop[pbest_idxs]
            
            # Select r1 (distinct from current i)
            # We shift indices by random offset to ensure r1 != i
            idxs = np.arange(pop_size)
            r1_offsets = np.random.randint(1, pop_size, pop_size)
            r1_idxs = (idxs + r1_offsets) % pop_size
            xr1 = pop[r1_idxs]
            
            # Select r2 (distinct from i and r1, from Population U Archive)
            n_arc = len(archive)
            n_union = pop_size + n_arc
            
            if n_arc > 0:
                union_arr = np.vstack((pop, np.array(archive)))
            else:
                union_arr = pop
                
            r2_idxs = np.random.randint(0, n_union, pop_size)
            
            # Fix r2 collisions (r2 != i and r2 != r1)
            # Fast fix loop
            for i in range(pop_size):
                while r2_idxs[i] == i or r2_idxs[i] == r1_idxs[i]:
                    r2_idxs[i] = np.random.randint(0, n_union)
            
            xr2 = union_arr[r2_idxs]
            
            # Compute Mutant Vectors
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            f_v = f[:, np.newaxis]
            mutant = pop + f_v * (xp - pop) + f_v * (xr1 - xr2)
            
            # 5. Crossover (Binomial)
            rand_mask = np.random.rand(pop_size, dim)
            cross_mask = rand_mask < cr[:, np.newaxis]
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_mask = np.zeros((pop_size, dim), dtype=bool)
            j_mask[np.arange(pop_size), j_rand] = True
            cross_mask = np.logical_or(cross_mask, j_mask)
            
            trial_pop = np.where(cross_mask, mutant, pop)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 6. Selection and Adaptation
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_fit = np.zeros(pop_size)
            
            for i in range(pop_size):
                if time_check(): return best_fitness
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    # Solution improved or same
                    if f_trial < fitness[i]:
                        # Save old solution to archive
                        archive.append(pop[i].copy())
                        succ_mask[i] = True
                        diff_fit[i] = fitness[i] - f_trial
                        
                    new_pop[i] = trial_pop[i]
                    new_fitness[i] = f_trial
                    
                    if f_trial < best_fitness:
                        best_fitness = f_trial
            
            pop = new_pop
            fitness = new_fitness
            
            # Manage Archive Size (Immediate cleanup)
            max_arc_size = int(pop_size * arc_rate)
            while len(archive) > max_arc_size:
                # Remove random solution
                archive.pop(np.random.randint(0, len(archive)))
            
            # 7. Update History Memory
            if np.any(succ_mask):
                s_f = f[succ_mask]
                s_cr = cr[succ_mask]
                df = diff_fit[succ_mask]
                
                # Weighted Lehmer Mean calculation
                weights = df / np.sum(df)
                
                # Mean F (Lehmer)
                mean_f = np.sum(weights * (s_f ** 2)) / (np.sum(weights * s_f) + 1e-9)
                
                # Mean CR (Weighted Arithmetic)
                mean_cr = np.sum(weights * s_cr)
                
                # Update memory at pointer k
                M_f[k_mem] = mean_f
                M_cr[k_mem] = mean_cr
                
                k_mem = (k_mem + 1) % H

    return best_fitness
