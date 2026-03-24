#Here is a fully operational, self-contained Python algorithm implementing **Restart L-SHADE (L-SHADE-R)**.
#
##### Improvement Logic
#1.  **Restart Mechanism**: The previous algorithm likely converged to a local optimum and stagnated. This version detects convergence (low population variance or small fitness range) and restarts the population while preserving the global best. This allows it to explore multiple basins of attraction within the time limit.
#2.  **Adaptive Budgeting per Epoch**: The algorithm estimates the cost of function evaluations and dynamically allocates a budget (Number of Function Evaluations - NFE) for each restart epoch.
#3.  **Linear Population Size Reduction (LPSR)**: Within each epoch, the population size linearly decays from a high initial value (for exploration) to a small minimum (for exploitation), driven by the epoch's NFE budget.
#4.  **Weighted Lehmer Mean**: Adaptive parameters ($F$ and $Cr$) are updated based on the magnitude of fitness improvements, prioritizing successful parameter values.
#5.  **Robust Bound Handling**: Uses "Midpoint Bounce-Back" to handle constraints, preserving population diversity better than simple clipping.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Restart L-SHADE (L-SHADE-R) Algorithm.
    Features:
    - Linear Population Size Reduction (LPSR)
    - Success-History based Parameter Adaptation (L-SHADE)
    - Automatic Restart upon stagnation or budget exhaustion
    - Adaptive Time Budgeting
    """
    
    # --- Configuration & Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    bound_diff = ub - lb
    
    # Global Best Tracking
    best_fitness = float('inf')
    
    # --- Initial Estimation of Computational Cost ---
    # Run a small batch to estimate 'func' execution time
    n_est = 5
    est_pop = lb + np.random.rand(n_est, dim) * bound_diff
    t_start_est = datetime.now()
    
    for i in range(n_est):
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness 
        val = func(est_pop[i])
        if val < best_fitness:
            best_fitness = val
            
    elapsed = (datetime.now() - t_start_est).total_seconds()
    
    # Avg time per evaluation
    if elapsed > 1e-6:
        avg_eval_time = elapsed / n_est
    else:
        avg_eval_time = 0.0
        
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check global timer
        now = datetime.now()
        if (now - start_time) >= time_limit:
            break
            
        remaining_seconds = (start_time + time_limit - now).total_seconds()
        if remaining_seconds <= 0:
            break
            
        # 1. Budgeting for this Epoch
        # Calculate max NFE we can afford in the remaining time
        if avg_eval_time > 0:
            est_nfe = int(remaining_seconds / avg_eval_time)
            # Use 95% of estimated capacity to be safe against overhead
            max_nfe_epoch = int(est_nfe * 0.95)
        else:
            max_nfe_epoch = 200000 # Default for extremely fast functions
            
        # If budget is too small for a meaningful run, attempt to run anyway 
        # but expect timeout.
        if max_nfe_epoch < dim * 10:
            max_nfe_epoch = dim * 10
            
        # 2. L-SHADE Initialization
        # Initial Pop Size: 18 * dim, clipped to reasonable limits
        N_init = int(round(18 * dim))
        N_init = max(10, min(N_init, 200))
        
        pop_size = N_init
        pop = lb + np.random.rand(pop_size, dim) * bound_diff
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        current_nfe = 0
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            val = func(pop[i])
            fitness[i] = val
            current_nfe += 1
            if val < best_fitness:
                best_fitness = val
                
        # Algorithm State
        H = 6 # Memory size
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        N_min = 4 # Minimum population size
        
        # 3. Epoch Loop
        while current_nfe < max_nfe_epoch:
            # Check Time
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            # --- LPSR: Linear Population Size Reduction ---
            progress = current_nfe / max_nfe_epoch
            target_size = int(round((N_min - N_init) * progress + N_init))
            target_size = max(N_min, target_size)
            
            # Reduce Population
            if pop_size > target_size:
                # Sort by fitness
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices[:target_size]]
                fitness = fitness[sort_indices[:target_size]]
                
                # Resize Archive (maintain size <= pop_size)
                if len(archive) > target_size:
                    random.shuffle(archive)
                    archive = archive[:target_size]
                    
                pop_size = target_size
                
            # Sort population for mutation strategies (best is at index 0)
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            # --- Stagnation Check ---
            # If population converged, restart early to save time
            if pop_size >= 4:
                fitness_range = np.abs(fitness[-1] - fitness[0])
                if fitness_range < 1e-8:
                    break # Break to outer loop -> Restart
            
            # --- Parameter Adaptation ---
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # Cr ~ Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(m_f, 0.1)
            f = np.zeros(pop_size)
            # Vectorized Cauchy with retry for <= 0
            todo_mask = np.ones(pop_size, dtype=bool)
            while np.any(todo_mask):
                n_todo = np.sum(todo_mask)
                f_gen = m_f[todo_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_todo) - 0.5))
                valid = f_gen > 0
                
                # Map back
                idxs = np.where(todo_mask)[0]
                valid_idxs = idxs[valid]
                f[valid_idxs] = np.minimum(f_gen[valid], 1.0)
                todo_mask[valid_idxs] = False
                
            # --- Mutation: current-to-pbest/1 ---
            # p-best selection (top p%)
            p = 0.11 # Typical value for L-SHADE
            p_num = max(1, int(round(p * pop_size)))
            
            # pbest indices
            pbest_idxs = np.random.randint(0, p_num, pop_size)
            x_pbest = pop[pbest_idxs]
            
            # r1 indices (r1 != i)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            conflict = r1_idxs == np.arange(pop_size)
            while np.any(conflict):
                r1_idxs[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
                conflict = r1_idxs == np.arange(pop_size)
            x_r1 = pop[r1_idxs]
            
            # r2 indices (r2 != r1, r2 != i, from Pop U Archive)
            if len(archive) > 0:
                arc_arr = np.array(archive)
                pop_all = np.vstack((pop, arc_arr))
            else:
                pop_all = pop
            
            n_all = len(pop_all)
            r2_idxs = np.random.randint(0, n_all, pop_size)
            
            c1 = r2_idxs == np.arange(pop_size)
            c2 = r2_idxs == r1_idxs
            conflict = c1 | c2
            while np.any(conflict):
                r2_idxs[conflict] = np.random.randint(0, n_all, np.sum(conflict))
                c1 = r2_idxs == np.arange(pop_size)
                c2 = r2_idxs == r1_idxs
                conflict = c1 | c2
            x_r2 = pop_all[r2_idxs]
            
            # Calculate Mutant
            v = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask = np.random.rand(pop_size, dim) < cr[:, None]
            cross_mask[np.arange(pop_size), j_rand] = True
            
            u = np.where(cross_mask, v, pop)
            
            # --- Constraint Handling (Midpoint Bounce-Back) ---
            mask_l = u < lb
            if np.any(mask_l):
                rows, cols = np.where(mask_l)
                u[rows, cols] = (pop[rows, cols] + lb[cols]) / 2.0
                
            mask_u = u > ub
            if np.any(mask_u):
                rows, cols = np.where(mask_u)
                u[rows, cols] = (pop[rows, cols] + ub[cols]) / 2.0
            
            u = np.clip(u, lb, ub)
            
            # --- Evaluation & Selection ---
            new_fitness = np.zeros(pop_size)
            
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_fitness
                
                val = func(u[i])
                new_fitness[i] = val
                current_nfe += 1
                
                if val < best_fitness:
                    best_fitness = val
            
            # Determine Improvements
            # pop/fitness are sorted, so indices match
            better_mask = new_fitness < fitness
            n_better = np.sum(better_mask)
            
            if n_better > 0:
                # Update Archive
                # Parents that are about to be replaced
                replaced_parents = pop[better_mask]
                for rp in replaced_parents:
                    archive.append(rp.copy())
                
                # Maintain Archive Size
                while len(archive) > pop_size:
                    archive.pop(random.randint(0, len(archive)-1))
                    
                # Update Memory (Weighted Lehmer Mean)
                diff = np.abs(fitness[better_mask] - new_fitness[better_mask])
                total_diff = np.sum(diff)
                
                if total_diff > 0:
                    weights = diff / total_diff
                    
                    # F update
                    f_succ = f[better_mask]
                    mean_f_wl = np.sum(weights * (f_succ ** 2)) / np.sum(weights * f_succ)
                    M_f[k_mem] = mean_f_wl
                    
                    # Cr update
                    cr_succ = cr[better_mask]
                    if np.max(cr_succ) == 0:
                        M_cr[k_mem] = 0
                    else:
                        denom = np.sum(weights * cr_succ)
                        if denom > 0:
                            mean_cr_wl = np.sum(weights * (cr_succ ** 2)) / denom
                            M_cr[k_mem] = mean_cr_wl
                        else:
                            M_cr[k_mem] = 0.5
                    
                    k_mem = (k_mem + 1) % H
                
                # Update Population
                pop[better_mask] = u[better_mask]
                fitness[better_mask] = new_fitness[better_mask]
                
            # --- Loop End Check ---
            if pop_size <= N_min:
                break # Restart
                
    return best_fitness
