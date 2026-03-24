#The following algorithm implements **L-SHADE-R (Linear Population Size Reduction Success-History Adaptive Differential Evolution with Restarts)**.
#
#### Key Improvements:
#1.  **Linear Population Size Reduction (LPSR)**: The population size linearly decreases from a large initial size (for exploration) to a small size (for exploitation) over the duration of an epoch. This focuses computational resources where they are most needed as the search progresses.
#2.  **Time-Based Epochs with Restarts**: Instead of a single run, the algorithm manages "epochs". If the population converges early, or the epoch finishes (based on time allocation), it restarts with the best solution preserved (Elitism). This utilizes the full `max_time` effectively, avoiding stagnation in local optima.
#3.  **SHADE Adaptation**: Uses a history memory ($M_F, M_{CR}$) to adapt mutation factor $F$ and crossover rate $CR$ based on successful updates, removing manual parameter tuning.
#4.  **Vectorized Operations**: Extensive use of NumPy vectorization for mutation, crossover, and boundary handling to maximize the number of function evaluations possible within the time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    L-SHADE-R: Linear Population Size Reduction Success-History based 
    Adaptive Differential Evolution with Restarts.
    """
    global_start = time.time()
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Algorithm Hyperparameters ---
    # Population sizing: Start large for exploration, shrink for exploitation
    pop_size_init = int(np.clip(20 * dim, 60, 200))
    pop_size_min = 4
    
    # SHADE Memory Parameters
    H = 5
    
    while True:
        # Check remaining time for this epoch
        now = time.time()
        remaining_time = max_time - (now - global_start)
        
        # Buffer to ensure we have enough time for at least a few evaluations
        if remaining_time < 0.05: 
            return best_val
            
        # --- Epoch Initialization ---
        epoch_start = now
        # We dedicate the remaining time to this epoch
        epoch_dur = remaining_time 
        
        # Initialize Population
        pop_size = pop_size_init
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Inject best known solution (Elitism across restarts)
        if best_sol is not None:
            pop[0] = best_sol.copy()
            
        fit = np.full(pop_size, np.inf)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if time.time() - global_start >= max_time:
                return best_val
            
            # Skip re-evaluation of injected best if unchanged
            if best_sol is not None and i == 0:
                val = best_val
            else:
                val = func(pop[i])
            
            fit[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
        
        # Initialize SHADE Memory
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        
        # Archive Initialization (Dynamic capacity)
        max_arc_size = int(2.0 * pop_size)
        archive = np.zeros((max_arc_size, dim))
        n_arc = 0
        
        # --- Epoch Loop ---
        while True:
            curr_time = time.time()
            if curr_time - global_start >= max_time:
                return best_val
            
            epoch_elapsed = curr_time - epoch_start
            progress = epoch_elapsed / epoch_dur
            
            # Check for epoch completion (timeout of allocated budget)
            if progress >= 1.0:
                break
            
            # 1. Linear Population Size Reduction (LPSR)
            plan_pop = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            plan_pop = max(pop_size_min, plan_pop)
            
            if pop_size > plan_pop:
                # Reduction: Sort and keep best
                sort_idx = np.argsort(fit)
                keep_idx = sort_idx[:plan_pop]
                pop = pop[keep_idx]
                fit = fit[keep_idx]
                
                # Resize Archive: Keep size <= 2.0 * new_pop_size
                target_arc = int(2.0 * plan_pop)
                if n_arc > target_arc:
                    # Random reduction
                    keep_arc = np.random.choice(n_arc, target_arc, replace=False)
                    archive[:target_arc] = archive[keep_arc]
                    n_arc = target_arc
                
                pop_size = plan_pop
                
            # 2. Convergence Check (Early Restart)
            # If population fitness variance is negligible, restart to escape local optimum
            if np.max(fit) - np.min(fit) < 1e-8:
                break
                
            # 3. Parameter Generation (Vectorized)
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Cauchy distribution for F
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Repair F: if <= 0 retry, if > 1 clamp to 1
            while True:
                mask_bad = f <= 0
                if not np.any(mask_bad): break
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.random.standard_cauchy(np.sum(mask_bad))
            f = np.minimum(f, 1.0)
            
            # Normal distribution for CR
            cr = np.random.normal(m_cr, 0.1, pop_size)
            cr = np.clip(cr, 0.0, 1.0)
            
            # 4. Mutation: current-to-pbest/1
            # Sort for pbest selection
            p_sort_idx = np.argsort(fit)
            
            # p value (top percentage) selected randomly in [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2, pop_size)
            n_pbest = (p_val * pop_size).astype(int)
            n_pbest = np.maximum(n_pbest, 2)
            
            # Select pbest indices
            rand_ranks = (np.random.rand(pop_size) * n_pbest).astype(int)
            pbest_ptr = p_sort_idx[rand_ranks]
            x_pbest = pop[pbest_ptr]
            
            # Select r1 (distinct from i)
            r1_ptr = np.random.randint(0, pop_size, pop_size)
            hit_self = r1_ptr == np.arange(pop_size)
            r1_ptr[hit_self] = (r1_ptr[hit_self] + 1) % pop_size
            x_r1 = pop[r1_ptr]
            
            # Select r2 (distinct from i, r1, from Union(Pop, Archive))
            n_total = pop_size + n_arc
            r2_ptr = np.random.randint(0, n_total, pop_size)
            # Note: We rely on DE's robustness to ignore minor collision probabilities for speed
            
            x_r2 = np.zeros((pop_size, dim))
            mask_pop = r2_ptr < pop_size
            x_r2[mask_pop] = pop[r2_ptr[mask_pop]]
            if n_arc > 0:
                mask_arc = ~mask_pop
                if np.any(mask_arc):
                    arc_ptr = r2_ptr[mask_arc] - pop_size
                    # Safety modulo
                    arc_ptr = arc_ptr % n_arc 
                    x_r2[mask_arc] = archive[arc_ptr]
            
            # Compute Mutant Vector V
            v = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross = np.random.rand(pop_size, dim) < cr[:, None]
            mask_cross[np.arange(pop_size), j_rand] = True
            
            u = np.where(mask_cross, v, pop)
            
            # 6. Constraint Handling (Midpoint)
            # If violated, set to average of bound and old value
            mask_l = u < min_b
            mask_h = u > max_b
            u[mask_l] = (min_b[mask_l] + pop[mask_l]) * 0.5
            u[mask_h] = (max_b[mask_h] + pop[mask_h]) * 0.5
            
            # 7. Selection and Update
            new_pop = pop.copy()
            new_fit = fit.copy()
            
            succ_f = []
            succ_cr = []
            diff_fitness = []
            
            limit_arc = int(2.0 * pop_size)
            
            for i in range(pop_size):
                if time.time() - global_start >= max_time:
                    return best_val
                
                f_u = func(u[i])
                
                if f_u <= fit[i]:
                    # Update Archive
                    if n_arc < limit_arc:
                        archive[n_arc] = pop[i].copy()
                        n_arc += 1
                    else:
                        if n_arc > 0:
                            ridx = np.random.randint(0, n_arc)
                            archive[ridx] = pop[i].copy()
                            
                    # Record success
                    if f_u < fit[i]:
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diff_fitness.append(fit[i] - f_u)
                        
                    new_pop[i] = u[i]
                    new_fit[i] = f_u
                    
                    if f_u < best_val:
                        best_val = f_u
                        best_sol = u[i].copy()
            
            pop = new_pop
            fit = new_fit
            
            # 8. Memory Update
            if len(succ_f) > 0:
                sf = np.array(succ_f)
                scr = np.array(succ_cr)
                sdf = np.array(diff_fitness)
                
                # Weighted Lehmer Mean for F
                w = sdf / np.sum(sdf)
                mean_f = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-15)
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                
                # Weighted Mean for CR
                mean_cr = np.sum(w * scr)
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                
                k_mem = (k_mem + 1) % H

    return best_val
