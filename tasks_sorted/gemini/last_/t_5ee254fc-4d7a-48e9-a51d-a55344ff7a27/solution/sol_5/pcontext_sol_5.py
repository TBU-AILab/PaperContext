#Here is the improved algorithm. It implements **L-SHADE-RSP (Rank-based Selective Pressure) with Restarts and MTS-LS1 Local Search**. 
#
#This approach improves upon the previous one by:
#1.  **Restart Mechanism**: Instead of getting stuck in a single local optimum (likely the cause of the result ~22), the algorithm detects stagnation (low population variance or no improvement) and restarts the population while preserving the global best.
#2.  **MTS-LS1 Local Search**: Replaces the simple coordinate descent with "Multiple Trajectory Search - Local Search 1". This uses a dynamic search range for each dimension, allowing it to efficiently drill down into narrow valleys or ridges where simple random mutations fail.
#3.  **Linear Population Reduction**: The population size decreases over time to balance exploration (early) and exploitation (late).
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Algorithm: L-SHADE with Restarts and MTS-LS1 Local Search.
    
    Features:
    - Global Search: L-SHADE (Adaptive DE with Linear Population Reduction)
    - Local Search: MTS-LS1 (applied to the best individual to refine solutions)
    - Restarts: Triggered by population convergence or stagnation to escape local optima.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    def get_time_ratio():
        elapsed = (datetime.now() - start_time).total_seconds()
        return elapsed / max_time

    # --- Problem Context ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracker ---
    best_global_val = float('inf')
    best_global_vec = None
    
    # --- L-SHADE Configuration ---
    # Initial population size
    r_N_init = 18
    N_init = int(r_N_init * dim)
    N_init = np.clip(N_init, 30, 200) # Cap to ensure speed
    N_min = 4
    
    # Memory size for adaptive parameters
    H = 6 
    
    # --- Main Restart Loop ---
    # The algorithm will restart the population if it converges or stagnates
    while not check_timeout():
        
        # 1. Initialize Population
        # Scale N based on remaining time? Standard L-SHADE uses N_init.
        # We will use N_init but reduce it linearly towards N_min based on global time.
        
        tr_init = get_time_ratio()
        current_N_init = int(N_init - (N_init - N_min) * tr_init)
        current_N_init = max(current_N_init, 10) # Minimum startup size
        
        N = current_N_init
        pop = min_b + np.random.rand(N, dim) * diff_b
        
        # Soft Restart: Inject the global best if it exists
        if best_global_vec is not None:
            pop[0] = best_global_vec.copy()
            # Inject a few mutants of the best to explore its vicinity
            for k in range(1, min(N, 4)):
                pop[k] = best_global_vec + np.random.normal(0, 0.01, dim) * diff_b
                pop[k] = np.clip(pop[k], min_b, max_b)

        # Evaluation
        fit = np.full(N, float('inf'))
        for i in range(N):
            if check_timeout(): return best_global_val
            fit[i] = func(pop[i])
            if fit[i] < best_global_val:
                best_global_val = fit[i]
                best_global_vec = pop[i].copy()
        
        # Memories for L-SHADE
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Local Search State (MTS-LS1)
        # Search Range initialization (0.4 of domain)
        SR = diff_b * 0.4
        
        # Stagnation counters
        gens_no_improve = 0
        last_best_fit = np.min(fit)
        
        # --- Inner Optimization Loop ---
        while not check_timeout():
            
            # Sort population (needed for rank-based mutation and reduction)
            sort_idx = np.argsort(fit)
            pop = pop[sort_idx]
            fit = fit[sort_idx]
            
            # --- MTS-LS1 Local Search ---
            # Apply periodically or if improving to refine the basin
            do_ls = False
            # Condition: If we just found a new global best, or periodically
            if fit[0] <= best_global_val:
                do_ls = True
            elif gens_no_improve > 0 and gens_no_improve % 10 == 0:
                do_ls = True
            
            if do_ls:
                ls_idx = 0 # Perform on the best
                ls_vec = pop[ls_idx].copy()
                ls_val = fit[ls_idx]
                improved_ls = False
                
                # In high dimensions, shuffle and limit LS budget
                search_dims = list(range(dim))
                if dim > 30:
                    random.shuffle(search_dims)
                    search_dims = search_dims[:30]
                
                for d in search_dims:
                    if check_timeout(): break
                    
                    # MTS-LS1 Logic
                    # Try Negative Step
                    original_val_d = ls_vec[d]
                    ls_vec[d] = np.clip(ls_vec[d] - SR[d], min_b[d], max_b[d])
                    val_new = func(ls_vec)
                    
                    if val_new < ls_val:
                        ls_val = val_new
                        fit[ls_idx] = ls_val
                        pop[ls_idx, d] = ls_vec[d]
                        SR[d] *= 1.5 # Expand search range
                        improved_ls = True
                        if ls_val < best_global_val:
                            best_global_val = ls_val
                            best_global_vec = ls_vec.copy()
                    else:
                        # Try Positive Step (0.5 size)
                        ls_vec[d] = original_val_d # Restore
                        ls_vec[d] = np.clip(ls_vec[d] + 0.5 * SR[d], min_b[d], max_b[d])
                        val_new = func(ls_vec)
                        
                        if val_new < ls_val:
                            ls_val = val_new
                            fit[ls_idx] = ls_val
                            pop[ls_idx, d] = ls_vec[d]
                            SR[d] *= 1.5
                            improved_ls = True
                            if ls_val < best_global_val:
                                best_global_val = ls_val
                                best_global_vec = ls_vec.copy()
                        else:
                            ls_vec[d] = original_val_d # Restore
                            SR[d] *= 0.5 # Shrink search range
                
                if improved_ls:
                    gens_no_improve = 0

            if check_timeout(): return best_global_val

            # --- Population Size Reduction (Linear) ---
            # Target N based on global time ratio
            tr = get_time_ratio()
            N_target = int(round(N_init + (N_min - N_init) * tr))
            N_target = max(N_min, N_target)
            
            if N > N_target:
                N = N_target
                # Pop is sorted, truncate worst
                pop = pop[:N]
                fit = fit[:N]
                # Resize archive
                max_arc = int(N * 2.0)
                if len(archive) > max_arc:
                    random.shuffle(archive)
                    archive = archive[:max_arc]
            
            # --- L-SHADE Generation Steps ---
            
            # 1. Parameter Adaptation
            r_idx = np.random.randint(0, H, N)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]
            
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            f = m_f + 0.1 * np.random.standard_cauchy(N)
            f[f > 1.0] = 1.0
            while np.any(f <= 0):
                neg = f <= 0
                f[neg] = m_f[neg] + 0.1 * np.random.standard_cauchy(np.sum(neg))
                f[f > 1.0] = 1.0
            
            # 2. Mutation: current-to-pbest/1
            # p-best selection
            p_val = np.random.uniform(2/N, 0.2)
            p_top = int(max(1, p_val * N))
            pbest_idx = np.random.randint(0, p_top, N)
            x_pbest = pop[pbest_idx]
            
            # r1 != i
            r1 = np.random.randint(0, N, N)
            for i in range(N):
                while r1[i] == i: r1[i] = np.random.randint(0, N)
            x_r1 = pop[r1]
            
            # r2 from Union(Pop, Archive)
            arc_arr = np.array(archive) if len(archive) > 0 else np.empty((0, dim))
            if len(arc_arr) > 0:
                union_pop = np.vstack((pop, arc_arr))
            else:
                union_pop = pop
            
            r2 = np.random.randint(0, len(union_pop), N)
            for i in range(N):
                while r2[i] == i or r2[i] == r1[i]: 
                    r2[i] = np.random.randint(0, len(union_pop))
            x_r2 = union_pop[r2]
            
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, N)
            mask = np.random.rand(N, dim) < cr[:, None]
            mask[np.arange(N), j_rand] = True
            trial = np.where(mask, mutant, pop)
            
            # 4. Bound Handling (Reflection)
            under = trial < min_b
            if np.any(under):
                trial[under] = 2*min_b[np.where(under)[1]] - trial[under]
            over = trial > max_b
            if np.any(over):
                trial[over] = 2*max_b[np.where(over)[1]] - trial[over]
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection
            new_pop = pop.copy()
            new_fit = fit.copy()
            succ_f = []
            succ_cr = []
            diff_f = []
            
            for i in range(N):
                if check_timeout(): return best_global_val
                val = func(trial[i])
                
                if val <= fit[i]:
                    new_pop[i] = trial[i]
                    new_fit[i] = val
                    if val < fit[i]:
                        archive.append(pop[i].copy())
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diff_f.append(fit[i] - val)
                        
                    if val < best_global_val:
                        best_global_val = val
                        best_global_vec = trial[i].copy()
            
            pop = new_pop
            fit = new_fit
            
            # Archive Maintenance
            if len(archive) > int(N * 2.0):
                random.shuffle(archive)
                archive = archive[:int(N * 2.0)]
                
            # 6. Memory Update (Weighted Lehmer Mean)
            if len(succ_f) > 0:
                succ_f = np.array(succ_f)
                succ_cr = np.array(succ_cr)
                diff_f = np.array(diff_f)
                
                weights = diff_f / np.sum(diff_f)
                
                m_cr_new = np.sum(weights * succ_cr)
                num_f = np.sum(weights * (succ_f**2))
                den_f = np.sum(weights * succ_f)
                m_f_new = num_f / den_f if den_f > 0 else 0.5
                
                M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * m_cr_new
                M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * m_f_new
                k_mem = (k_mem + 1) % H
            
            # --- Convergence/Restart Check ---
            curr_best = np.min(fit)
            if curr_best < last_best_fit:
                last_best_fit = curr_best
                gens_no_improve = 0
            else:
                gens_no_improve += 1
                
            # Triggers: 
            # 1. Population variance is very low (Converged)
            # 2. No improvement for significant duration (Stagnant)
            
            std_fit = np.std(fit)
            is_converged = (std_fit < 1e-9) and (np.max(fit) - np.min(fit) < 1e-8)
            
            # Restart if converged or stagnant, provided there is enough time left (10%)
            # to make a restart meaningful.
            if (is_converged or gens_no_improve > 150) and get_time_ratio() < 0.9:
                break # Break inner loop -> triggers re-initialization in outer loop

    return best_global_val
