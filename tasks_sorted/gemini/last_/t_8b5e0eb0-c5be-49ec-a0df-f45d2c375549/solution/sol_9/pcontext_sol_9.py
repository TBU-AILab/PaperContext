#Here is the improved algorithm.
#
#**Improvements Explanation:**
#1.  **OBL Initialization**: Uses Opposition-Based Learning during initialization. It generates a population and its opposite (inverse within bounds), evaluates both, and selects the fittest half. This dramatically improves initial coverage and convergence speed.
#2.  **Optimized L-SHADE (jSO-style)**: Incorporates Linear Population Size Reduction (LPSR) and a weighted memory update strategy. The "p" parameter for `current-to-pbest` mutation is dynamic, shifting from exploration (top 20%) to exploitation (top 2%) over time.
#3.  **Partial MTS-LS1**: The Local Search (MTS-LS1) is now applied selectively. Instead of scanning all dimensions (expensive), it selects a random subset (20%) of dimensions to refine the best solution in each iteration. This balances local refinement with the computational budget.
#4.  **Stagnation Restart**: Includes a safety restart mechanism. If the population variance collapses (convergence) before 75% of the time is used, it triggers a soft restart (keeping the best solution) to escape local optima.
#5.  **Robust Time Management**: Checks time limits strictly within loops (initialization, mutation, local search) to ensure the algorithm returns the best result immediately when time runs out.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Improved L-SHADE with Opposition-Based Learning (OBL) initialization 
    and Dimension-Reduced MTS-LS1 Local Search.
    """
    
    # --- Time Management ---
    t_start = datetime.now()
    t_limit = timedelta(seconds=max_time)
    
    def check_time(buffer_sec=0.0):
        """Returns True if there is enough time left minus a safety buffer."""
        return (datetime.now() - t_start) < (t_limit - timedelta(seconds=buffer_sec))

    # --- Setup ---
    rng = np.random.default_rng()
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing
    # Start with a larger population for OBL exploration, reduce linearly
    n_init = int(25 * dim)
    n_init = np.clip(n_init, 50, 300) 
    n_min = 4
    
    pop_size = n_init
    
    # --- OBL Initialization ---
    # 1. Generate random population
    p1 = min_b + rng.random((pop_size, dim)) * diff_b
    
    # 2. Generate opposite population (min + max - x)
    p2 = min_b + max_b - p1
    p2 = np.clip(p2, min_b, max_b)
    
    # 3. Pool to evaluate (Size 2 * N)
    pop_pool = np.vstack((p1, p2))
    fit_pool = np.zeros(2 * pop_size)
    
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate pool with strict time check
    count_eval = 0
    for i in range(2 * pop_size):
        if not check_time(0.01):
            if count_eval == 0: return float('inf') 
            break
        val = func(pop_pool[i])
        fit_pool[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop_pool[i].copy()
        count_eval += 1
            
    # Select best N individuals from the pool
    available_count = min(count_eval, pop_size)
    sorted_idx = np.argsort(fit_pool[:count_eval])
    
    pop = np.zeros((pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    # Fill population
    keep_idx = sorted_idx[:available_count]
    pop[:available_count] = pop_pool[keep_idx]
    fitness[:available_count] = fit_pool[keep_idx]
    
    # Fill remaining spots (if time out occurred early) with random data (unevaluated)
    if available_count < pop_size:
        remain = pop_size - available_count
        pop[available_count:] = min_b + rng.random((remain, dim)) * diff_b
    
    best_idx = 0 # Sorted, so 0 is best
    
    # --- SHADE Memory & Archive ---
    mem_size = 5
    m_cr = np.full(mem_size, 0.5)
    m_f = np.full(mem_size, 0.5)
    mem_k = 0
    
    archive = np.zeros((0, dim))
    arc_rate = 2.0 # Archive size relative to population
    
    # --- MTS-LS1 State ---
    ls_sr = diff_b * 0.4 # Initial search range
    ls_dim_count = max(1, int(dim * 0.2)) # Update 20% of dims per step
    
    # --- Main Optimization Loop ---
    while check_time(0.02):
        
        # 0. Time Progress
        elapsed = (datetime.now() - t_start).total_seconds()
        progress = elapsed / max_time
        
        # 1. Linear Population Size Reduction (LPSR)
        n_next = int(round((n_min - n_init) * progress + n_init))
        n_next = max(n_min, n_next)
        
        if pop_size > n_next:
            # Sort to put best at top
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            # Truncate
            pop = pop[:n_next]
            fitness = fitness[:n_next]
            pop_size = n_next
            
            # Reset best index
            best_idx = 0
            best_val = fitness[0]
            best_sol = pop[0].copy()
            
            # Resize archive (remove random)
            target_arc = int(pop_size * arc_rate)
            if archive.shape[0] > target_arc:
                n_remove = archive.shape[0] - target_arc
                rm_idx = rng.choice(archive.shape[0], n_remove, replace=False)
                archive = np.delete(archive, rm_idx, axis=0)
        
        # 2. Parameter Generation
        r_idx = rng.integers(0, mem_size, pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # CR ~ Normal
        cr = rng.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy
        f = mu_f + 0.1 * np.tan(np.pi * (rng.random(pop_size) - 0.5))
        f[f > 1] = 1.0 # Clamp max
        # Retry if F <= 0
        while True:
            bad = f <= 0
            if not np.any(bad): break
            f[bad] = mu_f[bad] + 0.1 * np.tan(np.pi * (rng.random(np.sum(bad)) - 0.5))
            
        # 3. Mutation (current-to-pbest/1)
        # p linearly decreases from 0.2 to min(2/N)
        p_val = 0.2 - 0.1 * progress
        p_val = max(p_val, 2/pop_size)
        p_count = int(max(2, pop_size * p_val))
        
        # Select p-best indices (approximate top p is sufficient and faster)
        top_indices = np.argpartition(fitness, p_count-1)[:p_count]
        pbest_idxs = rng.choice(top_indices, pop_size)
        x_pbest = pop[pbest_idxs]
        
        # Select r1 != i
        r1 = rng.integers(0, pop_size, pop_size)
        bad_r1 = (r1 == np.arange(pop_size))
        while np.any(bad_r1):
            r1[bad_r1] = rng.integers(0, pop_size, np.sum(bad_r1))
            bad_r1 = (r1 == np.arange(pop_size))
        x_r1 = pop[r1]
        
        # Select r2 != r1, r2 != i (from Population U Archive)
        if archive.shape[0] > 0:
            union_pop = np.vstack((pop, archive))
        else:
            union_pop = pop
            
        r2 = rng.integers(0, union_pop.shape[0], pop_size)
        bad_r2 = (r2 == r1) | (r2 == np.arange(pop_size))
        while np.any(bad_r2):
            r2[bad_r2] = rng.integers(0, union_pop.shape[0], np.sum(bad_r2))
            bad_r2 = (r2 == r1) | (r2 == np.arange(pop_size))
        x_r2 = union_pop[r2]
        
        # Compute mutant vectors
        f_col = f[:, None]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        j_rand = rng.integers(0, dim, pop_size)
        mask = rng.random((pop_size, dim)) < cr[:, None]
        mask[np.arange(pop_size), j_rand] = True
        trial = np.where(mask, mutant, pop)
        
        # 5. Bound Constraints (Midpoint Correction)
        low_b = trial < min_b
        trial[low_b] = (min_b[np.where(low_b)[1]] + pop[low_b]) * 0.5
        high_b = trial > max_b
        trial[high_b] = (max_b[np.where(high_b)[1]] + pop[high_b]) * 0.5
        
        # 6. Evaluation
        succ_f = []
        succ_cr = []
        succ_diff = []
        any_improvement = False
        
        for i in range(pop_size):
            if not check_time(): return best_val
            
            y = func(trial[i])
            
            # Selection
            if y <= fitness[i]:
                # Improvement found
                if y < fitness[i]:
                    # Update Archive
                    if archive.shape[0] < int(pop_size * arc_rate):
                        archive = np.vstack((archive, pop[i]))
                    else:
                        rep_idx = rng.integers(0, archive.shape[0])
                        archive[rep_idx] = pop[i]
                    
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    succ_diff.append(fitness[i] - y)
                    any_improvement = True
                
                fitness[i] = y
                pop[i] = trial[i]
                
                if y < best_val:
                    best_val = y
                    best_sol = trial[i].copy()
                    best_idx = i
        
        # 7. Update Memory (Weighted Lehmer Mean)
        if len(succ_diff) > 0:
            w = np.array(succ_diff)
            w = w / np.sum(w)
            
            s_f = np.array(succ_f)
            s_cr = np.array(succ_cr)
            
            mean_scr = np.sum(w * s_cr)
            
            sum_sf = np.sum(w * s_f)
            if sum_sf > 1e-15:
                mean_sf = np.sum(w * s_f**2) / sum_sf
            else:
                mean_sf = 0.5
                
            m_cr[mem_k] = 0.5 * m_cr[mem_k] + 0.5 * mean_scr
            m_f[mem_k] = 0.5 * m_f[mem_k] + 0.5 * mean_sf
            mem_k = (mem_k + 1) % mem_size
            
        # 8. Local Search (Partial MTS-LS1)
        # Apply only to best_sol, on random subset of dimensions
        if any_improvement and check_time(0.05):
            dims_sub = rng.choice(dim, ls_dim_count, replace=False)
            
            for d in dims_sub:
                if not check_time(): break
                
                x_c = best_sol[d]
                sr = ls_sr[d]
                
                # Check negative step
                t_val = np.clip(x_c - sr, min_b[d], max_b[d])
                best_sol[d] = t_val
                v_ls = func(best_sol)
                
                if v_ls < best_val:
                    best_val = v_ls
                    fitness[best_idx] = v_ls
                    pop[best_idx, d] = t_val
                    # Keep direction (SR stays same)
                else:
                    # Check positive step (half magnitude)
                    t_val = np.clip(x_c + 0.5 * sr, min_b[d], max_b[d])
                    best_sol[d] = t_val
                    v_ls = func(best_sol)
                    
                    if v_ls < best_val:
                        best_val = v_ls
                        fitness[best_idx] = v_ls
                        pop[best_idx, d] = t_val
                    else:
                        # Fail: revert and reduce SR
                        best_sol[d] = x_c
                        ls_sr[d] *= 0.5
            
            # Prevent SR from vanishing
            small_sr = ls_sr < 1e-15
            if np.any(small_sr):
                ls_sr[small_sr] = diff_b[small_sr] * 0.4
                
        # 9. Restart Check
        # If population variance is near zero but time remains, restart
        if progress < 0.75:
            std = np.mean(np.std(pop, axis=0))
            if std < 1e-6 * np.mean(diff_b):
                # Keep best, randomize rest
                pop = min_b + rng.random((pop_size, dim)) * diff_b
                pop[0] = best_sol.copy()
                fitness[:] = float('inf')
                fitness[0] = best_val
                
                # Reset internals
                m_cr.fill(0.5)
                m_f.fill(0.5)
                archive = np.zeros((0, dim))
                ls_sr = diff_b * 0.4
                
                # Evaluate new population
                for i in range(1, pop_size):
                    if not check_time(): return best_val
                    val = func(pop[i])
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_sol = pop[i].copy()
                        best_idx = i

    return best_val
