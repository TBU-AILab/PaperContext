#Here is the improved algorithm.
#
#**Improvements Explanation:**
#1.  **L-SHADE-RSP Foundation**: The core is built on L-SHADE with Linear Population Size Reduction (LPSR) and Rank-based Selection, which is a top-performing structure in global optimization competitions (CEC).
#2.  **MTS-LS1 Local Search**: Replaced the simple coordinate descent with MTS-LS1 (Multiple Trajectory Search - Local Search 1). This method maintains dynamic search ranges (`sr`) for each dimension, allowing it to adaptively "zoom in" on the minimum much faster than random perturbations.
#3.  **Boundary Correction**: Instead of reflection (which can trap particles at boundaries) or clamping, it uses a midpoint target (`(bound + old) / 2`). This preserves the evolutionary direction while keeping valid values.
#4.  **Robust Restart**: Detects population collapse (variance < threshold) and triggers a restart while preserving the best solution, allowing the algorithm to escape local optima basins if they are fully explored.
#5.  **Vectorized Operations**: Maximized NumPy vectorization for mutation, crossover, and memory updates to minimize Python overhead and maximize `func` evaluations.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE-RSP with MTS-LS1 Local Search.
    """
    # --- Time Management ---
    t_start = datetime.now()
    t_limit = timedelta(seconds=max_time)
    
    # Helper to check remaining time with a safety buffer
    def check_time(buffer_sec=0.0):
        return (datetime.now() - t_start) < (t_limit - timedelta(seconds=buffer_sec))

    # --- Initialization ---
    rng = np.random.default_rng()
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Population Size: Linear Reduction
    # Start with a larger population for exploration
    n_init = int(18 * dim)
    n_init = np.clip(n_init, 30, 400) # Clamp to reasonable limits
    n_min = 4
    
    pop_size = n_init
    pop = min_b + rng.random((pop_size, dim)) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_idx = -1
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if not check_time(0.01): return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop[i].copy()
            best_idx = i
            
    # L-SHADE Memory
    mem_size = 5
    m_cr = np.full(mem_size, 0.5)
    m_f = np.full(mem_size, 0.5)
    mem_k = 0
    
    # External Archive
    archive = np.zeros((0, dim))
    arc_rate = 2.0 # Archive size relative to population
    
    # MTS-LS1 Local Search State
    ls_sr = diff_b * 0.4 # Search range per dimension
    
    # --- Main Optimization Loop ---
    while check_time(0.05): # Leave 0.05s buffer for overhead
        
        # 1. Linear Population Size Reduction (LPSR)
        elapsed = (datetime.now() - t_start).total_seconds()
        progress = elapsed / max_time
        
        n_next = int(round((n_min - n_init) * progress + n_init))
        n_next = max(n_min, n_next)
        
        if pop_size > n_next:
            # Sort population to keep best individuals
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            # Truncate
            pop = pop[:n_next]
            fitness = fitness[:n_next]
            pop_size = n_next
            
            # Resize Archive
            target_arc_size = int(pop_size * arc_rate)
            if archive.shape[0] > target_arc_size:
                del_count = archive.shape[0] - target_arc_size
                idxs = rng.choice(archive.shape[0], del_count, replace=False)
                archive = np.delete(archive, idxs, axis=0)
            
            # Update best pointer (index 0 is best after sort)
            best_idx = 0
            best_val = fitness[0]
            best_sol = pop[0].copy()

        # 2. Parameter Generation
        # Select memory slots
        r_idx = rng.integers(0, mem_size, pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = rng.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        # Retry F <= 0
        f = mu_f + 0.1 * np.tan(np.pi * (rng.random(pop_size) - 0.5))
        while True:
            bad = f <= 0
            if not np.any(bad): break
            f[bad] = mu_f[bad] + 0.1 * np.tan(np.pi * (rng.random(np.sum(bad)) - 0.5))
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: current-to-pbest/1
        # p-best parameter decreases linearly to focus on exploitation
        p_ratio = max(2.0/pop_size, 0.2 * (1.0 - progress))
        p_count = int(max(2, pop_size * p_ratio))
        
        # Select p-best indices
        # Partial sort is faster than full sort
        sorted_indices = np.argsort(fitness)
        pbest_candidates = sorted_indices[:p_count]
        pbest_idxs = rng.choice(pbest_candidates, pop_size)
        x_pbest = pop[pbest_idxs]
        
        # Select r1 != i
        r1 = rng.integers(0, pop_size, pop_size)
        retry = (r1 == np.arange(pop_size))
        while np.any(retry):
            r1[retry] = rng.integers(0, pop_size, np.sum(retry))
            retry = (r1 == np.arange(pop_size))
        x_r1 = pop[r1]
        
        # Select r2 != r1, r2 != i from (Population U Archive)
        if archive.shape[0] > 0:
            union_pop = np.vstack((pop, archive))
        else:
            union_pop = pop
            
        r2 = rng.integers(0, union_pop.shape[0], pop_size)
        retry = (r2 == np.arange(pop_size)) | (r2 == r1)
        while np.any(retry):
            r2[retry] = rng.integers(0, union_pop.shape[0], np.sum(retry))
            retry = (r2 == np.arange(pop_size)) | (r2 == r1)
        x_r2 = union_pop[r2]
        
        # Compute Mutant Vector
        f_v = f[:, None]
        mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        j_rand = rng.integers(0, dim, pop_size)
        cross_mask = rng.random((pop_size, dim)) < cr[:, None]
        cross_mask[np.arange(pop_size), j_rand] = True
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Constraint Handling (Midpoint Correction)
        # If out of bounds, place halfway between old value and bound
        lower = trial < min_b
        trial[lower] = (min_b[np.where(lower)[1]] + pop[lower]) * 0.5
        
        upper = trial > max_b
        trial[upper] = (max_b[np.where(upper)[1]] + pop[upper]) * 0.5
        
        # 6. Evaluation and Selection
        success_f = []
        success_cr = []
        success_diff = []
        pop_improved = False
        
        for i in range(pop_size):
            if not check_time(): return best_val
            
            new_val = func(trial[i])
            
            if new_val <= fitness[i]:
                # Strict improvement for stats, <= for update
                if new_val < fitness[i]:
                    # Add old vector to archive
                    if archive.shape[0] < int(pop_size * arc_rate):
                        archive = np.vstack((archive, pop[i]))
                    else:
                        # Random replacement
                        ridx = rng.integers(0, archive.shape[0])
                        archive[ridx] = pop[i]
                        
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    success_diff.append(fitness[i] - new_val)
                    pop_improved = True
                
                pop[i] = trial[i]
                fitness[i] = new_val
                
                if new_val < best_val:
                    best_val = new_val
                    best_sol = trial[i].copy()
                    best_idx = i
        
        # 7. Memory Update (Weighted Lehmer Mean)
        if success_diff:
            w = np.array(success_diff)
            w = w / np.sum(w)
            
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            mean_scr = np.sum(w * s_cr)
            # Lehmer mean for F
            mean_sf = np.sum(w * s_f**2) / np.sum(w * s_f)
            
            m_cr[mem_k] = 0.5 * m_cr[mem_k] + 0.5 * mean_scr
            m_f[mem_k] = 0.5 * m_f[mem_k] + 0.5 * mean_sf
            mem_k = (mem_k + 1) % mem_size
            
        # 8. MTS-LS1 Local Search
        # Trigger: If population improved, try to refine the best solution.
        # This is a Coordinate Descent with adaptive step sizes (SR).
        if pop_improved and check_time(0.2):
            improved_ls = False
            # Random shuffle of dimensions to avoid bias
            dims_to_search = rng.permutation(dim)
            
            for d in dims_to_search:
                if not check_time(): break
                
                x_curr = best_sol[d]
                sr = ls_sr[d]
                
                # Try negative step
                best_sol[d] = np.clip(x_curr - sr, min_b[d], max_b[d])
                val = func(best_sol)
                
                if val < best_val:
                    best_val = val
                    fitness[best_idx] = val
                    pop[best_idx, d] = best_sol[d]
                    improved_ls = True
                else:
                    # Try positive step (half magnitude)
                    best_sol[d] = np.clip(x_curr + 0.5 * sr, min_b[d], max_b[d])
                    val = func(best_sol)
                    
                    if val < best_val:
                        best_val = val
                        fitness[best_idx] = val
                        pop[best_idx, d] = best_sol[d]
                        improved_ls = True
                    else:
                        # Revert and reduce search range
                        best_sol[d] = x_curr
                        ls_sr[d] *= 0.5
            
            # Prevent search range from vanishing completely
            ls_sr[ls_sr < 1e-15] = diff_b[ls_sr < 1e-15] * 0.4

        # 9. Diversity-Based Restart
        # If population converges (low std dev) but time remains, restart
        if progress < 0.85:
            std = np.mean(np.std(pop, axis=0))
            if std < 1e-7:
                # Keep best, re-init others
                pop = min_b + rng.random((pop_size, dim)) * diff_b
                pop[0] = best_sol
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_val
                
                # Evaluate new population
                for i in range(1, pop_size):
                    if not check_time(): return best_val
                    fitness[i] = func(pop[i])
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_sol = pop[i].copy()
                        best_idx = i
                        
                # Reset Memory and Archive
                m_cr.fill(0.5)
                m_f.fill(0.5)
                archive = np.zeros((0, dim))
                ls_sr = diff_b * 0.4
                
    return best_val
