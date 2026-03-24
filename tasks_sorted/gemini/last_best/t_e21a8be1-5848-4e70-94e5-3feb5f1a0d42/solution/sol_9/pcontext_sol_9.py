#The output value of the last generated algorithm is: 6.941715971031726
#
# The last generated algorithm code:
#The following algorithm implements **L-SHADE-RSP-OBL-iMTS**.
#
##### Key Improvements over previous versions:
#1.  **Interleaved Local Search (iMTS)**: Instead of waiting for the end of the time budget, this algorithm periodically (every 40 gens) applies a lightweight MTS local search to the global best solution. This helps the evolutionary "current-to-pbest" mutation strategy by pulling the population towards the *true* bottom of the current basin earlier.
#2.  **Gaussian-Random Restart**: If the population variance collapses (stagnation), instead of a purely random restart, it performs a 50/50 restart: half the population samples a Gaussian distribution around the current best (Exploitation of basin), and half explores globally (Exploration).
#3.  **Adaptive Time Management**: The switch from Evolutionary phase to Final Local Search phase is dynamic based on remaining time (last 15%).
#4.  **Vectorized Operations**: Optimized numpy operations for speed in the evolutionary loop.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-RSP-OBL-iMTS.
    Combines Opposition-Based Learning, L-SHADE with Linear Population Reduction,
    Interleaved Multiple Trajectory Search, and Gaussian Restarts.
    """

    # --- 1. Setup & Time Management ---
    t_start = time.time()
    t_end = t_start + max_time
    
    # Allocate last 15% of time for intensive Final Local Search
    ratio_ls = 0.15
    t_ls_start = t_start + (1.0 - ratio_ls) * max_time
    
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # --- 2. Initialization (OBL) ---
    # Initial Population Size: 18 * dim (standard for L-SHADE)
    pop_size_init = int(round(18 * dim))
    pop_size_min = 4
    pop_size = pop_size_init
    
    # SHADE Memory (H=6)
    H = 6
    M_cr = np.full(H, 0.5)
    M_f = np.full(H, 0.5)
    k_mem = 0
    archive = []
    
    # Generate Initial Population
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Opposition-Based Learning: x' = lb + ub - x
    pop_opp = lb + ub - pop
    pop_opp = np.clip(pop_opp, lb, ub)
    
    # Combine and Evaluate
    pop_pool = np.vstack((pop, pop_opp))
    n_pool = len(pop_pool)
    fit_pool = np.full(n_pool, float('inf'))
    
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate pool with strict time check
    for i in range(n_pool):
        if time.time() >= t_ls_start:
            # If init takes too long, truncate
            pop_pool = pop_pool[:i]
            fit_pool = fit_pool[:i]
            break
            
        val = func(pop_pool[i])
        fit_pool[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop_pool[i].copy()
            
    # Select best N individuals
    if len(pop_pool) > pop_size:
        idx = np.argsort(fit_pool)
        pop = pop_pool[idx[:pop_size]]
        fitness = fit_pool[idx[:pop_size]]
    else:
        pop = pop_pool
        fitness = fit_pool
        pop_size = len(pop)
        
    # --- 3. Evolutionary Loop (L-SHADE-RSP) ---
    gen = 0
    # Search Range for MTS (maintained per dimension)
    sr = (ub - lb) * 0.4
    
    while True:
        t_curr = time.time()
        if t_curr >= t_ls_start:
            break
            
        # A. Linear Population Size Reduction (LPSR)
        budget_total = t_ls_start - t_start
        budget_used = t_curr - t_start
        if budget_total <= 0: break
        progress = np.clip(budget_used / budget_total, 0, 1)
        
        target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            # Sort and truncate
            idx = np.argsort(fitness)
            pop = pop[idx[:target_size]]
            fitness = fitness[idx[:target_size]]
            pop_size = target_size
            
            # Resize Archive (maintain size ~ 2.0 * pop_size)
            arc_target = int(pop_size * 2.0)
            while len(archive) > arc_target:
                archive.pop(np.random.randint(0, len(archive)))
                
        # B. Parameter Adaptation
        r_idx = np.random.randint(0, H, pop_size)
        mcr = M_cr[r_idx]
        mf = M_f[r_idx]
        
        # CR ~ Normal(mcr, 0.1)
        cr = np.random.normal(mcr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(mf, 0.1)
        f = mf + 0.1 * np.random.standard_cauchy(pop_size)
        # Handle negative F by retrying
        for _ in range(5):
            bad = f <= 0
            if not np.any(bad): break
            f[bad] = mf[bad] + 0.1 * np.random.standard_cauchy(np.sum(bad))
        f = np.clip(f, 0, 1)
        f[f <= 0] = 0.5 # Fallback
        
        # C. Mutation: current-to-pbest/1
        # p adapts from 0.2 (Exploration) to 0.05 (Exploitation)
        p_val = 0.2 - 0.15 * progress
        p_val = max(0.05, p_val)
        p_num = max(2, int(pop_size * p_val))
        
        srt = np.argsort(fitness)
        # Select pbest
        pbest_idx = np.random.choice(srt[:p_num], pop_size)
        x_pbest = pop[pbest_idx]
        
        # Select r1 (distinct from i)
        r1 = np.random.randint(0, pop_size, pop_size)
        col = r1 == np.arange(pop_size)
        r1[col] = (r1[col] + 1) % pop_size
        x_r1 = pop[r1]
        
        # Select r2 (distinct from i, r1, from Pop U Archive)
        if len(archive) > 0:
            u_pop = np.vstack((pop, np.array(archive)))
        else:
            u_pop = pop
            
        r2 = np.random.randint(0, len(u_pop), pop_size)
        col2 = (r2 == np.arange(pop_size)) | (r2 == r1)
        r2[col2] = np.random.randint(0, len(u_pop), np.sum(col2))
        x_r2 = u_pop[r2]
        
        # Mutation Vector
        mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
        
        # D. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        mask[np.arange(pop_size), j_rand] = True
        trial = np.where(mask, mutant, pop)
        
        # E. Bound Handling (Reflection)
        bl = trial < lb
        if np.any(bl): trial[bl] = 2*lb[np.where(bl)[1]] - trial[bl]
        bu = trial > ub
        if np.any(bu): trial[bu] = 2*ub[np.where(bu)[1]] - trial[bu]
        trial = np.clip(trial, lb, ub)
        
        # F. Selection
        fit_new = np.zeros(pop_size)
        success = np.zeros(pop_size, dtype=bool)
        diff = np.zeros(pop_size)
        
        # Batch evaluation with check
        check_step = max(1, int(pop_size/3))
        
        for i in range(pop_size):
            if i % check_step == 0 and time.time() >= t_ls_start:
                break
                
            val = func(trial[i])
            fit_new[i] = val
            
            if val <= fitness[i]:
                if val < fitness[i]:
                    diff[i] = fitness[i] - val
                    success[i] = True
                    archive.append(pop[i].copy())
                    
                pop[i] = trial[i]
                fitness[i] = val
                
                if val < best_val:
                    best_val = val
                    best_sol = trial[i].copy()
                    
        # G. Memory Update (Weighted)
        if np.any(success):
            w = diff[success] / np.sum(diff[success])
            
            # Weighted Mean for CR
            m_scr = np.sum(w * cr[success])
            M_cr[k_mem] = 0.5 * M_cr[k_mem] + 0.5 * m_scr
            
            # Weighted Lehmer Mean for F
            s_f = f[success]
            num = np.sum(w * s_f**2)
            den = np.sum(w * s_f)
            if den > 1e-12:
                M_f[k_mem] = 0.5 * M_f[k_mem] + 0.5 * (num/den)
                
            k_mem = (k_mem + 1) % H
            
        # Maintain Archive Size
        while len(archive) > int(pop_size * 2.0):
            archive.pop(np.random.randint(0, len(archive)))
            
        # H. Interleaved MTS (iMTS)
        # Periodically refine best_sol to lead the population into the basin
        if gen > 0 and gen % 40 == 0 and time.time() < t_ls_start:
             # Apply MTS to best_sol
             temp_sol = best_sol.copy()
             temp_val = best_val
             improved_ls = False
             
             # Shuffle dimensions
             dims_idx = np.arange(dim)
             np.random.shuffle(dims_idx)
             
             for d_i in dims_idx:
                 if time.time() >= t_ls_start: break
                 
                 original = temp_sol[d_i]
                 
                 # 1. Try Negative Step
                 temp_sol[d_i] = np.clip(original - sr[d_i], lb[d_i], ub[d_i])
                 v = func(temp_sol)
                 
                 if v < temp_val:
                     temp_val = v
                     best_val = v
                     best_sol = temp_sol.copy()
                     improved_ls = True
                 else:
                     # 2. Try Positive Step (0.5 size)
                     temp_sol[d_i] = np.clip(original + 0.5 * sr[d_i], lb[d_i], ub[d_i])
                     v = func(temp_sol)
                     if v < temp_val:
                         temp_val = v
                         best_val = v
                         best_sol = temp_sol.copy()
                         improved_ls = True
                     else:
                         # Revert
                         temp_sol[d_i] = original
                         sr[d_i] *= 0.5 # Shrink search range on failure
                         
             if not improved_ls:
                 # If all dims failed and range is tiny, reset search range for next time
                 if np.max(sr) < 1e-8:
                     sr = (ub - lb) * 0.2
                     
        # I. Gaussian-Random Restart (Stagnation Handling)
        if gen % 20 == 0:
            if np.std(fitness) < 1e-10:
                # Keep best index
                idx_b = np.argmin(fitness)
                
                # Small sigma for exploitation around best
                sigma = (ub - lb) * 0.01 
                
                for i in range(pop_size):
                    if i == idx_b: continue
                    
                    # 50% Gaussian around best, 50% Random Global
                    if np.random.rand() < 0.5:
                        cand = np.random.normal(best_sol, sigma)
                    else:
                        cand = np.random.uniform(lb, ub)
                        
                    cand = np.clip(cand, lb, ub)
                    
                    if time.time() >= t_ls_start: break
                    v = func(cand)
                    pop[i] = cand
                    fitness[i] = v
                    
                    if v < best_val:
                        best_val = v
                        best_sol = cand.copy()
                
                # Reset Adaptation Memory
                M_cr.fill(0.5)
                M_f.fill(0.5)
                archive = []
                
        gen += 1
        
    # --- 4. Final Local Search (Intensive MTS) ---
    curr_sol = best_sol.copy()
    curr_val = best_val
    
    # Ensure search range is not too small to start
    sr = np.maximum(sr, (ub - lb) * 1e-8)
    
    while time.time() < t_end:
        dims_idx = np.arange(dim)
        np.random.shuffle(dims_idx)
        improved_any = False
        
        for i in dims_idx:
            if time.time() >= t_end: break
            
            orig = curr_sol[i]
            
            # MTS Step Logic
            curr_sol[i] = np.clip(orig - sr[i], lb[i], ub[i])
            v = func(curr_sol)
            
            if v < curr_val:
                curr_val = v
                best_val = v
                improved_any = True
            else:
                curr_sol[i] = np.clip(orig + 0.5 * sr[i], lb[i], ub[i])
                v = func(curr_sol)
                
                if v < curr_val:
                    curr_val = v
                    best_val = v
                    improved_any = True
                else:
                    curr_sol[i] = orig
                    sr[i] *= 0.5 # Reduce range
                    
        # Pattern Search Reset
        if not improved_any:
            # If converged, reset range to try to escape numerical trap
            if np.max(sr) < 1e-15:
                 sr = (ub - lb) * 0.1
                 
    return best_val
