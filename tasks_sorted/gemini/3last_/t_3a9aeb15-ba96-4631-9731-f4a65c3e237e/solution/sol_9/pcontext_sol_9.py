#The following algorithm implements an improved **L-SHADE with Latin Hypercube Initialization, Dynamic p-best, and Time-Based Restart**.
#
#### Key Improvements
#1.  **Latin Hypercube Sampling (LHS) Initialization**: Instead of purely random uniform initialization, the algorithm uses LHS to stratify the initial population. This ensures that the samples are evenly distributed across each dimension's range, maximizing initial exploration coverage.
#2.  **Adaptive Population Sizing with LHS**: The initial population size $N$ is dynamically calculated based on the available time budget and the function's evaluation cost. It balances the need for diversity (high $N$) with the need for convergence (generations).
#3.  **Dynamic Parameter $p$**: Following the **jSO** strategy, the "p-best" parameter (controlling greediness) scales linearly from exploratory ($0.17$) to exploitative ($0.05$) based on the session progress.
#4.  **Restart with Elitism**: If the population converges (stagnation) or the time schedule for the session completes, the algorithm triggers a restart using the remaining time. Crucially, it injects the global best solution into the new Latin Hypercube population to prevent regressing while exploring new basins.
#
#### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Optimized L-SHADE with LHS Initialization, Dynamic p-best, and Time-Based Restart.
    """
    t_start = time.time()
    
    # --- 1. Pre-computation & Calibration ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    
    # Calibrate function evaluation time to estimate budget
    # Use a small time slice (max 1% of total time or 0.05s) to estimate cost
    cal_duration = min(max_time * 0.01, 0.05)
    t_cal_start = time.time()
    n_cal = 0
    while time.time() - t_cal_start < cal_duration:
        x_dumb = lb + np.random.rand(dim) * (ub - lb)
        func(x_dumb)
        n_cal += 1
        if n_cal >= 10: break 
    
    t_cal_end = time.time()
    # Average time per evaluation
    t_per_eval = (t_cal_end - t_cal_start) / n_cal if n_cal > 0 else 0.0
    
    # Global best tracking
    best_x = None
    best_f = float('inf')
    
    # Constants
    H_MEM = 5       # Memory size
    N_MIN = 4       # Minimum population size for LPSR
    
    # --- 2. Main Restart Loop ---
    # The optimization is divided into sessions.
    # If a session converges or finishes its time schedule, we restart.
    
    while True:
        # Check total remaining time
        t_now = time.time()
        t_total_elapsed = t_now - t_start
        t_remaining = max_time - t_total_elapsed
        
        # Buffer to ensure we return safely
        if t_remaining < 0.05: 
            return best_f
            
        # --- Session Configuration ---
        
        # Estimate available evaluations in the remaining time
        # Safety factor 0.9 to account for overhead
        eval_budget = (t_remaining * 0.9) / (t_per_eval + 1e-9)
        
        # Heuristic for Population Size (N)
        # 1. Based on Dimension (Standard L-SHADE: 18*D)
        n_dim_based = int(18 * dim)
        
        # 2. Based on Budget (Need at least ~40 generations for convergence)
        # If function is slow, we can't sustain a large population
        n_budget_based = int(eval_budget / 40)
        
        # Combine: prefer dim-based, but cap by budget
        N = min(n_dim_based, n_budget_based)
        
        # Hard Constraints
        # We need enough diversity (min 30) but not too slow (max 800)
        N = max(30, N)
        N = min(N, 800)
        
        N_init = N
        
        # --- Population Initialization (LHS) ---
        # Latin Hypercube Sampling for better initial coverage
        pop = np.zeros((N, dim))
        for d in range(dim):
            # Create N strata
            edges = np.linspace(lb[d], ub[d], N + 1)
            # Sample uniformly in each stratum
            r = np.random.uniform(0, 1, N)
            vals = edges[:-1] + r * (edges[1:] - edges[:-1])
            # Shuffle to mix dimensions
            np.random.shuffle(vals)
            pop[:, d] = vals
            
        # Elitism: Inject global best if available (Restart)
        # We replace index 0 with the best found so far
        fit = np.zeros(N)
        start_idx = 0
        
        if best_x is not None:
            pop[0] = best_x
            fit[0] = best_f
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, N):
            if (time.time() - t_start) >= max_time: return best_f
            val = func(pop[i])
            fit[i] = val
            if val < best_f:
                best_f = val
                best_x = pop[i].copy()
                
        # --- L-SHADE Setup ---
        M_CR = np.full(H_MEM, 0.5)
        M_F = np.full(H_MEM, 0.5)
        k_mem = 0
        archive = []
        
        # Session Time Management
        t_sess_start = time.time()
        # Allocate the rest of the time to this session
        sess_budget_time = max_time - (t_sess_start - t_start)
        if sess_budget_time <= 0: return best_f
        
        # --- Evolutionary Cycle ---
        n_curr = N
        
        while True:
            t_curr = time.time()
            if (t_curr - t_start) >= max_time: return best_f
            
            # Calculate Progress (0 -> 1)
            sess_elapsed = t_curr - t_sess_start
            progress = sess_elapsed / sess_budget_time
            if progress > 1.0: progress = 1.0
            
            # --- 1. LPSR: Linear Population Size Reduction ---
            n_target = int(round((N_MIN - N_init) * progress + N_init))
            n_target = max(N_MIN, n_target)
            
            if n_curr > n_target:
                # Truncate population (keep best)
                sort_idx = np.argsort(fit)
                pop = pop[sort_idx[:n_target]]
                fit = fit[sort_idx[:n_target]]
                
                # Resize Archive (Maintain Archive size ~ 2 * Population)
                arc_target = int(n_target * 2.0)
                if len(archive) > arc_target:
                    # Random removal
                    keep_mask = np.random.choice(len(archive), arc_target, replace=False)
                    archive = [archive[k] for k in keep_mask]
                
                n_curr = n_target
            
            # --- 2. Restart Criteria ---
            # Stagnation check
            std_fit = np.std(fit)
            is_stagnant = std_fit < 1e-9
            is_done = progress > 0.98
            
            if is_stagnant or is_done:
                # If we have meaningful time left (>0.15s), restart
                remaining = max_time - (time.time() - t_start)
                if remaining > 0.15:
                    break # Break inner loop -> triggers new session
                elif is_stagnant:
                    # Converged and no time left to restart effectively
                    return best_f
            
            # --- 3. Parameter Generation ---
            # Dynamic p-best parameter (jSO: 0.17 -> 0.05)
            p_val = 0.17 - (0.12 * progress)
            n_pbest = max(2, int(n_curr * p_val))
            
            # Select memories
            r_idx = np.random.randint(0, H_MEM, n_curr)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # Generate CR ~ Normal(mu, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(mu, 0.1)
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(n_curr) - 0.5))
            
            # Handle F Constraints (<=0 regenerate, >1 clamp)
            while True:
                bad_f = f <= 0
                if not np.any(bad_f): break
                f[bad_f] = mu_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_f)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # --- 4. Mutation (current-to-pbest/1) ---
            # Sort for p-best
            sorted_idx = np.argsort(fit)
            pbest_pool = sorted_idx[:n_pbest]
            
            # Indices
            idx_pbest = np.random.choice(pbest_pool, n_curr)
            x_pbest = pop[idx_pbest]
            
            idx_r1 = np.random.randint(0, n_curr, n_curr)
            # Fix r1 == i
            col_r1 = idx_r1 == np.arange(n_curr)
            idx_r1[col_r1] = (idx_r1[col_r1] + 1) % n_curr
            x_r1 = pop[idx_r1]
            
            # r2 from Pop U Archive
            if len(archive) > 0:
                pool = np.vstack((pop, np.array(archive)))
            else:
                pool = pop
                
            idx_r2 = np.random.randint(0, len(pool), n_curr)
            # Fix r2 collisions (approximate)
            col_r2 = (idx_r2 == np.arange(n_curr)) | (idx_r2 == idx_r1)
            if np.any(col_r2):
                idx_r2[col_r2] = np.random.randint(0, len(pool), np.sum(col_r2))
            x_r2 = pool[idx_r2]
            
            # Mutation Vector: v = x + F(xp - x) + F(xr1 - xr2)
            f_mat = f[:, np.newaxis]
            mutant = pop + f_mat * (x_pbest - pop) + f_mat * (x_r1 - x_r2)
            
            # --- 5. Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, n_curr)
            mask = np.random.rand(n_curr, dim) < cr[:, np.newaxis]
            mask[np.arange(n_curr), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # --- 6. Bound Handling (Midpoint Correction) ---
            vio_l = trial < lb
            if np.any(vio_l):
                trial[vio_l] = (pop[vio_l] + np.tile(lb, (n_curr, 1))[vio_l]) * 0.5
                
            vio_u = trial > ub
            if np.any(vio_u):
                trial[vio_u] = (pop[vio_u] + np.tile(ub, (n_curr, 1))[vio_u]) * 0.5
                
            # --- 7. Selection & Evaluation ---
            new_fit = np.zeros(n_curr)
            
            succ_cr = []
            succ_f = []
            diff_f = []
            
            for i in range(n_curr):
                # Periodic time check for responsiveness
                if i % 50 == 0 and (time.time() - t_start) >= max_time:
                    return best_f
                
                f_trial = func(trial[i])
                new_fit[i] = f_trial
                
                if f_trial <= fit[i]:
                    # Strictly better for memory updates
                    if f_trial < fit[i]:
                        succ_cr.append(cr[i])
                        succ_f.append(f[i])
                        diff_f.append(fit[i] - f_trial)
                        archive.append(pop[i].copy())
                        
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial[i].copy()
            
            # Update Population
            imp = new_fit <= fit
            pop[imp] = trial[imp]
            fit[imp] = new_fit[imp]
            
            # --- 8. Memory Update (Weighted Lehmer Mean) ---
            if len(diff_f) > 0:
                w = np.array(diff_f)
                total = np.sum(w)
                if total > 0:
                    w = w / total
                    
                    # M_CR update
                    m_cr_new = np.sum(w * np.array(succ_cr))
                    M_CR[k_mem] = m_cr_new
                    
                    # M_F update
                    sf = np.array(succ_f)
                    num = np.sum(w * (sf ** 2))
                    den = np.sum(w * sf)
                    m_f_new = num / den if den > 0 else 0.5
                    M_F[k_mem] = m_f_new
                    
                    k_mem = (k_mem + 1) % H_MEM

    return best_f
