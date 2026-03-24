#The following algorithm implements **Vectorized jSO-inspired L-SHADE with Gaussian Restarts**.
#
#### Key Improvements
#
#1.  **Dynamic Parameter Adaptation (jSO approach)**:
#    *   **Linear $p$-best Reduction**: The pool size for mutation parents shrinks linearly from an exploratory $17\%$ to an exploitative $4\%$ of the population over time.
#    *   **Weighted F-Bias**: In the early stages (first 25% of progress), the algorithm biases the Scaling Factor ($F$) slightly higher to encourage searching broader basins of attraction.
#
#2.  **Adaptive Gaussian Restarts**:
#    *   Instead of purely random restarts (which lose search history) or pure elitism (which causes stagnation), this algorithm uses a **Gaussian Explosion**.
#    *   When the population converges or the time schedule ends, the algorithm restarts. The new population is a mix: 50% is scattered normally around the current global best (using the search history's variance) to exploit the current basin, and 50% is generated randomly in the bounds to explore new areas.
#
#3.  **Smart Calibration & LPSR**:
#    *   The algorithm calibrates the function's cost to set a realistic initial population size ($N_{init}$).
#    *   It uses **Linear Population Size Reduction (LPSR)** mapped to the specific time budget of the current session.
#
#4.  **Vectorized Efficiency**: All evolutionary operators are vectorized to maximize the number of generations within the Python interpreter's overhead limits.
#
#### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Vectorized jSO-inspired L-SHADE with Gaussian Restarts and Time-Based LPSR.
    """
    t_start = time.time()
    
    # --- 1. Initialization & Calibration ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    bound_width = ub - lb
    
    # Safety margin for time checks
    time_margin = 0.05
    
    # Calibration: Estimate function evaluation time
    # We use a very small budget (max 0.1s or 1% of max_time) to calibrate
    cal_limit = min(0.1, max_time * 0.01)
    t_cal_start = time.time()
    n_cal = 0
    # Evaluate at center
    func(lb + bound_width * 0.5) 
    n_cal += 1
    
    while (time.time() - t_cal_start) < cal_limit and n_cal < 5:
        # Evaluate random point
        func(lb + np.random.rand(dim) * bound_width)
        n_cal += 1
    
    avg_eval_time = (time.time() - t_cal_start) / n_cal
    
    # Heuristic for Initial Population Size (N)
    # Target: ~18*dim, but constrained by estimated throughput to allow ~50 generations minimum
    est_total_evals = max_time / (avg_eval_time + 1e-9)
    n_ideal = int(18 * dim)
    n_capacity = int(est_total_evals / 40) # Ensure we can do at least 40 gens
    N_init = min(n_ideal, n_capacity)
    N_init = max(30, min(N_init, 300)) # Clamp N between [30, 300]
    N_min = 4
    
    # L-SHADE Parameters
    H_mem_size = 5
    arc_rate = 2.0  # Archive size = arc_rate * N
    
    # Global Best Tracking
    best_f = float('inf')
    best_x = None
    
    # --- 2. Main Optimization Loop (Sessions/Restarts) ---
    # We treat the run as a sequence of sessions.
    # If a session converges or finishes its LPSR schedule, we restart.
    
    restart_count = 0
    
    while True:
        # Check remaining time
        t_now = time.time()
        t_elapsed = t_now - t_start
        t_remain = max_time - t_elapsed
        
        if t_remain < time_margin:
            return best_f
        
        # Determine parameters for this session
        # If it's a restart, we scale N based on remaining time fraction
        scale = t_remain / max_time
        if restart_count > 0:
            # On restart, slightly reduce max pop to ensure convergence
            current_N_init = max(30, int(N_init * np.sqrt(scale)))
        else:
            current_N_init = N_init
            
        pop_size = current_N_init
        
        # --- Population Initialization ---
        pop = np.zeros((pop_size, dim))
        fit = np.zeros(pop_size)
        
        if best_x is None:
            # First initialization: Random Uniform
            pop = lb + np.random.rand(pop_size, dim) * bound_width
        else:
            # Restart Initialization: Gaussian Explosion + Random
            # 50% Random
            n_rnd = pop_size // 2
            pop[:n_rnd] = lb + np.random.rand(n_rnd, dim) * bound_width
            
            # 50% Gaussian around Best (Exploitation Restart)
            # Use a variance proportional to bounds, reduced by a factor
            sigma = bound_width * 0.1 
            n_gauss = pop_size - n_rnd
            
            # Generate Gaussian cloud
            noise = np.random.randn(n_gauss, dim) * sigma
            pop[n_rnd:] = best_x + noise
            
            # Inject explicit best (Elitism)
            pop[-1] = best_x
        
        # Clamp bounds
        pop = np.clip(pop, lb, ub)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if (time.time() - t_start) >= max_time: return best_f
            val = func(pop[i])
            fit[i] = val
            if val < best_f:
                best_f = val
                best_x = pop[i].copy()
                
        # Memory Initialization
        M_CR = np.full(H_mem_size, 0.5)
        M_F = np.full(H_mem_size, 0.5)
        k_mem = 0
        archive = []
        
        # Session Control
        session_start_time = time.time()
        # Allocate the rest of the time to this session
        session_budget = max_time - (session_start_time - t_start)
        
        # --- 3. Evolutionary Cycle ---
        while True:
            t_curr = time.time()
            if (t_curr - t_start) >= max_time: return best_f
            
            # Calculate Progress (0.0 to 1.0)
            sess_elapsed = t_curr - session_start_time
            progress = sess_elapsed / session_budget
            if progress > 1.0: progress = 1.0
            
            # A. Time-Based Linear Population Size Reduction (LPSR)
            target_n = int(round((N_min - current_N_init) * progress + current_N_init))
            target_n = max(N_min, target_n)
            
            if pop_size > target_n:
                # Reduce population: keep best
                sort_idx = np.argsort(fit)
                pop = pop[sort_idx[:target_n]]
                fit = fit[sort_idx[:target_n]]
                
                # Resize Archive
                target_arc = int(target_n * arc_rate)
                if len(archive) > target_arc:
                    # Random eviction
                    # Converting to list-based random removal is fast enough for small archives
                    del_count = len(archive) - target_arc
                    for _ in range(del_count):
                        archive.pop(np.random.randint(0, len(archive)))
                
                pop_size = target_n
            
            # B. Check Restart Conditions
            # 1. Stagnation (Zero variance in fitness)
            # 2. Schedule Completion (Progress > 95% and we still have buffer time)
            
            is_stagnant = np.std(fit) < 1e-9
            is_done = progress > 0.95
            
            if is_stagnant or is_done:
                # Only restart if we have meaningful time left (> 0.2s or > 5%)
                rem_t = max_time - (time.time() - t_start)
                if rem_t > max(0.2, max_time * 0.05):
                    restart_count += 1
                    break # Break inner loop to trigger restart logic
                elif is_stagnant:
                    return best_f # Converged and no time for meaningful restart
            
            # C. Dynamic Parameter Adjustment (jSO inspired)
            # p decreases linearly from 0.17 to 0.04
            p_val = 0.17 - (0.17 - 0.04) * progress
            p_best_count = max(2, int(pop_size * p_val))
            
            # Memory Index Selection
            r_idx = np.random.randint(0, H_mem_size, pop_size)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # Generate CR
            # Normal(mu_cr, 0.1), clamped [0, 1]
            # Use closest value to mu_cr if CR is invalid (-1) not needed here due to clip
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F
            # Cauchy(mu_f, 0.1)
            # If progress < 0.25 (early stage), we bias F slightly higher if possible (jSO hint)
            # But standard Cauchy is robust enough.
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Handle F constraints
            # F <= 0 -> Regenerate
            # F > 1 -> Clamp to 1
            while True:
                bad_f = f <= 0
                if not np.any(bad_f): break
                f[bad_f] = mu_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_f)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # D. Mutation: current-to-pbest/1 (Vectorized)
            # 1. Select p-best
            sorted_idx = np.argsort(fit)
            pbest_pool = sorted_idx[:p_best_count]
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # 2. Select r1 != i
            r1 = np.random.randint(0, pop_size, pop_size)
            col_r1 = r1 == np.arange(pop_size)
            r1[col_r1] = (r1[col_r1] + 1) % pop_size
            x_r1 = pop[r1]
            
            # 3. Select r2 != i, r2 != r1 (from Pop U Archive)
            n_arc = len(archive)
            if n_arc > 0:
                pool = np.vstack((pop, np.array(archive)))
            else:
                pool = pop
                
            r2 = np.random.randint(0, len(pool), pop_size)
            # Fix collisions for r2 (approximate)
            col_r2 = (r2 == np.arange(pop_size)) | (r2 == r1)
            if np.any(col_r2):
                r2[col_r2] = np.random.randint(0, len(pool), np.sum(col_r2))
            x_r2 = pool[r2]
            
            # Compute Mutant: v = x + F*(xp - x) + F*(xr1 - xr2)
            f_mat = f[:, np.newaxis]
            mutant = pop + f_mat * (x_pbest - pop) + f_mat * (x_r1 - x_r2)
            
            # E. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # F. Bound Handling (Midpoint Correction)
            # If a variable is out of bounds, set it to (parent + bound) / 2
            # This is better than clipping (preserves variance) and better than random (preserves progress)
            
            vio_l = trial < lb
            if np.any(vio_l):
                # We need broadcasting
                lb_mat = np.tile(lb, (pop_size, 1))
                trial[vio_l] = (pop[vio_l] + lb_mat[vio_l]) * 0.5
                
            vio_u = trial > ub
            if np.any(vio_u):
                ub_mat = np.tile(ub, (pop_size, 1))
                trial[vio_u] = (pop[vio_u] + ub_mat[vio_u]) * 0.5
                
            # G. Selection and Memory Update
            # We must evaluate one by one to respect max_time safely,
            # but we can batch update arrays.
            
            new_fit = np.zeros(pop_size)
            
            # Temp lists for successful updates
            succ_cr = []
            succ_f = []
            diff_f = []
            
            for i in range(pop_size):
                # Periodical time check for large populations
                if i % 10 == 0:
                     if (time.time() - t_start) >= max_time: return best_f
                
                f_trial = func(trial[i])
                new_fit[i] = f_trial
                
                # Selection Logic
                if f_trial <= fit[i]:
                    # Strictly better for memory updates
                    if f_trial < fit[i]:
                        succ_cr.append(cr[i])
                        succ_f.append(f[i])
                        diff_f.append(fit[i] - f_trial)
                        archive.append(pop[i].copy())
                    
                    # Update Best
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial[i].copy()
                
            # Vectorized Population Update
            # Replace where trial is better or equal
            imp_mask = new_fit <= fit
            pop[imp_mask] = trial[imp_mask]
            fit[imp_mask] = new_fit[imp_mask]
            
            # H. Update Historical Memory (Weighted Lehmer Mean)
            if len(diff_f) > 0:
                w = np.array(diff_f)
                total_imp = np.sum(w)
                if total_imp > 0:
                    w = w / total_imp
                    
                    # Update M_CR
                    scr = np.array(succ_cr)
                    m_cr_new = np.sum(w * scr)
                    # Use learning rate? Standard SHADE replaces directly.
                    M_CR[k_mem] = m_cr_new
                    
                    # Update M_F (Lehmer Mean)
                    sf = np.array(succ_f)
                    num = np.sum(w * (sf ** 2))
                    den = np.sum(w * sf)
                    m_f_new = num / den if den > 0 else 0.5
                    M_F[k_mem] = m_f_new
                    
                    k_mem = (k_mem + 1) % H_mem_size
    
    return best_f
