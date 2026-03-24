#The following algorithm implements an improved **Vectorized L-SHADE with Adaptive Budgeting and Time-Based Linear Population Size Reduction**.
#
##### Key Improvements
#1.  **Adaptive Initialization**: The algorithm begins with a calibration step to measure the execution time of the objective function. It then dynamically calculates an initial population size that fits the problem's computational cost, ensuring enough generations can be run for the evolutionary logic to work effectively, whether the function is fast or slow.
#2.  **Time-Based LPSR**: The Linear Population Size Reduction (LPSR) strategy—shrinking the population from exploration to exploitation—is mapped directly to the **wall-clock time**. This ensures the algorithm fully utilizes the available `max_time` without overrunning or finishing too early.
#3.  **Soft Restarts with Elitism**: If the population converges early or the time schedule completes with time remaining, the algorithm triggers a restart. It carries over the global best solution (Elitism) to the new population to refine the search (intensification) while the new random individuals explore new basins of attraction.
#4.  **Vectorized Operations**: Genetic operations (mutation, crossover, boundary handling) are fully vectorized using NumPy, minimizing overhead and maximizing the number of evaluations per second.
#
##### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Vectorized L-SHADE with Adaptive Budgeting, Time-Based LPSR, and Restarts.
    """
    t0 = time.time()
    
    # --- 1. Initialization & Calibration ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    
    # Measure function cost with a single evaluation to estimate the budget.
    t_cal_start = time.time()
    # Sample the center point for calibration
    x_cal = lb + (ub - lb) * 0.5
    f_cal = func(x_cal)
    t_cal_end = time.time()
    dt = t_cal_end - t_cal_start
    
    # Global best tracking
    best_x = x_cal.copy()
    best_f = f_cal
    
    # Parameters for L-SHADE
    r_size = 18         # Initial population size factor (N = r * dim)
    n_min = 4           # Minimum population size for LPSR
    arc_rate = 2.6      # Archive size factor
    h_mem = 6           # Memory size
    p_best_rate = 0.11  # Top p-best selection rate
    
    # --- 2. Main Optimization Loop (Session/Restart) ---
    while True:
        # Check global time budget
        t_curr = time.time()
        t_rem = max_time - (t_curr - t0)
        
        # Buffer to return safely
        if t_rem < 0.05:
            return best_f
            
        # Determine Initial Population Size for this session based on remaining time.
        # We calculate how many evaluations are possible.
        est_evals = t_rem / (dt + 1e-9)
        
        # Upper bound N: Standard L-SHADE recommendation is 18*D
        n_target_init = int(r_size * dim)
        
        # Capacity check: ensure we can run at least ~30 generations for convergence.
        # If function is slow, we reduce N. If fast, we cap N to avoid overhead.
        n_cap = int(est_evals / 30)
        n_init = min(n_target_init, n_cap)
        # Clamps: [20, 500]
        n_init = max(20, min(n_init, 500)) 
        
        # Session State
        pop_size = n_init
        pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fit = np.zeros(pop_size)
        
        # Elitism: Inject the global best into the new population
        # This acts as a seed for local search in the new random population
        pop[0] = best_x
        fit[0] = best_f
        
        # Evaluate initial population (start from 1, index 0 is known)
        for i in range(1, pop_size):
            if (time.time() - t0) >= max_time: return best_f
            val = func(pop[i])
            fit[i] = val
            if val < best_f:
                best_f = val
                best_x = pop[i].copy()
                
        # Initialize L-SHADE Memories
        m_cr = np.full(h_mem, 0.5)
        m_f = np.full(h_mem, 0.5)
        k_mem = 0
        archive = []
        
        # Session Time Management
        t_sess_start = time.time()
        # We plan to use the rest of the available time for this session.
        # If it converges early, we break and restart using the remaining time.
        sess_budget = max_time - (t_sess_start - t0)
        if sess_budget < 0.01: return best_f
        
        # --- 3. Evolutionary Cycle ---
        while True:
            t_now = time.time()
            if (t_now - t0) >= max_time: return best_f
            
            # Progress calculation (0.0 -> 1.0)
            sess_elapsed = t_now - t_sess_start
            progress = sess_elapsed / sess_budget
            
            if progress >= 1.0:
                # Session time budget finished naturally
                break
                
            # A. Linear Population Size Reduction (LPSR)
            # Linearly interpolate current target size based on progress
            target_n = int(round((n_min - n_init) * progress + n_init))
            target_n = max(n_min, target_n)
            
            if pop_size > target_n:
                # Reduce population: keep best individuals
                sort_idx = np.argsort(fit)
                pop = pop[sort_idx[:target_n]]
                fit = fit[sort_idx[:target_n]]
                
                # Resize Archive
                target_arc = int(target_n * arc_rate)
                if len(archive) > target_arc:
                    # Randomly remove elements
                    keep_idx = np.random.choice(len(archive), target_arc, replace=False)
                    archive = [archive[k] for k in keep_idx]
                
                pop_size = target_n
                
            # B. Check for Convergence (Early Restart)
            # If population fitness variance is essentially zero, we are stuck.
            if np.std(fit) < 1e-9 and progress < 0.95:
                # Break to restart and use remaining time for a new search
                break
                
            # C. Parameter Generation (Vectorized)
            r_idx = np.random.randint(0, h_mem, pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # CR ~ Normal(mu, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu, 0.1)
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Fix F values (<= 0 -> regenerate, > 1 -> clamp)
            while True:
                bad_f = f <= 0
                if not np.any(bad_f): break
                f[bad_f] = mu_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_f)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # D. Mutation (current-to-pbest/1)
            sorted_idx = np.argsort(fit)
            n_pbest = max(2, int(pop_size * p_best_rate))
            pbest_pool = sorted_idx[:n_pbest]
            
            # Select pbest
            xp = pop[np.random.choice(pbest_pool, pop_size)]
            
            # Select r1 != i
            r1 = np.random.randint(0, pop_size, pop_size)
            col_r1 = r1 == np.arange(pop_size)
            r1[col_r1] = (r1[col_r1] + 1) % pop_size
            xr1 = pop[r1]
            
            # Select r2 != i, r2 != r1 (from Pop U Archive)
            if len(archive) > 0:
                pool = np.vstack((pop, np.array(archive)))
            else:
                pool = pop
                
            r2 = np.random.randint(0, len(pool), pop_size)
            # Approximate collision fix for r2
            col_r2 = (r2 == np.arange(pop_size)) | (r2 == r1)
            if np.any(col_r2):
                r2[col_r2] = np.random.randint(0, len(pool), np.sum(col_r2))
            xr2 = pool[r2]
            
            # Calculate Mutant Vectors
            # v = x + F*(xp - x) + F*(xr1 - xr2)
            f_mat = f[:, np.newaxis]
            mutant = pop + f_mat * (xp - pop) + f_mat * (xr1 - xr2)
            
            # E. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # F. Bound Handling (Midpoint Correction)
            # Lower bounds
            vio_l = trial < lb
            if np.any(vio_l):
                lb_broad = np.tile(lb, (pop_size, 1))
                trial[vio_l] = (pop[vio_l] + lb_broad[vio_l]) * 0.5
                
            # Upper bounds
            vio_u = trial > ub
            if np.any(vio_u):
                ub_broad = np.tile(ub, (pop_size, 1))
                trial[vio_u] = (pop[vio_u] + ub_broad[vio_u]) * 0.5
                
            # G. Selection & Evaluation
            new_pop = np.zeros_like(pop)
            new_fit = np.zeros_like(fit)
            
            succ_cr = []
            succ_f = []
            diff_f = []
            
            for i in range(pop_size):
                # Check time periodically inside the evaluation loop
                if i % 20 == 0:
                    if (time.time() - t0) >= max_time: return best_f
                    
                val = func(trial[i])
                
                if val <= fit[i]:
                    new_pop[i] = trial[i]
                    new_fit[i] = val
                    
                    if val < fit[i]:
                        succ_cr.append(cr[i])
                        succ_f.append(f[i])
                        diff_f.append(fit[i] - val)
                        archive.append(pop[i].copy())
                    
                    if val < best_f:
                        best_f = val
                        best_x = trial[i].copy()
                else:
                    new_pop[i] = pop[i]
                    new_fit[i] = fit[i]
            
            pop = new_pop
            fit = new_fit
            
            # H. Memory Update (Weighted Lehmer Mean)
            if len(diff_f) > 0:
                w = np.array(diff_f)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    
                    # Update M_CR
                    m_cr[k_mem] = np.sum(w * np.array(succ_cr))
                    
                    # Update M_F
                    sf = np.array(succ_f)
                    num = np.sum(w * (sf ** 2))
                    den = np.sum(w * sf)
                    m_f[k_mem] = num / den if den > 0 else 0.5
                    
                    k_mem = (k_mem + 1) % h_mem
                    
    return best_f
