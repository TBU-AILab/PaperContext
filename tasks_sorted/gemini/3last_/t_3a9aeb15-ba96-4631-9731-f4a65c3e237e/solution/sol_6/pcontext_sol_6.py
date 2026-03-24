#The following algorithm implements an improved **Vectorized L-SHADE with Adaptive Budgeting and Calibration**.
#
#### Key Improvements
#1.  **Calibration Phase**: Before starting, the algorithm briefly benchmarks the objective function (`func`). It adapts the initial population size (`N_init`) based on the function's evaluation speed. This prevents the algorithm from being too slow on computationally expensive functions or too sparse on fast ones.
#2.  **Dynamic Session Management**: The algorithm divides the `max_time` into "sessions". It calculates a Linear Population Size Reduction (LPSR) schedule based on the *remaining time*. If the population converges early or the schedule completes, it triggers a **Soft Restart**, carrying over the global best solution (Elitism) and rescaling the next session to fit the new remaining time perfectly.
#3.  **Vectorized Operations**: All mutation, crossover, and bound handling operations are fully vectorized using NumPy. This minimizes Python interpreter overhead, maximizing the number of generations per second.
#4.  **Robust L-SHADE Logic**: Implements the `current-to-pbest/1` mutation strategy with weighted Lehmer mean updates for historical memory ($M_{CR}, M_F$), ensuring efficient adaptation to the fitness landscape.
#
#### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Vectorized L-SHADE with Adaptive Budgeting and Calibration.
    """
    start_time = time.time()
    
    # --- 1. Setup & Calibration ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    
    # Estimate function cost to scale population size properly
    # Allocate a tiny fraction of time for calibration (max 1% or 0.2s)
    cal_budget = min(max_time * 0.01, 0.2)
    cal_start = time.time()
    n_cal = 0
    while time.time() - cal_start < cal_budget and n_cal < 10:
        # Evaluate a random point
        x_dummy = min_b + np.random.rand(dim) * (max_b - min_b)
        func(x_dummy)
        n_cal += 1
    
    elapsed_cal = time.time() - cal_start
    avg_eval_time = elapsed_cal / n_cal if n_cal > 0 else 0.0
    
    # Heuristic: Estimate total evaluations possible
    est_total_evals = max_time / (avg_eval_time + 1e-9)
    
    # Configure Initial Population (N_init)
    # Standard L-SHADE recommends 18*dim, but we cap it for slow functions
    base_n = int(18 * dim)
    
    # We want to ensure we get enough generations even if func is slow.
    # Cap N such that N_init is at most ~5% of estimated capacity to allow reduction
    cap_by_speed = int(est_total_evals / 20)
    n_init = min(base_n, cap_by_speed)
    n_init = max(30, min(n_init, 200)) # Hard limits [30, 200]
    
    n_min = 5
    
    # L-SHADE Parameters
    H = 5             # Memory size
    arc_rate = 2.0    # Archive size factor
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- 2. Main Optimization Loop (Restarts) ---
    while True:
        # Time Check
        current_time = time.time()
        elapsed_global = current_time - start_time
        remaining_time = max_time - elapsed_global
        
        # Buffer to ensure clean return
        if remaining_time < 0.05:
            return best_val
            
        # Session Configuration
        # We treat the remaining time as the budget for this session.
        # If the population converges early, we restart and recalculate.
        session_start = current_time
        session_budget = remaining_time
        
        # Initialize Population
        pop_size = n_init
        pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
        fitness = np.zeros(pop_size)
        
        # Elitism: Inject global best from previous session if available
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol
            fitness[0] = best_val
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
                
        # Initialize Memory and Archive
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k_mem = 0
        archive = [] 
        
        # --- 3. Evolutionary Cycle (L-SHADE) ---
        while True:
            t_now = time.time()
            if (t_now - start_time) >= max_time:
                return best_val
            
            # A. Linear Population Size Reduction (LPSR)
            # Calculate progress relative to the CURRENT session budget
            session_elapsed = t_now - session_start
            progress = session_elapsed / session_budget
            
            target_n = int(round((n_min - n_init) * progress + n_init))
            target_n = max(n_min, target_n)
            
            if pop_size > target_n:
                # Reduce Population (keep best)
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices[:target_n]]
                fitness = fitness[sort_indices[:target_n]]
                pop_size = target_n
                
                # Reduce Archive
                max_arc_size = int(pop_size * arc_rate)
                if len(archive) > max_arc_size:
                    # Randomly remove excess
                    keep_idxs = np.random.choice(len(archive), max_arc_size, replace=False)
                    archive = [archive[ix] for ix in keep_idxs]
            
            # B. Restart Check
            # If schedule is nearly done (progress > 0.95) OR population stagnated (std ~ 0)
            if progress > 0.95 or np.std(fitness) < 1e-9:
                break # Break inner loop to trigger restart with new time budget
            
            # C. Parameter Generation (Vectorized)
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]
            
            # CR ~ Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Handle F violations (F <= 0 -> regenerate, F > 1 -> clamp)
            while True:
                bad_f_mask = f <= 0
                if not np.any(bad_f_mask): break
                f[bad_f_mask] = m_f[bad_f_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_f_mask)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # D. Mutation: current-to-pbest/1 (Vectorized)
            sorted_indices = np.argsort(fitness)
            p_best_count = max(2, int(pop_size * 0.11))
            p_best_pool = sorted_indices[:p_best_count]
            
            # Choose p-best for each individual
            p_best_idx = np.random.choice(p_best_pool, pop_size)
            x_pbest = pop[p_best_idx]
            
            # Choose r1 != i
            r1_idx = np.random.randint(0, pop_size, pop_size)
            col_r1 = r1_idx == np.arange(pop_size)
            r1_idx[col_r1] = (r1_idx[col_r1] + 1) % pop_size # Simple collision fix
            x_r1 = pop[r1_idx]
            
            # Choose r2 != i, r2 != r1 from P U A
            n_arc = len(archive)
            if n_arc > 0:
                # Efficient pool construction
                pool = np.vstack((pop, np.array(archive)))
            else:
                pool = pop
                
            r2_idx = np.random.randint(0, len(pool), pop_size)
            # Collision fix for r2 (approximate for speed)
            col_r2 = (r2_idx == np.arange(pop_size)) | (r2_idx == r1_idx)
            if np.any(col_r2):
                r2_idx[col_r2] = np.random.randint(0, len(pool), np.sum(col_r2))
                
            x_r2 = pool[r2_idx]
            
            # Calculate Mutant Vector
            # v = x + F*(xp - x) + F*(xr1 - xr2)
            f_mat = f[:, np.newaxis]
            mutant = pop + f_mat * (x_pbest - pop) + f_mat * (x_r1 - x_r2)
            
            # E. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # F. Bound Handling (Midpoint Correction)
            # Instead of clipping, place between parent and bound to preserve distribution
            lower_vio = trial < min_b
            upper_vio = trial > max_b
            
            if np.any(lower_vio):
                trial[lower_vio] = (pop[lower_vio] + np.tile(min_b, (pop_size,1))[lower_vio]) / 2.0
            if np.any(upper_vio):
                trial[upper_vio] = (pop[upper_vio] + np.tile(max_b, (pop_size,1))[upper_vio]) / 2.0
            
            # G. Selection & Memory Update
            new_fitness = np.zeros(pop_size)
            S_CR, S_F, S_df = [], [], []
            
            for i in range(pop_size):
                # Check time periodically inside the loop if population is large
                if i % 10 == 0 and (time.time() - start_time) >= max_time:
                    return best_val
                    
                f_trial = func(trial[i])
                new_fitness[i] = f_trial
                
                if f_trial <= fitness[i]:
                    # Update Memory info if strictly better
                    if f_trial < fitness[i]:
                        S_CR.append(cr[i])
                        S_F.append(f[i])
                        S_df.append(fitness[i] - f_trial)
                        archive.append(pop[i].copy())
                    
                    # Update Global Best
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = trial[i].copy()
                else:
                    # Keep parent (Handled by masking below)
                    pass
            
            # Update Population
            improve_mask = new_fitness <= fitness
            pop[improve_mask] = trial[improve_mask]
            fitness[improve_mask] = new_fitness[improve_mask]
            
            # Update Memories (Weighted Lehmer Mean)
            if len(S_df) > 0:
                w = np.array(S_df)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    
                    s_cr = np.array(S_CR)
                    s_f = np.array(S_F)
                    
                    mu_cr_new = np.sum(w * s_cr)
                    M_CR[k_mem] = mu_cr_new
                    
                    num = np.sum(w * (s_f ** 2))
                    den = np.sum(w * s_f)
                    mu_f_new = num / den if den > 0 else 0.5
                    M_F[k_mem] = mu_f_new
                    
                    k_mem = (k_mem + 1) % H

    return best_val
