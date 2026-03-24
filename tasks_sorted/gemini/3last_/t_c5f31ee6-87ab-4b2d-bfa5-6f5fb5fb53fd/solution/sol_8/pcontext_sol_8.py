#The following algorithm is an optimized hybrid of **L-SHADE** (Linear Population Size Reduction Success-History Adaptive Differential Evolution) and **MTS-LS1** (Multiple Trajectory Search - Local Search 1), specifically tuned for time-constrained black-box optimization.
#
#### Key Improvements over Previous Versions:
#1.  **Partial Coordinate Descent (Efficiency):** In high-dimensional spaces ($D > 10$), standard MTS-LS1 is too expensive per generation. This version randomly selects a subset (25%) of dimensions to optimize during the local search phase. This maintains the refinement capability without stalling the global evolutionary search.
#2.  **Aggressive Restart on Precision Limit:** Instead of waiting for population variance to drop (which can be slow), the algorithm triggers a "Soft Restart" immediately if the Local Search step size (`sr`) drops below machine precision. This prevents wasting time on insignificant improvements.
#3.  **Time-Based Budgeting:** The population size reduction is strictly coupled to the `max_time` progress, transitioning smoothly from exploration (high diversity) to exploitation (convergence) as the deadline approaches.
#4.  **jSO-Style Weighted Parameters:** The adaptation of $F$ and $CR$ parameters uses a weighted Lehmer mean that prioritizes highly successful changes, allowing the algorithm to "learn" the landscape topology faster.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Time-Constrained L-SHADE with Partial MTS-LS1 Local Search.
    
    1. L-SHADE: Global search with linear population reduction.
    2. Partial MTS-LS1: Local search applied to the best solution when stagnation occurs,
       optimizing only a subset of dimensions to save time.
    3. Soft Restarts: Resets population but keeps the best solution when search radius vanishes.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- Helper: Check Time ---
    def check_limit():
        return datetime.now() - start_time >= time_limit

    # --- Helper: Time Progress ---
    def get_progress():
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best
    best_val = float('inf')
    best_sol = None
    
    # L-SHADE Parameters
    initial_pop_size = int(np.clip(18 * dim, 30, 150)) # Cap for speed
    min_pop_size = 4
    memory_size = 5
    arc_rate = 1.4
    
    # --- Main Loop (Restarts) ---
    while not check_limit():
        
        # 1. Initialize Population
        pop_size = initial_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Soft Restart: Inject best solution
        start_eval = 0
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_val
            start_eval = 1
            
        # Evaluate Initial Population
        for i in range(start_eval, pop_size):
            if check_limit(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
                
        # Initialize Memory (M_CR, M_F)
        m_cr = np.full(memory_size, 0.5)
        m_f = np.full(memory_size, 0.5)
        k_mem = 0
        archive = []
        
        # Local Search Parameters
        sr = diff_b * 0.4  # Search range
        sr_min = 1e-15     # Precision floor
        stag_count = 0     # Stagnation counter
        
        # --- Evolutionary Loop ---
        while not check_limit():
            
            # A. Linear Population Size Reduction (LPSR)
            progress = get_progress()
            target_size = int(round(initial_pop_size - (initial_pop_size - min_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Remove worst individuals
                sorted_indices = np.argsort(fitness)
                keep_indices = sorted_indices[:target_size]
                pop = pop[keep_indices]
                fitness = fitness[keep_indices]
                pop_size = target_size
                
                # Resize Archive
                curr_arc_cap = int(pop_size * arc_rate)
                if len(archive) > curr_arc_cap:
                    del_cnt = len(archive) - curr_arc_cap
                    for _ in range(del_cnt):
                        archive.pop(np.random.randint(0, len(archive)))

            # B. Parameter Generation
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)
            # F ~ Cauchy(mu_f, 0.1)
            f_params = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Robust F handling
            while np.any(f_params <= 0):
                mask = f_params <= 0
                f_params[mask] = mu_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f_params = np.clip(f_params, 0, 1)
            
            # C. Mutation: current-to-pbest/1
            # Sort for pbest
            sorted_idx = np.argsort(fitness)
            # Linear decrease of p from 0.2 to 0.05 approx (jSO strategy)
            # simple version: random between 2/N and 0.2
            p_val = np.random.uniform(2/pop_size, 0.2)
            num_pbest = int(max(2, pop_size * p_val))
            pbest_indices = sorted_idx[:num_pbest]
            
            idx_pbest = np.random.choice(pbest_indices, pop_size)
            x_pbest = pop[idx_pbest]
            
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[idx_r1]
            
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            idx_r2 = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[idx_r2]
            
            f_col = f_params[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # D. Crossover: Binomial
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cr = rand_vals < cr[:, None]
            mask_cr[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cr, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # E. Selection
            succ_f = []
            succ_cr = []
            diff_fitness = []
            improved = False
            
            for i in range(pop_size):
                if check_limit(): return best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        archive.append(pop[i].copy())
                        succ_f.append(f_params[i])
                        succ_cr.append(cr[i])
                        diff_fitness.append(fitness[i] - f_trial)
                        improved = True
                    
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = pop[i].copy()
                        stag_count = 0
            
            # Resize Archive
            curr_arc_cap = int(pop_size * arc_rate)
            while len(archive) > curr_arc_cap:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update Memory (Weighted Lehmer Mean)
            if len(succ_f) > 0:
                w = np.array(diff_fitness)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    f_s = np.array(succ_f)
                    cr_s = np.array(succ_cr)
                    
                    mean_f = np.sum(w * (f_s ** 2)) / np.sum(w * f_s)
                    mean_cr = np.sum(w * cr_s)
                    
                    m_f[k_mem] = 0.5 * m_f[k_mem] + 0.5 * mean_f
                    m_cr[k_mem] = 0.5 * m_cr[k_mem] + 0.5 * mean_cr
                    k_mem = (k_mem + 1) % memory_size
            
            if not improved:
                stag_count += 1
                
            # F. Partial MTS-LS1 Local Search
            # Trigger: Stagnation or periodic check
            if stag_count > 3:
                # Work on best solution
                ls_x = best_sol.copy()
                ls_f = best_val
                ls_improved = False
                
                # Optimization: Only select a random subset of dimensions if dim is high
                # This keeps the generation loop fast.
                if dim > 10:
                    num_scan = max(5, int(dim * 0.25))
                    dims_to_check = np.random.choice(dim, num_scan, replace=False)
                else:
                    dims_to_check = np.arange(dim)
                    
                for d in dims_to_check:
                    if check_limit(): return best_val
                    
                    original_x = ls_x[d]
                    
                    # Try Negative Move
                    ls_x[d] = np.clip(original_x - sr[d], min_b[d], max_b[d])
                    val = func(ls_x)
                    
                    if val < ls_f:
                        ls_f = val
                        best_val = val
                        best_sol = ls_x.copy()
                        ls_improved = True
                        # Don't restore, keep moving
                    else:
                        # Try Positive Move (0.5 step)
                        ls_x[d] = np.clip(original_x + 0.5 * sr[d], min_b[d], max_b[d])
                        val = func(ls_x)
                        
                        if val < ls_f:
                            ls_f = val
                            best_val = val
                            best_sol = ls_x.copy()
                            ls_improved = True
                        else:
                            # Restore and Reduce Search Range for this dimension
                            ls_x[d] = original_x
                            sr[d] *= 0.5
                
                if ls_improved:
                    # Update population with new best
                    best_idx = np.argmin(fitness)
                    pop[best_idx] = best_sol.copy()
                    fitness[best_idx] = best_val
                    stag_count = 0
            
            # G. Restart Check
            # If search ranges are exhausted (precision limit reached), restart.
            if np.all(sr < sr_min) or (pop_size <= min_pop_size and stag_count > 10):
                break
                
    return best_val
