#An improved algorithm based on **jSO (a specialized variant of L-SHADE)** is proposed below.
#
#### Rationale for Improvement
#The previous best-performing algorithm (Code 1, Output ~1.38) utilized **L-SHADE with Time-Based Epochs**. While successful in approaching the global minimum basin, it likely struggled to converge to exactly `0.0` due to a lack of explicit **stagnation detection**. In difficult landscapes (like the Rosenbrock valley), the algorithm can make infinitesimal progress that doesn't trigger a variance-based convergence stop, wasting valuable time.
#
#This improved version, **jSO with Stagnation-Based Restart and Adaptive LPSR**, introduces:
#1.  **Stagnation Detection**: Explicitly monitors the global best solution. If no significant improvement occurs for a defined number of generations (scaled by dimension), the current "epoch" is aborted early to restart.
#2.  **Optimized Restart Strategy**: When restarting, the algorithm preserves the elite solution (elitism) but resets the population and adaptation memory. This ensures the algorithm doesn't get "trapped" in a slow-converging path and allows the adaptive parameters ($F, CR$) to relearn the landscape dynamics.
#3.  **Population Capping**: Caps the maximum population size to ensure high generation throughput even in high dimensions, preventing the algorithm from spending too much time on a single generation.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using jSO (L-SHADE variant) with Stagnation-Based Restart.
    
    Key Features:
    - Time-based Epochs: LPSR schedule scales to the remaining time of the current epoch.
    - Stagnation Detection: Restarts early if best solution stops improving.
    - Elitism: Preserves the global best across restarts.
    - Midpoint Bound Correction: Enhances convergence near boundaries.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population Size:
    # Use 20*dim but cap at 300 to ensure sufficient generations within limit.
    # Lower bound 40 ensures diversity for small dims.
    pop_size_init = int(min(300, max(40, 20 * dim)))
    pop_size_min = 4
    
    # Stagnation Limit:
    # If no improvement for this many generations, trigger restart.
    stagnation_limit = 50 + int(dim)
    
    # SHADE Memory Parameters
    H = 5
    
    # Pre-process bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    first_run = True
    
    # --- Main Outer Loop (Epochs) ---
    while True:
        # Check remaining time
        now = datetime.now()
        if now >= end_time:
            break
            
        remaining_seconds = (end_time - now).total_seconds()
        
        # Stop restarting if remaining time is too short for a meaningful run
        # (e.g. less than 0.1 seconds or < 1% of total time)
        if not first_run and remaining_seconds < 0.1:
            break
            
        # --- Epoch Initialization ---
        pop_size = pop_size_init
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject best found so far
        if not first_run:
            pop[0] = best_vec
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population of Epoch
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_val
            
            # Skip re-evaluation of elite
            if not first_run and i == 0:
                val = best_val
            else:
                val = func(pop[i])
                
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_vec = pop[i].copy()
                
        # Reset Memory for fresh adaptation
        # jSO defaults: M_f = 0.5, M_cr = 0.8
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.8)
        k_mem = 0
        
        # Archive for current-to-pbest mutation
        archive = []
        
        # Epoch Control Variables
        run_start = datetime.now()
        no_improve_count = 0
        last_best_in_run = best_val
        
        # --- Inner Loop: Optimization ---
        while True:
            t_now = datetime.now()
            if t_now >= end_time: return best_val
            
            # 1. Calculate Progress (Time-based LPSR)
            elapsed_run = (t_now - run_start).total_seconds()
            progress = elapsed_run / remaining_seconds
            if progress > 1.0: progress = 1.0
            
            # 2. Linear Population Size Reduction
            target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Reduce population: keep best
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:target_size]]
                fitness = fitness[sorted_idx[:target_size]]
                pop_size = target_size
                
                # Resize archive
                if len(archive) > pop_size:
                    # Randomly remove elements (shuffling first ensures randomness)
                    import random
                    random.shuffle(archive)
                    archive = archive[:pop_size]
            
            # 3. Check Restart Conditions
            # A. Convergence (Variance is zero)
            fit_range = np.max(fitness) - np.min(fitness)
            if fit_range < 1e-12:
                break # Trigger restart
            
            # B. LPSR Finished (Population Minimized)
            if pop_size <= pop_size_min:
                break # Trigger restart
                
            # C. Stagnation (No improvement for X generations)
            if best_val < last_best_in_run:
                last_best_in_run = best_val
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > stagnation_limit:
                break # Trigger restart
            
            # 4. Parameter Generation
            r_idxs = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idxs]
            m_f = mem_f[r_idxs]
            
            # CR ~ Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(M_F, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Retry F <= 0, Clip F > 1
            bad_f = f <= 0
            while np.any(bad_f):
                f[bad_f] = m_f[r_idxs[bad_f]] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
                bad_f = f <= 0
            f[f > 1] = 1.0
            
            # 5. Evolution: current-to-pbest/1
            # Dynamic 'p' value: 0.25 -> 0.05
            p_val = 0.25 - (0.20 * progress)
            if p_val < 0.05: p_val = 0.05
            
            sorted_indices = np.argsort(fitness)
            top_p_cnt = int(max(2, p_val * pop_size))
            p_best_indices = sorted_indices[:top_p_cnt]
            
            new_pop = np.empty_like(pop)
            new_fitness = np.empty_like(fitness)
            
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_vals = np.zeros(pop_size)
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_val
                
                x_i = pop[i]
                
                # Select p-best
                p_idx = np.random.choice(p_best_indices)
                x_pbest = pop[p_idx]
                
                # Select r1 != i
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # Select r2 != i, r1 from Union(Pop, Archive)
                limit = pop_size + len(archive)
                r2 = np.random.randint(0, limit)
                while True:
                    if r2 < pop_size:
                        if r2 != i and r2 != r1: break
                    else:
                        break
                    r2 = np.random.randint(0, limit)
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                    
                # Mutation
                mutant = x_i + f[i] * (x_pbest - x_i) + f[i] * (x_r1 - x_r2)
                
                # Crossover (Binomial)
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                trial = np.where(mask, mutant, x_i)
                
                # Bound Correction (Midpoint Rule)
                # Significantly helps with convergence near bounds
                below = trial < min_b
                above = trial > max_b
                if np.any(below):
                    trial[below] = (x_i[below] + min_b[below]) / 2.0
                if np.any(above):
                    trial[above] = (x_i[above] + max_b[above]) / 2.0
                    
                # Selection
                f_trial = func(trial)
                
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    
                    if f_trial < fitness[i]:
                        succ_mask[i] = True
                        diff_vals[i] = fitness[i] - f_trial
                        archive.append(x_i.copy())
                        
                    if f_trial < best_val:
                        best_val = f_trial
                        best_vec = trial.copy()
                else:
                    new_pop[i] = x_i
                    new_fitness[i] = fitness[i]
            
            pop = new_pop
            fitness = new_fitness
            
            # Trim archive
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))
                
            # 6. Memory Update (Weighted Lehmer Mean)
            succ_idx = np.where(succ_mask)[0]
            if len(succ_idx) > 0:
                w = diff_vals[succ_idx]
                total_w = np.sum(w)
                if total_w > 0:
                    w = w / total_w
                    
                    s_cr = cr[succ_idx]
                    s_f = f[succ_idx]
                    
                    mean_cr = np.sum(w * s_cr)
                    mean_f = np.sum(w * (s_f**2)) / (np.sum(w * s_f) + 1e-12)
                    
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                    
                    k_mem = (k_mem + 1) % H
        
        first_run = False

    return best_val
