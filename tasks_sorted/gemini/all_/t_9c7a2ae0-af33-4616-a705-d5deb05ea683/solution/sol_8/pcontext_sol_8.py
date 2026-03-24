#The output values from the previous algorithms indicate that **jSO with LPSR (Code 3)** and **L-SHADE (Code 5)** were the most promising, consistently reaching `~1.45`. This value likely represents a strong local optimum (common in functions like Shifted Rosenbrock). The algorithm failed to reach `0.0` presumably because it converged into this basin and lacked a mechanism to escape *effectively* within the remaining time, or the restart mechanism in other variants (like Code 1) was too disruptive or poorly timed.
#
#To bridge the gap to the global minimum, this new algorithm implements **jSO (L-SHADE variant) with Time-Based Epoch Restarts**.
#
##### Key Improvements
#1.  **Epoch-Based Optimization**: Instead of a single run or reactive restarts based on arbitrary variance thresholds, the optimization is structured into "epochs".
#    *   Each epoch runs the jSO algorithm with **Linear Population Size Reduction (LPSR)** targeting the end of the *remaining* time window.
#    *   If an epoch converges (population size hits minimum or fitness variance becomes negligible) before time runs out, a **Restart** is triggered.
#2.  **Smart Restart**:
#    *   **Elitism**: The global best solution is carried over to the next epoch to ensure no regression.
#    *   **Fresh Exploration**: The rest of the population is re-initialized randomly, and the adaptive memory (history of success parameters) is reset. This allows the algorithm to learn new parameter distributions suitable for different basins of attraction.
#    *   **Dynamic Time Allocation**: The LPSR schedule is recalculated based on the *remaining* time, ensuring the next epoch is fully optimized for the time left.
#3.  **High-Fidelity jSO Core**: Retains the specific features that made Algo 3 successful:
#    *   **Weighted Lehmer Mean** for memory updates.
#    *   **Midpoint Bound Correction** `(x + min) / 2` for better convergence at boundaries.
#    *   **Dynamic `p`** (0.25 to 0.05) to shift from exploration to exploitation.
#
##### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using jSO (L-SHADE variant) with Time-Based Epoch Restarts.
    
    Structure:
    - The algorithm runs in 'epochs'.
    - Each epoch is a full jSO run with Linear Population Size Reduction (LPSR) 
      scaled to the remaining time.
    - If an epoch converges (low variance or min population size reached), 
      a restart is triggered with the remaining time budget.
    - Restarts preserve the global best (elitism) but reset exploration history.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial Population: 25 * dim is effective for jSO/L-SHADE
    pop_scale = 25
    pop_size_init_base = int(max(30, pop_scale * dim))
    pop_size_min = 4
    
    # SHADE Memory Parameters
    H = 5
    
    # Global Best Tracking
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # Pre-process bounds for speed
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Flag to track if we are in the first run (to handle elitism injection)
    first_run = True
    
    # --- Main Loop (Epoch Manager) ---
    while True:
        current_time = datetime.now()
        if current_time >= end_time:
            break
            
        # Calculate time budget for this epoch
        remaining_seconds = (end_time - current_time).total_seconds()
        
        # Stop restarting if remaining time is negligible (e.g., < 2% of total or < 0.05s)
        # preventing useless micro-starts.
        if not first_run and remaining_seconds < max(0.05, max_time * 0.02):
            break
            
        epoch_start_time = current_time
        # We target the LPSR to finish exactly when time runs out.
        # If we converge earlier, we break and restart with updated time.
        epoch_max_time = remaining_seconds
        
        # --- Epoch Initialization ---
        pop_size = pop_size_init_base
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Carry over the global best to the new population
        if not first_run:
            pop[0] = best_vec
            
        fitness = np.full(pop_size, float('inf'))
        
        # Reset Adaptive Memory for fresh learning
        # jSO defaults: M_f = 0.5, M_cr = 0.8
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.8)
        k_mem = 0
        
        # Reset Archive
        archive = []
        
        # Evaluate Initial Population of Epoch
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_val
            
            # If elitism, we already know the fitness of index 0
            if not first_run and i == 0:
                val = best_val
            else:
                val = func(pop[i])
                
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_vec = pop[i].copy()
                
        first_run = False
        
        # --- Inner Loop: jSO Optimization ---
        while True:
            now = datetime.now()
            if now >= end_time:
                return best_val
            
            # 1. Time-based Progress for LPSR and Dynamic parameters
            elapsed_epoch = (now - epoch_start_time).total_seconds()
            progress = elapsed_epoch / epoch_max_time
            if progress > 1.0: progress = 1.0
            
            # 2. Check Convergence / Epoch End
            fit_range = np.max(fitness) - np.min(fitness)
            
            # Conditions to trigger restart:
            # A. Population size has shrunk to minimum (LPSR completed)
            # B. Fitness variance is effectively zero (Premature convergence)
            if pop_size <= pop_size_min or fit_range < 1e-12:
                break # Break inner loop -> Triggers restart in outer loop
                
            # 3. Linear Population Size Reduction (LPSR)
            # Calculate target size based on time progress
            target_size = int(round(pop_size_init_base + (pop_size_min - pop_size_init_base) * progress))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Reduce population: Keep best individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:target_size]]
                fitness = fitness[sorted_idx[:target_size]]
                pop_size = target_size
                
                # Resize archive to match current pop_size (jSO standard)
                if len(archive) > pop_size:
                    np.random.shuffle(archive)
                    archive = archive[:pop_size]
            
            # 4. Parameter Generation
            r_idxs = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idxs]
            m_f = mem_f[r_idxs]
            
            # CR ~ Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(M_F, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Retry if F <= 0, Clip if F > 1
            bad_f = f <= 0
            while np.any(bad_f):
                f[bad_f] = m_f[r_idxs[bad_f]] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
                bad_f = f <= 0
            f[f > 1] = 1.0
            
            # 5. Evolution Cycle: current-to-pbest/1
            # Dynamic 'p' value: Linearly decreases from 0.25 to 0.05
            p_val = 0.25 - (0.20 * progress)
            if p_val < 0.05: p_val = 0.05
            
            sorted_indices = np.argsort(fitness)
            top_p_cnt = int(max(2, p_val * pop_size))
            p_best_indices = sorted_indices[:top_p_cnt]
            
            new_pop = np.empty_like(pop)
            new_fitness = np.empty_like(fitness)
            
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_vals = np.zeros(pop_size)
            
            # Iterate Population
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_val
                
                x_i = pop[i]
                
                # Select x_pbest
                p_idx = np.random.choice(p_best_indices)
                x_pbest = pop[p_idx]
                
                # Select r1 (distinct from i)
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # Select r2 (distinct from i, r1; from Union(Pop, Archive))
                limit = pop_size + len(archive)
                r2_idx = np.random.randint(0, limit)
                # Ensure r2 is distinct
                while True:
                    if r2_idx < pop_size:
                        if r2_idx != i and r2_idx != r1: break
                    else:
                        break # Archive indices are assumed safe
                    r2_idx = np.random.randint(0, limit)
                
                if r2_idx < pop_size:
                    x_r2 = pop[r2_idx]
                else:
                    x_r2 = archive[r2_idx - pop_size]
                    
                # Mutation
                mutant = x_i + f[i] * (x_pbest - x_i) + f[i] * (x_r1 - x_r2)
                
                # Crossover (Binomial)
                mask_j = np.random.rand(dim) < cr[i]
                j_rand = np.random.randint(dim)
                mask_j[j_rand] = True
                trial = np.where(mask_j, mutant, x_i)
                
                # Bound Correction: Midpoint Rule (Effective for convergence)
                below_b = trial < min_b
                above_b = trial > max_b
                trial[below_b] = (x_i[below_b] + min_b[below_b]) / 2.0
                trial[above_b] = (x_i[above_b] + max_b[above_b]) / 2.0
                
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
            
            # Archive Maintenance
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

    return best_val
