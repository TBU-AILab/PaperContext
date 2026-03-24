#The following Python code implements **jSO-RSP** (jSO variant with Rescaled Epoch-based Restarts).
#
#**Key Improvements:**
#1.  **jSO Parameter Adaptation**: Implements the "High F" heuristic (forcing $F \ge 0.7$ in the early search phase) to prevent premature convergence, and uses weighted Memory Updates (Weighted Arithmetic Mean for CR, Weighted Lehmer Mean for F) to learn efficient parameters.
#2.  **Epoch-based Time Management**: The algorithm treats the remaining time as a dynamic epoch. The **Linear Population Size Reduction (LPSR)** schedule is rescaled to fit the exact remaining time, ensuring the search transitions from exploration to exploitation regardless of when a restart occurs.
#3.  **Vectorized Retry Logic**: Uses efficient NumPy masking to handle parameter constraints (like Cauchy generation retries) without slow loops.
#4.  **Robust Restarts**: Triggers restarts upon population stagnation or convergence, preserving the global best solution (Elitism) to escape local optima.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using jSO-RSP (jSO variant with Rescaled Epoch-based Restarts).
    
    This algorithm employs:
    - Linear Population Size Reduction (LPSR) scaled to remaining time.
    - jSO parameter adaptation (High F heuristic, weighted memory).
    - An epoch-based restart mechanism to escape local optima.
    """
    
    # --- Configuration ---
    # Population Size Heuristic
    # Start with a moderate size for exploration, reducing to min_pop_size.
    # Clip ensures scalability across dimensions without excessive cost.
    init_pop_size = int(np.clip(20 * dim, 50, 200))
    min_pop_size = 5
    
    # jSO / L-SHADE Parameters
    H = 5                   # Historical memory size
    arc_rate = 1.4          # Archive size relative to population
    
    # Stagnation limit (generations without improvement before restart)
    stag_limit = 50 
    
    # --- Time Management ---
    start_run = datetime.now()
    end_run = start_run + timedelta(seconds=max_time)
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = None
    
    def check_timeout():
        return datetime.now() >= end_run

    # --- Main Optimization Loop (Epochs) ---
    while not check_timeout():
        
        # Mark the start of a new Epoch (Restart)
        epoch_start = datetime.now()
        
        # Initialize Population for this Epoch
        pop_size = init_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best solution from previous epochs
        start_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if check_timeout(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
                
        # Initialize jSO Memory
        # CR starts high (0.8) to encourage mixing, F starts mid (0.5)
        mem_cr = np.full(H, 0.8) 
        mem_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Stagnation Counters
        stag_count = 0
        last_best_val = global_best_val
        
        # --- Epoch Loop ---
        while True:
            if check_timeout(): return global_best_val
            
            # 1. Calculate Progress (0.0 -> 1.0)
            # Progress scales based on the time remaining for this specific run context
            now = datetime.now()
            total_duration = (end_run - epoch_start).total_seconds()
            elapsed = (now - epoch_start).total_seconds()
            
            if total_duration <= 1e-9:
                progress = 1.0
            else:
                progress = elapsed / total_duration
                if progress > 1.0: progress = 1.0
            
            # 2. Linear Population Size Reduction (LPSR)
            # Calculate target size based on progress
            target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Sort population by fitness (best first)
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx]
                fitness = fitness[sort_idx]
                
                # Truncate population
                pop_size = target_size
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive
                target_arc = int(pop_size * arc_rate)
                if len(archive) > target_arc:
                    random.shuffle(archive)
                    archive = archive[:target_arc]
            
            # 3. Parameter Adaptation
            # p (greediness) reduces from 0.25 to 0.05
            p_val = 0.25 - (0.20 * progress)
            
            # Select Memory Indices
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(m_f, 0.1) with Retry Logic
            f = np.zeros(pop_size)
            
            # Initial sampling
            f_raw = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Identify invalid F values (<= 0)
            bad_mask = f_raw <= 0
            f[~bad_mask] = f_raw[~bad_mask]
            
            # Retry loop for invalid values (vectorized)
            for _ in range(5): 
                if not np.any(bad_mask): break
                n_bad = np.sum(bad_mask)
                # Resample for bad indices
                f_new = m_f[bad_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
                
                # Update bad mask
                current_bad_indices = np.where(bad_mask)[0]
                still_bad = f_new <= 0
                
                # Store valid ones
                f[current_bad_indices] = f_new
                
                # Update mask to only track persistent bad values
                new_bad_mask = np.zeros(pop_size, dtype=bool)
                new_bad_mask[current_bad_indices[still_bad]] = True
                bad_mask = new_bad_mask
            
            # Fallback for persistent bad values and clamping > 1
            f[f <= 0] = 0.5
            f[f > 1.0] = 1.0
            
            # jSO Heuristic: Force high F (>= 0.7) in the first 60% of the epoch
            # to promote exploration
            if progress < 0.6:
                f = np.where(f < 0.7, 0.7, f)
            
            # 4. Mutation: current-to-pbest/1
            # Sort to find p-best
            sorted_idx = np.argsort(fitness)
            p_count = max(2, int(pop_size * p_val))
            pbest_pool = sorted_idx[:p_count]
            
            # Random p-best for each individual
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # r1 != i
            shift = np.random.randint(1, pop_size, pop_size)
            r1_idx = (np.arange(pop_size) + shift) % pop_size
            x_r1 = pop[r1_idx]
            
            # r2 != i, r2 != r1 (from Union of Pop + Archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            
            # Rejection sampling for r2
            r2_idx = np.random.randint(0, len(union_pop), pop_size)
            curr_idx = np.arange(pop_size)
            
            # Bad r2 condition
            bad_r2 = (r2_idx < pop_size) & (r2_idx == curr_idx) | (r2_idx == r1_idx)
            
            retry = 0
            while np.any(bad_r2) and retry < 5:
                n_bad = np.sum(bad_r2)
                r2_idx[bad_r2] = np.random.randint(0, len(union_pop), n_bad)
                bad_r2 = (r2_idx < pop_size) & (r2_idx == curr_idx) | (r2_idx == r1_idx)
                retry += 1
            
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant Vector
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, np.newaxis]
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # 6. Evaluation and Selection
            new_pop = pop.copy()
            new_fit = fitness.copy()
            
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_fit = np.zeros(pop_size)
            
            for i in range(pop_size):
                if check_timeout(): return global_best_val
                val = func(trial[i])
                
                # Greedy Selection
                if val <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fit[i] = val
                    if val < fitness[i]:
                        succ_mask[i] = True
                        diff_fit[i] = fitness[i] - val
                        archive.append(pop[i].copy())
                    
                    if val < global_best_val:
                        global_best_val = val
                        global_best_sol = trial[i].copy()
            
            pop = new_pop
            fitness = new_fit
            
            # 7. Weighted Memory Update
            if np.any(succ_mask):
                w_diff = diff_fit[succ_mask]
                w_cr = cr[succ_mask]
                w_f = f[succ_mask]
                
                total_diff = np.sum(w_diff)
                if total_diff > 0:
                    weights = w_diff / total_diff
                    
                    # CR: Weighted Arithmetic Mean
                    m_cr_new = np.sum(weights * w_cr)
                    mem_cr[k_mem] = np.clip(m_cr_new, 0, 1)
                    
                    # F: Weighted Lehmer Mean
                    num = np.sum(weights * (w_f**2))
                    den = np.sum(weights * w_f)
                    if den > 1e-10:
                        m_f_new = num / den
                    else:
                        m_f_new = 0.5
                    mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                    
                    k_mem = (k_mem + 1) % H
            
            # 8. Restart Checks
            # Stagnation Detection
            if global_best_val < last_best_val:
                last_best_val = global_best_val
                stag_count = 0
            else:
                stag_count += 1
            
            # Trigger A: No improvement for 'stag_limit' generations
            if stag_count >= stag_limit:
                break
            
            # Trigger B: Population minimal and converged (low variance)
            if pop_size <= min_pop_size:
                if np.std(fitness) < 1e-9:
                    break

    return global_best_val
