#The following Python code implements an optimized **L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution) with Restarts**.
#
#**Key Improvements for Robust Performance:**
#1.  **Time-Dependent Linear Population Reduction (LPSR):** Instead of relying on generation counts (which are unknown in time-constrained settings), the population size is reduced linearly based on the `elapsed_time / max_time` ratio. This ensures the algorithm shifts from exploration (large population) to exploitation (small population) exactly as the deadline approaches.
#2.  **Adaptive Parameters (SHADE):** It uses historical memory ($M_F, M_{CR}$) to adapt mutation factor $F$ and crossover rate $CR$ during the run, avoiding manual tuning.
#3.  **Global Restarts:** If the population converges (low variance) or stagnates (no fitness improvement) before time runs out, the algorithm restarts with a fresh population. Crucially, the *restart population size* also adheres to the global LPSR schedule, meaning late restarts are smaller and faster, focusing on quick local convergence.
#4.  **Vectorized Operations:** All genetic operations are fully vectorized using NumPy to maximize the number of function evaluations within the limited time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE with Time-based Population Reduction and Restarts.
    """
    # 1. Setup
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds for speed
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track global best solution
    global_best_val = float('inf')
    
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    # 2. Algorithm Hyperparameters
    # Initial population: Large enough for exploration, scaled by dim.
    # We clip it to [50, 200] to ensure it runs well on short time limits.
    initial_pop_size = int(np.clip(20 * dim, 50, 200))
    min_pop_size = 4
    
    # Archive size relative to population
    arc_rate = 2.0
    
    # Memory size for parameter adaptation
    H = 5
    
    # 3. Main Loop (Restarts)
    while not is_time_up():
        
        # --- Initialization for this Restart ---
        pop_size = initial_pop_size
        
        # Initialize population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if is_time_up(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Initialize SHADE Memory
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize External Archive
        max_arc_size = int(initial_pop_size * arc_rate)
        archive = np.empty((max_arc_size, dim))
        archive_cnt = 0
        
        # Stagnation tracking
        run_best = np.min(fitness)
        stag_count = 0
        
        # --- Evolutionary Loop ---
        while not is_time_up():
            
            # A. Calculate Global Progress (0.0 to 1.0)
            elapsed_sec = (datetime.now() - start_time).total_seconds()
            progress = min(elapsed_sec / max_time, 1.0)
            
            # B. Linear Population Size Reduction (LPSR) based on Time
            # Target size scales linearly from initial to min based on time used
            target_pop = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
            target_pop = max(min_pop_size, target_pop)
            
            # Shrink Population if needed
            if pop_size > target_pop:
                n_remove = pop_size - target_pop
                # Remove worst individuals (highest fitness)
                sort_idx = np.argsort(fitness)
                keep_idx = sort_idx[:target_pop]
                
                pop = pop[keep_idx]
                fitness = fitness[keep_idx]
                pop_size = target_pop
                
                # Resize Archive capacity
                curr_arc_cap = int(pop_size * arc_rate)
                if archive_cnt > curr_arc_cap:
                    archive_cnt = curr_arc_cap
            
            # C. Parameter Adaptation
            # Adapt 'p' for current-to-pbest mutation: starts at 0.2, drops to 0.05
            p_val = 0.2 * (1.0 - progress) + 0.05
            p_val = np.clip(p_val, 2.0/max(pop_size, 2), 0.2)
            
            # Select memory slots
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR (Normal distribution) and F (Cauchy distribution)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Retry F if <= 0 until valid (vectorized)
            while True:
                bad_mask = f <= 0
                if not np.any(bad_mask): break
                f[bad_mask] = m_f[bad_mask] + 0.1 * np.random.standard_cauchy(np.sum(bad_mask))
            f = np.minimum(f, 1.0)
            
            # D. Mutation: current-to-pbest/1
            # Identify p-best individuals
            sorted_indices = np.argsort(fitness)
            num_pbest = max(2, int(p_val * pop_size))
            pbest_indices = np.random.choice(sorted_indices[:num_pbest], pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (random from pop)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_indices]
            
            # Select r2 (random from Union(Pop, Archive))
            pool_size = pop_size + archive_cnt
            r2_indices = np.random.randint(0, pool_size, pop_size)
            
            x_r2 = np.empty((pop_size, dim))
            mask_in_pop = r2_indices < pop_size
            x_r2[mask_in_pop] = pop[r2_indices[mask_in_pop]]
            
            if archive_cnt > 0:
                mask_in_arc = ~mask_in_pop
                x_r2[mask_in_arc] = archive[r2_indices[mask_in_arc] - pop_size]
            else:
                # Fallback if archive empty (rare edge case with logic above)
                x_r2[~mask_in_pop] = pop[r2_indices[~mask_in_pop] % pop_size]
            
            # Compute Mutant Vector: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # E. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            rand_u = np.random.rand(pop_size, dim)
            mask_cross = rand_u < cr[:, None]
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # F. Selection
            trial_fitness = np.zeros(pop_size)
            mask_improved = np.zeros(pop_size, dtype=bool)
            
            diffs = []
            succ_f = []
            succ_cr = []
            
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                f_trial = func(trial[i])
                trial_fitness[i] = f_trial
                
                if f_trial <= fitness[i]:
                    mask_improved[i] = True
                    diffs.append(fitness[i] - f_trial)
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
            
            # Update Archive with replaced parents
            n_improved = np.sum(mask_improved)
            if n_improved > 0:
                parents = pop[mask_improved]
                curr_cap = int(pop_size * arc_rate)
                
                # Insert into archive
                if archive_cnt < curr_cap:
                    take = min(n_improved, curr_cap - archive_cnt)
                    archive[archive_cnt : archive_cnt + take] = parents[:take]
                    archive_cnt += take
                    # If more parents than space, replace randomly
                    if n_improved > take:
                        rem = n_improved - take
                        idx = np.random.randint(0, curr_cap, rem)
                        archive[idx] = parents[take:]
                else:
                    # Archive full, replace randomly
                    idx = np.random.randint(0, curr_cap, n_improved)
                    archive[idx] = parents
            
            # Update Population and Fitness
            pop[mask_improved] = trial[mask_improved]
            fitness[mask_improved] = trial_fitness[mask_improved]
            
            # G. Update Memory (Weighted Lehmer Mean)
            if len(diffs) > 0:
                diff_np = np.array(diffs)
                total_diff = np.sum(diff_np)
                if total_diff > 0:
                    w = diff_np / total_diff
                    sf_np = np.array(succ_f)
                    scr_np = np.array(succ_cr)
                    
                    mean_f = np.sum(w * (sf_np ** 2)) / np.sum(w * sf_np)
                    mean_cr = np.sum(w * scr_np)
                    
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # H. Check Stagnation/Convergence for Restart
            current_run_best = np.min(fitness)
            if current_run_best < run_best - 1e-8:
                run_best = current_run_best
                stag_count = 0
            else:
                stag_count += 1
            
            # Restart if variance is 0 (converged) or no improvement for ~45 gens
            if np.std(fitness) < 1e-9 or stag_count > 45:
                break
                
    return global_best_val
