#The following Python code implements an **L-SHADE with Restarts (Linear Population Size Reduction Success-History based Adaptive Differential Evolution)**.
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR):** The population size starts large (to explore) and linearly decreases by 1 individual every generation (to exploit). This aggressive reduction strategy forces convergence and is a key feature of top-performing algorithms like L-SHADE.
#2.  **SHADE Adaptation:** Uses historical memory ($M_{CR}, M_F$) to adaptively learn optimal control parameters ($CR$ and $F$) based on successful evolutionary steps.
#3.  **Restarts:** The algorithm wraps the L-SHADE process in a restart loop. If the population converges (based on variance or size limit) or stagnates, it restarts with a fresh, large population to explore new basins of attraction within the remaining time.
#4.  **Vectorized Operations:** Maximizes execution speed using NumPy for all genetic operations (mutation, crossover, selection), leaving the objective function evaluation as the only potential bottleneck.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE with Restarts.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best tracking
    global_best_val = float('inf')
    
    def is_time_up():
        return (datetime.now() - start_time) >= limit
    
    # --- Algorithm Hyperparameters ---
    # Initial population size: Sufficiently large to cover search space.
    # L-SHADE typically uses 18*D. We clip it to a safe range [30, 200] for speed.
    initial_pop_size = int(np.clip(18 * dim, 30, 200))
    min_pop_size = 4
    
    # Archive size relative to population (A_rate)
    arc_rate = 2.0
    
    # Memory size for SHADE adaptation
    H = 6
    
    # --- Main Restart Loop ---
    # Continues to restart the optimization process until time runs out
    while not is_time_up():
        
        # 1. Initialization for current run
        pop_size = initial_pop_size
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
        
        # Initialize Archive
        # We pre-allocate maximum capacity but track current count
        max_archive_cap = int(initial_pop_size * arc_rate)
        archive = np.empty((max_archive_cap, dim))
        archive_cnt = 0
        
        # Stagnation tracking
        run_best = np.min(fitness)
        stag_count = 0
        
        # --- Evolutionary Loop ---
        # Runs until population shrinks to min_pop_size or time runs out
        while pop_size > min_pop_size and not is_time_up():
            
            # A. Parameter Generation
            # Select random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR (Normal distribution)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F (Cauchy distribution)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            f = np.minimum(f, 1.0) # Cap at 1.0
            
            # Retry if F <= 0 (Vectorized)
            while True:
                mask_bad = f <= 0
                if not np.any(mask_bad): break
                count = np.sum(mask_bad)
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.random.standard_cauchy(count)
                f = np.minimum(f, 1.0)
            
            # B. Mutation: current-to-pbest/1
            # Sort population to find p-best
            sorted_indices = np.argsort(fitness)
            
            # Random p in range [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            p_count = int(max(2, p_val * pop_size))
            top_p_indices = sorted_indices[:p_count]
            
            # Select vectors
            pbest_idx = np.random.choice(top_p_indices, pop_size)
            x_pbest = pop[pbest_idx]
            
            r1_idx = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_idx]
            
            # r2 is chosen from Union(Population, Archive)
            pool_size = pop_size + archive_cnt
            r2_raw_idx = np.random.randint(0, pool_size, pop_size)
            
            x_r2 = np.empty((pop_size, dim))
            
            # Indices < pop_size take from pop, others from archive
            mask_pop = r2_raw_idx < pop_size
            mask_arc = ~mask_pop
            
            x_r2[mask_pop] = pop[r2_raw_idx[mask_pop]]
            if archive_cnt > 0 and np.any(mask_arc):
                arc_indices = r2_raw_idx[mask_arc] - pop_size
                x_r2[mask_arc] = archive[arc_indices]
            
            # Compute Mutant: v = x + F*(xp - x) + F*(xr1 - xr2)
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # C. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            rand_u = np.random.rand(pop_size, dim)
            mask = rand_u < cr[:, None]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # D. Selection and Memory/Archive Update
            trial_fitness = np.zeros(pop_size)
            
            # Arrays to store successful updates for memory
            diff_list = []
            succ_f = []
            succ_cr = []
            
            # Mask for individuals that are better than parent
            mask_replace = np.zeros(pop_size, dtype=bool)
            
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                f_trial = func(trial[i])
                trial_fitness[i] = f_trial
                
                if f_trial < global_best_val:
                    global_best_val = f_trial
                
                if f_trial <= fitness[i]:
                    mask_replace[i] = True
                    diff = fitness[i] - f_trial
                    diff_list.append(diff)
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
            
            # Update Archive with replaced parents
            replaced_idx = np.where(mask_replace)[0]
            if len(replaced_idx) > 0:
                parents = pop[replaced_idx]
                n_parents = len(parents)
                
                if archive_cnt + n_parents <= max_archive_cap:
                    # Append
                    archive[archive_cnt : archive_cnt + n_parents] = parents
                    archive_cnt += n_parents
                else:
                    # Fill space then replace random
                    rem = max_archive_cap - archive_cnt
                    if rem > 0:
                        archive[archive_cnt:] = parents[:rem]
                        parents = parents[rem:]
                        archive_cnt = max_archive_cap
                    
                    if len(parents) > 0:
                        rep_idx = np.random.randint(0, max_archive_cap, len(parents))
                        archive[rep_idx] = parents
            
            # Update Population
            pop[mask_replace] = trial[mask_replace]
            fitness[mask_replace] = trial_fitness[mask_replace]
            
            # Update Memory (Weighted Lehmer Mean)
            if len(diff_list) > 0:
                diffs = np.array(diff_list)
                total_diff = np.sum(diffs)
                if total_diff > 0:
                    w = diffs / total_diff
                    sf = np.array(succ_f)
                    scr = np.array(succ_cr)
                    
                    mean_f = np.sum(w * (sf ** 2)) / np.sum(w * sf)
                    mean_cr = np.sum(w * scr)
                    
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # E. Linear Population Size Reduction
            # Remove the worst individual to shrink population
            worst_idx = np.argmax(fitness)
            
            # Delete from population (using indices)
            # Create a mask to keep all except worst_idx
            keep_mask = np.ones(pop_size, dtype=bool)
            keep_mask[worst_idx] = False
            
            pop = pop[keep_mask]
            fitness = fitness[keep_mask]
            pop_size -= 1
            
            # Shrink Archive capacity logic (implicit)
            # Archive capacity target is arc_rate * current_pop_size
            target_arc_cap = int(arc_rate * pop_size)
            if archive_cnt > target_arc_cap:
                # Truncate archive to match new capacity
                archive_cnt = target_arc_cap
            
            # F. Convergence/Stagnation Check
            # Check if population is extremely close (variance)
            if np.std(fitness) < 1e-9:
                break
            
            # Check for lack of improvement
            current_run_best = np.min(fitness)
            if current_run_best < run_best - 1e-8:
                run_best = current_run_best
                stag_count = 0
            else:
                stag_count += 1
            
            # If stagnated for significant generations, restart
            if stag_count > max(25, dim):
                break
                
    return global_best_val
