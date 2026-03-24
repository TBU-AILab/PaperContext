#The following Python code implements an optimized **L-SHADE-RSP** (L-SHADE with Rescaled Restarts) algorithm.
#
#**Key Improvements:**
#1.  **Fully Vectorized Operations**: The entire evolutionary cycle (parameter generation, mutation, crossover, selection, and memory updates) is vectorized using NumPy. This minimizes interpreter overhead and maximizes function evaluations per second.
#2.  **Efficient Memory Management**: The external archive uses a pre-allocated NumPy array (`archive_storage`) to avoid costly dynamic list resizing. Memory operations are handled via pointer arithmetic.
#3.  **Rescaled Restarts**: The algorithm treats the remaining time as a dynamic epoch. When a restart occurs (due to stagnation or convergence), the **Linear Population Size Reduction (LPSR)** schedule is recalculated to fit the exact remaining time, ensuring a full exploration-exploitation cycle is always performed.
#4.  **Robust Parameter Generation**: Uses a vectorized retry mechanism for Cauchy-distributed mutation factors ($F$) to handle constraint violations statistically correctly, rather than simple clamping.
#5.  **Optimized Selection**: Selection and archive updates are batched, significantly reducing the overhead compared to iterating through the population.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Optimized L-SHADE with Rescaled Restarts (L-SHADE-RSP).
    
    The algorithm features:
    - Linear Population Size Reduction (LPSR) scaled to remaining time (Rescaled Restarts).
    - Success-History based Adaptive Differential Evolution (SHADE).
    - Fully vectorized evolutionary operators and efficient memory management.
    """

    # --- Configuration ---
    # Population Size Heuristic
    # Scaled by dimension but clipped to ensure performance on various problem sizes.
    init_pop_size = int(np.clip(25 * dim, 50, 350))
    min_pop_size = 5
    
    # L-SHADE Parameters
    H = 5                   # Historical Memory size
    arc_rate = 2.0          # Archive capacity relative to population size
    
    # Restart Triggers
    stag_limit = 50         # Generations without improvement before restart
    
    # Time Management
    start_run = datetime.now()
    end_run = start_run + timedelta(seconds=max_time)
    
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = None
    
    def check_timeout():
        return datetime.now() >= end_run

    # --- Main Optimization Loop (Restarts) ---
    while not check_timeout():
        
        # Mark start of new epoch for LPSR scheduling
        epoch_start = datetime.now()
        
        # --- Initialization ---
        pop_size = init_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous epochs
        start_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_idx = 1
            
        # Initial Evaluation
        for i in range(start_idx, pop_size):
            if check_timeout(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
                
        # Initialize L-SHADE Memory
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive (Pre-allocated for speed)
        # Capacity scales with pop_size, we allocate max needed
        archive_cap_init = int(init_pop_size * arc_rate)
        archive_storage = np.empty((archive_cap_init, dim))
        arc_count = 0
        
        # Stagnation Counter
        stag_count = 0
        last_best_val = global_best_val
        
        # --- Epoch Loop ---
        while True:
            if check_timeout(): return global_best_val
            
            # 1. Time Progress & Linear Population Size Reduction (LPSR)
            now = datetime.now()
            total_duration = (end_run - epoch_start).total_seconds()
            elapsed = (now - epoch_start).total_seconds()
            
            # Calculate progress (0.0 -> 1.0) relative to THIS epoch's available time
            if total_duration <= 1e-9:
                progress = 1.0
            else:
                progress = elapsed / total_duration
                if progress > 1.0: progress = 1.0
                
            # Calculate Target Population Size
            target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            # Reduce Population
            if pop_size > target_size:
                # Sort by fitness (best first)
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx]
                fitness = fitness[sort_idx]
                
                # Truncate
                pop_size = target_size
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive Limit
                # (We don't reallocate, just strictly enforce the limit)
                curr_arc_cap = int(pop_size * arc_rate)
                if arc_count > curr_arc_cap:
                    arc_count = curr_arc_cap # Effectively truncate excess
            
            curr_arc_cap = int(pop_size * arc_rate)

            # 2. Adaptive Parameter Generation
            # Greediness 'p' reduces from 0.2 to 0.05
            p_val = 0.2 - (0.15 * progress)
            p_val = max(0.05, p_val)
            
            # Select memory slots
            r_idx = np.random.randint(0, H, pop_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Vectorized Retry for invalid F (<= 0)
            bad_f = f <= 0
            while np.any(bad_f):
                n_bad = np.sum(bad_f)
                f[bad_f] = mu_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
                bad_f = f <= 0
            
            f = np.clip(f, 0.0, 1.0) # Clip upper bound
            
            # 3. Mutation: current-to-pbest/1
            # Sort for p-best selection
            sorted_idx = np.argsort(fitness)
            p_count = max(2, int(pop_size * p_val))
            pbest_indices = sorted_idx[:p_count]
            
            # Select p-best
            pbest_idx = np.random.choice(pbest_indices, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1 (distinct from i)
            shift = np.random.randint(1, pop_size, pop_size)
            r1_idx = (np.arange(pop_size) + shift) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 (distinct from i, r1) from Union(Pop, Archive)
            # Create virtual union view
            if arc_count > 0:
                union_pop = np.vstack((pop, archive_storage[:arc_count]))
            else:
                union_pop = pop
            
            n_union = len(union_pop)
            r2_idx = np.random.randint(0, n_union, pop_size)
            
            # Vectorized Collision Handling for r2
            # r2 must not be i (if in pop) and must not be r1
            curr_idx = np.arange(pop_size)
            bad_r2 = (r2_idx == curr_idx) | (r2_idx == r1_idx)
            
            retry_limit = 5
            while np.any(bad_r2) and retry_limit > 0:
                n_bad = np.sum(bad_r2)
                r2_idx[bad_r2] = np.random.randint(0, n_union, n_bad)
                bad_r2 = (r2_idx == curr_idx) | (r2_idx == r1_idx)
                retry_limit -= 1
            
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant Vector
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            rand_cr = np.random.rand(pop_size, dim)
            cross_mask = rand_cr < cr[:, np.newaxis]
            # Ensure at least one dimension
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Evaluation (Loop required by interface)
            trial_fitness = np.empty(pop_size)
            for i in range(pop_size):
                if check_timeout(): return global_best_val
                trial_fitness[i] = func(trial[i])
            
            # 6. Selection and Archive Update (Vectorized)
            # Identify better solutions
            mask_better = trial_fitness < fitness
            mask_better_eq = trial_fitness <= fitness
            
            # Update Global Best
            min_trial_idx = np.argmin(trial_fitness)
            if trial_fitness[min_trial_idx] < global_best_val:
                global_best_val = trial_fitness[min_trial_idx]
                global_best_sol = trial[min_trial_idx].copy()
                stag_count = 0
            else:
                stag_count += 1
            
            # Update Archive with replaced individuals
            # Only strictly better solutions push parent to archive
            if np.any(mask_better):
                replaced_pop = pop[mask_better]
                n_replaced = len(replaced_pop)
                
                # Add to archive (Random replacement if full)
                if arc_count + n_replaced <= curr_arc_cap:
                    archive_storage[arc_count:arc_count+n_replaced] = replaced_pop
                    arc_count += n_replaced
                else:
                    # Fill remaining space
                    space = curr_arc_cap - arc_count
                    if space > 0:
                        archive_storage[arc_count:curr_arc_cap] = replaced_pop[:space]
                        remaining_to_add = replaced_pop[space:]
                        arc_count = curr_arc_cap
                    else:
                        remaining_to_add = replaced_pop
                    
                    # Randomly replace existing members
                    if len(remaining_to_add) > 0:
                        n_rem = len(remaining_to_add)
                        rnd_pos = np.random.randint(0, curr_arc_cap, n_rem)
                        archive_storage[rnd_pos] = remaining_to_add
            
            # Update Population
            pop[mask_better_eq] = trial[mask_better_eq]
            
            # Calculate improvements for memory update
            diff_vals = np.abs(fitness[mask_better] - trial_fitness[mask_better])
            fitness[mask_better_eq] = trial_fitness[mask_better_eq]
            
            # 7. Update Historical Memory
            if np.any(mask_better):
                succ_cr = cr[mask_better]
                succ_f = f[mask_better]
                
                # Weighted Mean based on improvement
                total_diff = np.sum(diff_vals)
                if total_diff > 0:
                    weights = diff_vals / total_diff
                    
                    # CR: Weighted Arithmetic Mean
                    m_cr = np.sum(weights * succ_cr)
                    mem_cr[k_mem] = np.clip(m_cr, 0, 1)
                    
                    # F: Weighted Lehmer Mean
                    m_f_num = np.sum(weights * (succ_f**2))
                    m_f_den = np.sum(weights * succ_f)
                    m_f = m_f_num / m_f_den if m_f_den > 1e-10 else 0.5
                    mem_f[k_mem] = np.clip(m_f, 0, 1)
                    
                    k_mem = (k_mem + 1) % H
            
            # 8. Restart Checks
            if stag_count >= stag_limit:
                break
            
            # Convergence check (small population and low variance)
            if pop_size <= min_pop_size and np.std(fitness) < 1e-8:
                break
                
    return global_best_val
