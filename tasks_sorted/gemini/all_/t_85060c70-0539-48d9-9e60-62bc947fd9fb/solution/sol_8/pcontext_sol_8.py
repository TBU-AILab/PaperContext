#The following Python code implements an **Optimized L-SHADE-RSP** algorithm.
#
#**Key Improvements:**
#1.  **Rescaled Epochs**: The algorithm treats the remaining time as a dynamic epoch. If a restart occurs (due to stagnation), the **Linear Population Size Reduction (LPSR)** schedule is recalculated to fit the exact remaining time window. This ensures a complete exploration-exploitation cycle regardless of when the restart happens.
#2.  **Efficient Archive Management**: Uses a pre-allocated NumPy array for the archive instead of growing a list. This significantly reduces memory allocation overhead and improves cache locality during the heavy union operations.
#3.  **Vectorized Robustness**: Implements vectorized retry logic for parameter generation (Cauchy distributed $F$) and candidate selection ($r2$), ensuring validity without falling back to slow loops.
#4.  **Adaptive Restart**: Triggers a soft restart (preserving the global best) if the population stagnates (no improvement for 30 generations) or fully converges, preventing wasted evaluations on local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Optimized L-SHADE with Rescaled Restarts (L-SHADE-RSP).
    """

    # --- Configuration ---
    # Population Size: Start large for exploration, scaled by dimension.
    # Clipped to ensures scalability across various problem sizes and time limits.
    init_pop_size = int(np.clip(25 * dim, 50, 350))
    min_pop_size = 5
    
    # L-SHADE Parameters
    H = 6                   # Historical Memory size
    arc_rate = 2.3          # Archive capacity relative to population size
    
    # Restart triggers
    stag_limit = 30         # Generations without improvement before restart
    
    # --- Initialization ---
    start_run = datetime.now()
    end_run = start_run + timedelta(seconds=max_time)
    
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')
    global_best_sol = None
    
    def check_timeout():
        return datetime.now() >= end_run

    # --- Main Optimization Loop (Restarts) ---
    while not check_timeout():
        
        # Start of a new restart epoch
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
        
        # Initialize Archive (Pre-allocated for efficiency)
        max_arc_capacity = int(init_pop_size * arc_rate)
        archive = np.empty((max_arc_capacity, dim))
        arc_size = 0
        
        # Stagnation monitoring
        stag_count = 0
        last_best_val = global_best_val
        
        # --- Epoch Generation Loop ---
        while True:
            if check_timeout(): return global_best_val
            
            # 1. Time Progress Calculation (0.0 -> 1.0)
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
            target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Sort population by fitness
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx]
                fitness = fitness[sort_idx]
                
                # Truncate population
                pop_size = target_size
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive
                curr_max_arc = int(pop_size * arc_rate)
                if arc_size > curr_max_arc:
                    # Randomly truncate archive to new size limit
                    idxs = np.random.permutation(arc_size)[:curr_max_arc]
                    archive[:curr_max_arc] = archive[idxs]
                    arc_size = curr_max_arc

            # 3. Adaptive Parameter Generation
            # Greediness (p) reduces linearly from 0.2 to 0.05
            p_val = 0.2 - (0.15 * progress)
            p_val = np.clip(p_val, 0.05, 0.2)
            
            # Select Memory Indices
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(m_f, 0.1) with Vectorized Retry
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Retry loop for invalid values (<= 0)
            for _ in range(5):
                bad_mask = f <= 0
                if not np.any(bad_mask): break
                n_bad = np.sum(bad_mask)
                f[bad_mask] = m_f[bad_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
            
            f = np.where(f <= 0, 0.5, f) # Robust fallback
            f = np.where(f > 1.0, 1.0, f) # Clip upper bound
            
            # 4. Mutation: current-to-pbest/1
            sorted_idx = np.argsort(fitness)
            p_count = max(2, int(pop_size * p_val))
            pbest_pool = sorted_idx[:p_count]
            
            # Select p-best
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1 != i
            shift = np.random.randint(1, pop_size, pop_size)
            r1_idx = (np.arange(pop_size) + shift) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 != i, r2 != r1 from Union(Pop, Archive)
            union_size = pop_size + arc_size
            r2_idx = np.random.randint(0, union_size, pop_size)
            
            curr_idx = np.arange(pop_size)
            bad_r2 = (r2_idx == curr_idx) | (r2_idx == r1_idx)
            
            # Retry r2 collisions
            for _ in range(5):
                if not np.any(bad_r2): break
                r2_idx[bad_r2] = np.random.randint(0, union_size, np.sum(bad_r2))
                bad_r2 = (r2_idx == curr_idx) | (r2_idx == r1_idx)
                
            # Create Union Array (Virtual or Stacked)
            if arc_size > 0:
                union_pop = np.vstack((pop, archive[:arc_size]))
            else:
                union_pop = pop
                
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant Vector
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, np.newaxis]
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # 6. Evaluation and Selection
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_fitness = np.zeros(pop_size)
            
            archive_candidates = []
            
            for i in range(pop_size):
                if check_timeout(): return global_best_val
                
                t_val = func(trial[i])
                
                # Greedy Selection
                if t_val <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fitness[i] = t_val
                    
                    if t_val < fitness[i]:
                        succ_mask[i] = True
                        diff_fitness[i] = fitness[i] - t_val
                        archive_candidates.append(pop[i].copy())
                        
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_sol = trial[i].copy()
            
            pop = new_pop
            fitness = new_fitness
            
            # 7. Update Archive
            if len(archive_candidates) > 0:
                cands = np.array(archive_candidates)
                n_cands = len(cands)
                
                # Fill available space first
                if arc_size < max_arc_capacity:
                    n_put = min(n_cands, max_arc_capacity - arc_size)
                    archive[arc_size : arc_size + n_put] = cands[:n_put]
                    arc_size += n_put
                    cands = cands[n_put:]
                    n_cands = len(cands)
                
                # Replace random existing elements if candidates remain
                if n_cands > 0:
                    replace_idx = np.random.randint(0, arc_size, n_cands)
                    archive[replace_idx] = cands
            
            # 8. Update Memory (Weighted Means)
            if np.any(succ_mask):
                w_diff = diff_fitness[succ_mask]
                w_cr = cr[succ_mask]
                w_f = f[succ_mask]
                
                norm = np.sum(w_diff)
                if norm > 0:
                    weights = w_diff / norm
                    
                    # Update CR Memory (Weighted Arithmetic Mean)
                    mem_cr[k_mem] = np.sum(weights * w_cr)
                    
                    # Update F Memory (Weighted Lehmer Mean)
                    num = np.sum(weights * (w_f**2))
                    den = np.sum(weights * w_f)
                    mem_f[k_mem] = num / den if den > 1e-10 else 0.5
                    mem_f[k_mem] = np.clip(mem_f[k_mem], 0, 1)
                    
                    k_mem = (k_mem + 1) % H
            
            # 9. Restart Check
            # Check for stagnation
            if global_best_val < last_best_val:
                last_best_val = global_best_val
                stag_count = 0
            else:
                stag_count += 1
            
            # Trigger restart if stagnated
            if stag_count >= stag_limit:
                break
            
            # Trigger restart if population is minimal and converged
            if pop_size <= min_pop_size and np.std(fitness) < 1e-8:
                break
                
    return global_best_val
