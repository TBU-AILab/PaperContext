#The following Python code implements an improved **L-SHADE-R (L-SHADE with Restarts)** algorithm.
#
#### Key Improvements
#1.  **LHS Initialization**: Uses vectorized Latin Hypercube Sampling for the initial population to ensure a more uniform coverage of the search space than random sampling.
#2.  **Midpoint Boundary Handling**: Instead of simple clipping (which clumps values at bounds), variables violating bounds are corrected to the midpoint between the parent value and the bound. This preserves population diversity near the edges.
#3.  **Time-Based LPSR**: Linearly reduces population size based on elapsed time, transitioning from exploration to exploitation. Late restarts utilize this to perform fast "micro-searches".
#4.  **Vectorized Operations**: Fully vectorized mutation, crossover, and boundary handling logic for maximum speed.
#5.  **Robust Restarts**: Detects stagnation (no improvement) or convergence (low variance) to restart the search, allowing the algorithm to escape local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE with Restarts, Time-based Population Reduction, 
    and Midpoint Boundary Correction.
    """
    # 1. Setup Timing and Constants
    start_time = datetime.now()
    limit_delta = timedelta(seconds=max_time)
    
    # Helper to calculate global progress (0.0 to 1.0)
    def get_progress():
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    def is_time_up():
        return (datetime.now() - start_time) >= limit_delta

    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    
    # --- Algorithm Configuration ---
    # Population sizing: High dim requires larger pop, but time constraints apply.
    # Scaled by dim but clipped to [30, 300] to ensure responsiveness.
    initial_pop_size = int(np.clip(18 * dim, 30, 300))
    min_pop_size = 4
    
    # Archive parameters
    arc_rate = 2.6
    
    # SHADE Memory parameters
    H = 5 # Memory size
    
    # --- Main Restart Loop ---
    # Continuously restarts the optimization process until time expires
    while not is_time_up():
        
        # 2. Initialization for current run
        pop_size = initial_pop_size
        
        # Latin Hypercube Sampling (LHS) for better initial coverage
        # Vectorized generation of stratified samples
        seg = 1.0 / pop_size
        r_lhs = np.random.rand(pop_size, dim) * seg
        # Random permutations for each dimension
        idx_lhs = np.argsort(np.random.rand(pop_size, dim), axis=0)
        offsets = idx_lhs * seg
        samples_norm = offsets + r_lhs
        pop = min_b + samples_norm * diff_b
        
        # Evaluate Initial Population
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if is_time_up(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val

        # Initialize SHADE Memory (History of successful F and CR)
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize External Archive
        max_arc_size = int(pop_size * arc_rate)
        archive = np.empty((max_arc_size, dim))
        archive_cnt = 0
        
        # Stagnation and Convergence Tracking
        run_best = np.min(fitness)
        stag_count = 0
        # Allow more stagnation tolerance for higher dimensions
        max_stag = max(20, dim) 
        
        # --- Evolutionary Generation Loop ---
        while not is_time_up():
            
            # 3. Linear Population Size Reduction (LPSR)
            # Calculate target population size based on Time Progress
            progress = get_progress()
            target_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
            target_size = max(min_pop_size, target_size)
            
            # Shrink Population if needed
            if pop_size > target_size:
                # Retain only the best individuals
                n_keep = target_size
                sort_idx = np.argsort(fitness)
                keep_idx = sort_idx[:n_keep]
                
                pop = pop[keep_idx]
                fitness = fitness[keep_idx]
                pop_size = n_keep
                
                # Resize Archive capacity proportionately
                max_arc_size = int(pop_size * arc_rate)
                if archive_cnt > max_arc_size:
                    archive_cnt = max_arc_size
            
            # 4. Parameter Generation (JADE/SHADE logic)
            # Pick random memory slots
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # CR ~ Normal(mean, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mean, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Repair F <= 0
            while True:
                bad_mask = f <= 0
                if not np.any(bad_mask): break
                f[bad_mask] = m_f[bad_mask] + 0.1 * np.random.standard_cauchy(np.sum(bad_mask))
            f = np.minimum(f, 1.0)
            
            # 5. Mutation: current-to-pbest/1
            # Sort population to identify p-best
            sorted_idx = np.argsort(fitness)
            
            # Random p in [2/N, 0.2] for each individual
            p_min = 2.0 / pop_size
            p_i = np.random.uniform(p_min, 0.2, pop_size)
            p_counts = (p_i * pop_size).astype(int)
            p_counts = np.maximum(2, p_counts)
            
            # Vectorized selection of pbest indices
            # Select rank index from [0, p_count]
            rand_ranks = (np.random.rand(pop_size) * p_counts).astype(int)
            pbest_indices = sorted_idx[rand_ranks]
            x_pbest = pop[pbest_indices]
            
            # Select r1 (random from pop)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_indices]
            
            # Select r2 (from Union of Pop and Archive)
            n_union = pop_size + archive_cnt
            r2_indices = np.random.randint(0, n_union, pop_size)
            
            x_r2 = np.empty((pop_size, dim))
            mask_in_pop = r2_indices < pop_size
            x_r2[mask_in_pop] = pop[r2_indices[mask_in_pop]]
            
            # Fetch from archive if index >= pop_size
            if archive_cnt > 0:
                mask_in_arc = ~mask_in_pop
                x_r2[mask_in_arc] = archive[r2_indices[mask_in_arc] - pop_size]
            else:
                # Fallback if archive empty (rare)
                x_r2[~mask_in_pop] = pop[r2_indices[~mask_in_pop] % pop_size]
                
            # Compute Mutant Vector
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 6. Midpoint Boundary Handling
            # If mutant is out of bounds, set it to (parent + bound) / 2
            # This preserves diversity better than clipping
            mask_l = mutant < min_b
            if np.any(mask_l):
                # Advanced indexing: get correct min_b for each column
                cols = np.where(mask_l)[1]
                mutant[mask_l] = (pop[mask_l] + min_b[cols]) / 2.0
                
            mask_u = mutant > max_b
            if np.any(mask_u):
                cols = np.where(mask_u)[1]
                mutant[mask_u] = (pop[mask_u] + max_b[cols]) / 2.0
            
            # 7. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            rand_u = np.random.rand(pop_size, dim)
            mask_cross = rand_u < cr[:, None]
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # 8. Selection and Memory Update
            trial_fitness = np.zeros(pop_size)
            mask_improved = np.zeros(pop_size, dtype=bool)
            
            # Arrays to collect successful parameters
            diffs = []
            succ_f = []
            succ_cr = []
            
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                f_val = func(trial[i])
                trial_fitness[i] = f_val
                
                if f_val < global_best_val:
                    global_best_val = f_val
                
                if f_val <= fitness[i]:
                    mask_improved[i] = True
                    diffs.append(fitness[i] - f_val)
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
            
            # Update Archive with replaced parents
            n_imp = np.sum(mask_improved)
            if n_imp > 0:
                parents = pop[mask_improved]
                
                if archive_cnt + n_imp <= max_arc_size:
                    # Append
                    archive[archive_cnt : archive_cnt + n_imp] = parents
                    archive_cnt += n_imp
                else:
                    # Fill remaining space
                    space = max_arc_size - archive_cnt
                    if space > 0:
                        archive[archive_cnt:] = parents[:space]
                        parents = parents[space:]
                        archive_cnt = max_arc_size
                    
                    # Randomly replace existing members
                    if len(parents) > 0:
                        idx = np.random.randint(0, max_arc_size, len(parents))
                        archive[idx] = parents
            
            # Update Population
            pop[mask_improved] = trial[mask_improved]
            fitness[mask_improved] = trial_fitness[mask_improved]
            
            # Update SHADE Memory (Weighted Lehmer Mean)
            if len(diffs) > 0:
                w = np.array(diffs)
                total_w = np.sum(w)
                if total_w > 0:
                    w = w / total_w
                    sf = np.array(succ_f)
                    scr = np.array(succ_cr)
                    
                    mean_f = np.sum(w * (sf ** 2)) / np.sum(w * sf)
                    mean_cr = np.sum(w * scr)
                    
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # 9. Check for Restart (Stagnation or Convergence)
            cur_best = np.min(fitness)
            if cur_best < run_best - 1e-8:
                run_best = cur_best
                stag_count = 0
            else:
                stag_count += 1
            
            std_fit = np.std(fitness)
            
            # Restart if variance is tiny (converged) or stagnation limit reached
            if std_fit < 1e-9 or stag_count > max_stag:
                break
    
    return global_best_val
