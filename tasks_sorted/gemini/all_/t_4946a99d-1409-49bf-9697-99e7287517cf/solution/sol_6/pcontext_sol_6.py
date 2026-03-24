#The following Python code implements an optimized **L-SHADE-R (Linear Population Size Reduction Success-History Adaptive Differential Evolution with Restarts)**.
#
#### Key Improvements
#1.  **LHS Initialization**: Uses **Latin Hypercube Sampling** for the initial population of every restart. This ensures a stratified and uniform coverage of the search space, providing a better starting point than random sampling.
#2.  **Efficient Restart Sizing**: When a restart occurs, the population size is initialized directly based on the **remaining time** (using the LPSR schedule). This avoids wasting function evaluations on a large population that would be immediately culled in late-stage restarts.
#3.  **Adaptive Parameters (jSO-style)**: Implements the `current-to-pbest` mutation with a dynamic $p$ value that decays from `0.2` to `0.05` as time progresses, shifting focus from exploration to exploitation.
#4.  **Optimized Time Management**: Checks the time limit periodically (batch-wise) within loops to minimize system call overhead while ensuring strict adherence to `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE with Restarts, LHS initialization, 
    and optimized population sizing based on remaining time.
    """
    # 1. Setup Timing
    start_time = datetime.now()
    # Subtract a small buffer (0.05s) to ensure strict adherence to the limit
    limit = timedelta(seconds=max_time - 0.05)
    
    def is_time_up():
        return (datetime.now() - start_time) >= limit

    # 2. Pre-process Bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # 3. Algorithm Hyperparameters
    # Initial Population: Start large for exploration, but clip for speed
    initial_pop_size = int(np.clip(20 * dim, 60, 300))
    min_pop_size = 5
    
    # L-SHADE specific constants
    arc_rate = 2.0   # Archive capacity relative to population
    H = 5            # Memory size for parameter adaptation
    
    global_best_val = float('inf')
    
    # 4. Main Restart Loop
    # Continues to restart the optimization process until time runs out
    while not is_time_up():
        
        # --- A. Initialization for Current Restart ---
        
        # Calculate dynamic population size based on global progress.
        # This ensures that late restarts start with a small population (micro-search),
        # saving evaluations for refinement rather than exploration.
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = min(elapsed / max_time, 1.0)
        
        current_pop_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
        current_pop_size = max(min_pop_size, current_pop_size)
        pop_size = current_pop_size
        
        # Latin Hypercube Sampling (LHS)
        # Generates stratified samples [0, 1]
        if pop_size > 1:
            perc = np.tile(np.arange(pop_size)[:, None], (1, dim)) / pop_size
            for d in range(dim):
                np.random.shuffle(perc[:, d])
            rand_offsets = np.random.rand(pop_size, dim) / pop_size
            samples_norm = perc + rand_offsets
        else:
            samples_norm = np.random.rand(pop_size, dim)
            
        pop = min_b + samples_norm * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            # Check time periodically (every 10 evals) to reduce overhead
            if (i % 10 == 0) and is_time_up():
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                
        # Initialize SHADE Memory (History of successful F and CR)
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive = np.empty((int(initial_pop_size * arc_rate), dim))
        archive_cnt = 0
        
        # Stagnation and Convergence Tracking
        run_best = np.min(fitness)
        stag_count = 0
        stag_limit = max(20, dim) # Dynamic tolerance based on dimension
        
        # --- B. Evolutionary Loop ---
        while not is_time_up():
            
            # 1. Linear Population Size Reduction (LPSR)
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = min(elapsed / max_time, 1.0)
            
            target_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Retain only the best individuals
                n_keep = target_size
                sort_idx = np.argsort(fitness)
                
                pop = pop[sort_idx[:n_keep]]
                fitness = fitness[sort_idx[:n_keep]]
                pop_size = n_keep
                
                # Shrink Archive if needed
                curr_cap = int(pop_size * arc_rate)
                if archive_cnt > curr_cap:
                    # Randomly remove elements to fit capacity
                    keep_idx = np.random.choice(archive_cnt, curr_cap, replace=False)
                    archive[:curr_cap] = archive[keep_idx]
                    archive_cnt = curr_cap
            
            # 2. Parameter Generation
            # Pick random memory slots
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Repair F values <= 0 (Vectorized)
            while True:
                mask_bad = f <= 0
                if not np.any(mask_bad): break
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.random.standard_cauchy(np.sum(mask_bad))
            f = np.minimum(f, 1.0)
            
            # 3. Mutation: current-to-pbest/1
            # Dynamic 'p' (top %) decays from 0.2 to 0.05 over time
            p_val = 0.2 * (1.0 - progress) + 0.05
            p_val = np.clip(p_val, 0.05, 0.2)
            
            top_p_count = int(max(2, round(p_val * pop_size)))
            sorted_idx = np.argsort(fitness)
            pbest_inds = np.random.choice(sorted_idx[:top_p_count], pop_size)
            
            x_pbest = pop[pbest_inds]
            
            # r1: random from population
            r1_inds = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_inds]
            
            # r2: random from Union(Population, Archive)
            n_union = pop_size + archive_cnt
            r2_inds = np.random.randint(0, n_union, pop_size)
            
            x_r2 = np.empty((pop_size, dim))
            mask_pop = r2_inds < pop_size
            x_r2[mask_pop] = pop[r2_inds[mask_pop]]
            
            # Fetch from archive if index >= pop_size
            if archive_cnt > 0:
                mask_arc = ~mask_pop
                if np.any(mask_arc):
                    x_r2[mask_arc] = archive[r2_inds[mask_arc] - pop_size]
            else:
                x_r2[~mask_pop] = pop[r2_inds[~mask_pop] % pop_size]
            
            # Compute Mutant: v = x + F(pbest - x) + F(r1 - r2)
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            rand_u = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross = rand_u < cr[:, None]
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # 5. Selection and Updates
            trial_fitness = np.zeros(pop_size)
            succ_mask = np.zeros(pop_size, dtype=bool)
            diffs = []
            
            for i in range(pop_size):
                if (i % 10 == 0) and is_time_up():
                    return global_best_val
                
                f_new = func(trial[i])
                trial_fitness[i] = f_new
                
                if f_new <= fitness[i]:
                    succ_mask[i] = True
                    diffs.append(fitness[i] - f_new)
                    if f_new < global_best_val:
                        global_best_val = f_new
            
            # Update Archive with replaced parents
            n_succ = np.sum(succ_mask)
            if n_succ > 0:
                parents = pop[succ_mask]
                curr_cap = int(pop_size * arc_rate)
                
                if archive_cnt < curr_cap:
                    take = min(n_succ, curr_cap - archive_cnt)
                    archive[archive_cnt : archive_cnt + take] = parents[:take]
                    archive_cnt += take
                    # Replace random existing members if we have more parents than space
                    if n_succ > take:
                        idx = np.random.randint(0, curr_cap, n_succ - take)
                        archive[idx] = parents[take:]
                else:
                    idx = np.random.randint(0, curr_cap, n_succ)
                    archive[idx] = parents
            
            # Update Population
            pop[succ_mask] = trial[succ_mask]
            fitness[succ_mask] = trial_fitness[succ_mask]
            
            # Update Memory (Weighted Lehmer Mean)
            if len(diffs) > 0:
                w = np.array(diffs)
                total_w = np.sum(w)
                if total_w > 1e-15:
                    w = w / total_w
                    sf = f[succ_mask]
                    scr = cr[succ_mask]
                    
                    mean_f = np.sum(w * (sf ** 2)) / np.sum(w * sf)
                    mean_cr = np.sum(w * scr)
                    
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # 6. Check for Restart (Convergence or Stagnation)
            cur_best = np.min(fitness)
            if cur_best < run_best - 1e-9:
                run_best = cur_best
                stag_count = 0
            else:
                stag_count += 1
                
            fit_range = np.max(fitness) - np.min(fitness)
            
            # Restart if stagnated too long or population converged to a point
            if stag_count > stag_limit or fit_range < 1e-9:
                break
                
    return global_best_val
