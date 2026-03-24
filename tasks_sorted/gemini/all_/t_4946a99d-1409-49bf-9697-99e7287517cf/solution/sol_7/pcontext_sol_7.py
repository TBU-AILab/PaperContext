#The following Python code implements an improved **L-SHADE-R (Linear Population Size Reduction Success-History Adaptive Differential Evolution with Restarts)** algorithm, enhanced with a **Hybrid Initialization** strategy.
#
##### Key Improvements
#1.  **Hybrid Restart Initialization**: When the algorithm restarts (due to stagnation or convergence), the new population is formed using a mix of:
#    *   **LHS (Latin Hypercube Sampling)**: 80% of individuals, to maintain global search capability and diversity.
#    *   **Guided Gaussian Sampling**: 20% of individuals are sampled around the `global_best` found so far. This allows the algorithm to rapidly refine the solution in a promising basin of attraction without losing the ability to escape local optima.
#2.  **Time-Based LPSR (Linear Population Size Reduction)**: The population size for *each* restart is calculated based on the *remaining time*. Late-stage restarts begin with a smaller population, effectively turning the algorithm into a fast local search as the deadline approaches.
#3.  **Adaptive Parameter Control (SHADE)**: Uses historical memory to adapt crossover ($CR$) and mutation ($F$) rates, with a weighted Lehmer mean update to favor parameters that produce larger fitness improvements.
#4.  **Optimized Time Management**: Checks the time limit in batches (during function evaluations) to minimize system call overhead while ensuring strict adherence to `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE with Restarts, Hybrid Initialization, 
    and Time-based Population Reduction.
    """
    # 1. Setup Timing
    start_time = datetime.now()
    # Use a small buffer to ensure strict adherence to the time limit
    limit = timedelta(seconds=max_time - 0.05)

    def is_time_up():
        return (datetime.now() - start_time) >= limit

    # 2. Pre-process Bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # 3. Algorithm Hyperparameters
    # Initial population: start slightly larger to improve basin finding
    initial_pop_size = int(np.clip(25 * dim, 50, 350))
    min_pop_size = 5
    
    # SHADE Constants
    H = 6               # Memory size for parameter adaptation
    arc_rate = 2.3      # Archive size relative to population
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_x = None

    # 4. Main Restart Loop
    # Continues to restart the optimization process until time runs out
    while not is_time_up():
        
        # --- A. Initialization for Current Restart ---
        
        # Calculate progress (0.0 to 1.0)
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        progress = min(elapsed / max_time, 1.0)
        
        # Determine initial population size for this restart based on remaining time.
        # As time runs out, restarts become smaller and faster (exploitation).
        current_init_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
        pop_size = max(min_pop_size, current_init_size)
        
        # Allocate Population
        pop = np.zeros((pop_size, dim))
        
        # Hybrid Initialization Strategy:
        # Mix of Global Search (LHS) and Local Refinement (Guided Gaussian)
        n_guided = 0
        if global_best_x is not None and pop_size > 5:
            # Use 20% of population to explore around the best found so far
            n_guided = max(1, int(0.2 * pop_size))
            
        n_lhs = pop_size - n_guided
        
        # 1. Latin Hypercube Sampling (Global Search)
        if n_lhs > 0:
            if n_lhs > 1:
                # Generate stratified samples
                perc = np.tile(np.arange(n_lhs)[:, None], (1, dim)) / n_lhs
                for d in range(dim):
                    np.random.shuffle(perc[:, d])
                # Add jitter
                samples = perc + np.random.rand(n_lhs, dim) / n_lhs
                pop[:n_lhs] = min_b + samples * diff_b
            else:
                pop[:n_lhs] = min_b + np.random.rand(n_lhs, dim) * diff_b
        
        # 2. Guided Gaussian Sampling (Local Refinement)
        if n_guided > 0:
            # Sample around global_best with sigma = 5% of domain width
            sigma = 0.05 * diff_b 
            guided_samples = global_best_x + np.random.normal(0, 1, (n_guided, dim)) * sigma
            pop[n_lhs:] = np.clip(guided_samples, min_b, max_b)
            
        # Evaluate Initial Population
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            # Check time periodically to reduce overhead
            if (i % 20 == 0) and is_time_up():
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_x = pop[i].copy()
                
        # Initialize SHADE Memory (M_CR, M_F)
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        # We pre-allocate max capacity but manage actual size dynamically
        max_arc_size = int(pop_size * arc_rate)
        archive = np.empty((int(initial_pop_size * arc_rate), dim))
        archive_cnt = 0
        
        # Stagnation Detection Variables
        run_best = np.min(fitness)
        stag_count = 0
        stag_limit = max(25, dim) # Restart if no improvement for X gens
        
        # --- B. Evolutionary Loop ---
        while not is_time_up():
            
            # 1. Linear Population Size Reduction (LPSR)
            # Linearly decrease population size based on time progress
            elapsed_loop = (datetime.now() - start_time).total_seconds()
            prog_loop = min(elapsed_loop / max_time, 1.0)
            
            target_size = int(round((min_pop_size - initial_pop_size) * prog_loop + initial_pop_size))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Remove worst individuals
                sort_idx = np.argsort(fitness)
                n_keep = target_size
                
                pop = pop[sort_idx[:n_keep]]
                fitness = fitness[sort_idx[:n_keep]]
                pop_size = n_keep
                
                # Resize Archive capacity
                max_arc_size = int(pop_size * arc_rate)
                if archive_cnt > max_arc_size:
                    # Randomly reduce archive to fit
                    keep_idx = np.random.choice(archive_cnt, max_arc_size, replace=False)
                    archive[:max_arc_size] = archive[keep_idx]
                    archive_cnt = max_arc_size
            
            # 2. Parameter Generation
            # Update 'p' for current-to-pbest (decays from 0.2 to 0.05)
            p_val = 0.2 * (1.0 - prog_loop) + 0.05
            p_val = np.clip(p_val, 0.05, 0.2)
            
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # CR ~ Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(M_F, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Retry if F <= 0 (Vectorized)
            while True:
                mask_bad = f <= 0
                if not np.any(mask_bad): break
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.random.standard_cauchy(np.sum(mask_bad))
            f = np.minimum(f, 1.0)
            
            # 3. Mutation: current-to-pbest/1
            sorted_idx = np.argsort(fitness)
            n_pbest = max(2, int(p_val * pop_size))
            pbest_pool = sorted_idx[:n_pbest]
            
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # r1: Random from population
            r1_idx = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_idx]
            
            # r2: Random from Union(Population, Archive)
            n_union = pop_size + archive_cnt
            r2_idx = np.random.randint(0, n_union, pop_size)
            
            x_r2 = np.empty((pop_size, dim))
            mask_pop = r2_idx < pop_size
            x_r2[mask_pop] = pop[r2_idx[mask_pop]]
            
            if archive_cnt > 0:
                mask_arc = ~mask_pop
                # Access archive (shifted index)
                x_r2[mask_arc] = archive[r2_idx[mask_arc] - pop_size]
            else:
                x_r2[~mask_pop] = pop[r2_idx[~mask_pop] % pop_size]
            
            # Compute Mutant Vector
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            rand_u = np.random.rand(pop_size, dim)
            mask_cross = rand_u < cr[:, None]
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # 5. Selection
            trial_fitness = np.empty(pop_size)
            improved_mask = np.zeros(pop_size, dtype=bool)
            diffs = []
            succ_f = []
            succ_cr = []
            
            for i in range(pop_size):
                # Check time limit safely (every 10 evaluations)
                if (i % 10 == 0) and is_time_up():
                    return global_best_val
                
                val = func(trial[i])
                trial_fitness[i] = val
                
                if val <= fitness[i]:
                    improved_mask[i] = True
                    diffs.append(fitness[i] - val)
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    
                    if val < global_best_val:
                        global_best_val = val
                        global_best_x = trial[i].copy()
                        
            # 6. Archive Update
            n_improved = np.sum(improved_mask)
            if n_improved > 0:
                parents = pop[improved_mask]
                
                # Add replaced parents to archive
                if archive_cnt < max_arc_size:
                    space = max_arc_size - archive_cnt
                    take = min(n_improved, space)
                    archive[archive_cnt : archive_cnt + take] = parents[:take]
                    archive_cnt += take
                    
                    if n_improved > take:
                        # Replace random existing members
                        rem = n_improved - take
                        idx = np.random.randint(0, max_arc_size, rem)
                        archive[idx] = parents[take:]
                else:
                    idx = np.random.randint(0, max_arc_size, n_improved)
                    archive[idx] = parents
            
            # Update Population
            pop[improved_mask] = trial[improved_mask]
            fitness[improved_mask] = trial_fitness[improved_mask]
            
            # 7. Memory Update (Weighted Lehmer Mean)
            if len(diffs) > 0:
                w = np.array(diffs)
                total_w = np.sum(w)
                # Avoid division by zero
                if total_w > 1e-12:
                    w = w / total_w
                    sf = np.array(succ_f)
                    scr = np.array(succ_cr)
                    
                    mean_f = np.sum(w * (sf**2)) / np.sum(w * sf)
                    mean_cr = np.sum(w * scr)
                    
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # 8. Stagnation / Convergence Check
            cur_best = np.min(fitness)
            if cur_best < run_best - 1e-8:
                run_best = cur_best
                stag_count = 0
            else:
                stag_count += 1
                
            # Restart if stagnated or converged
            if stag_count > stag_limit or np.std(fitness) < 1e-9:
                break
                
    return global_best_val
