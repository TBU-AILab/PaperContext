#The output value is: 1.1966382488450478
#
#The algorithm code is:
#The following Python code implements a high-performance **L-SHADE-R (L-SHADE with Restarts)** algorithm.
#
#### Key Improvements
#1.  **Hybrid Initialization with Soft Restarts**: Instead of discarding all knowledge upon restarting, the algorithm keeps the global best solution and generates a portion of the new population (Guided Sampling) around it. This intensifies exploitation in promising regions. The rest of the population is initialized using **Latin Hypercube Sampling (LHS)** to maintain global exploration.
#2.  **Time-Consistent LPSR**: The Linear Population Size Reduction (LPSR) is calculated based on the *global* elapsed time. This means that late-stage restarts automatically begin with smaller populations, turning the algorithm into a fast local search (micro-restart) as the deadline approaches.
#3.  **Adaptive Parameters (SHADE)**: Uses historical memory ($M_{CR}, M_F$) to adapt crossover and mutation rates, with a weighted Lehmer mean update to favor parameters that generate larger fitness improvements.
#4.  **Dynamic Guided Sampling**: The spread ($\sigma$) of the guided samples around the global best decreases as time progresses, effectively refining the search focus from basin-exploration to peak-exploitation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE with Restarts, Hybrid Initialization,
    and Global Time-based Population Reduction.
    """
    # 1. Setup Timing
    start_time = datetime.now()
    # Buffer to ensure strict adherence to the time limit
    limit = timedelta(seconds=max_time - 0.05)

    def is_time_up():
        return (datetime.now() - start_time) >= limit

    def get_progress():
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    # 2. Pre-process Bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # 3. Algorithm Hyperparameters
    # Initial population: start large for exploration
    initial_pop_size = int(np.clip(18 * dim, 50, 250))
    min_pop_size = 5
    
    # SHADE Memory parameters
    H = 6               # Memory size
    arc_rate = 2.3      # Archive size relative to population
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_x = None

    # 4. Main Restart Loop
    # Continues to restart the optimization process until time runs out
    while not is_time_up():
        
        # --- A. Initialization for Current Restart ---
        
        # Calculate population size based on GLOBAL progress.
        # Late restarts -> Small population (Exploitation)
        progress = get_progress()
        current_init_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
        pop_size = max(min_pop_size, current_init_size)
        
        # Allocate Population
        pop = np.zeros((pop_size, dim))
        fitness = np.full(pop_size, float('inf'))
        
        # Hybrid Initialization Strategy:
        # 1. Elitism: Keep global best at index 0
        idx_start = 0
        if global_best_x is not None:
            pop[0] = global_best_x
            fitness[0] = global_best_val
            idx_start = 1
            
        # 2. Guided Gaussian Sampling (Exploitation)
        # Sample around global_best. Sigma shrinks as time passes.
        n_guided = 0
        if global_best_x is not None and pop_size > 5:
            n_guided = int(0.3 * pop_size)
            if n_guided > 0:
                # Sigma decays from 5% to ~0% of domain width
                sigma_scale = 0.05 * (1.0 - progress) 
                sigma = sigma_scale * diff_b 
                
                guided_samples = global_best_x + np.random.normal(0, 1, (n_guided, dim)) * sigma
                guided_samples = np.clip(guided_samples, min_b, max_b)
                
                end_idx = min(pop_size, idx_start + n_guided)
                pop[idx_start:end_idx] = guided_samples[:end_idx-idx_start]
                idx_start = end_idx
            
        # 3. Latin Hypercube Sampling (Global Exploration)
        # Fill the rest of the population
        n_lhs = pop_size - idx_start
        if n_lhs > 0:
            if n_lhs > 1:
                # Stratified sampling
                perc = np.tile(np.arange(n_lhs)[:, None], (1, dim)) / n_lhs
                for d in range(dim):
                    np.random.shuffle(perc[:, d])
                # Jitter
                samples = perc + np.random.rand(n_lhs, dim) / n_lhs
                pop[idx_start:] = min_b + samples * diff_b
            else:
                pop[idx_start:] = min_b + np.random.rand(n_lhs, dim) * diff_b
        
        # Evaluate Initial Population
        # Skip index 0 if it's the elite preserved from previous run
        start_eval = 1 if (global_best_x is not None) else 0
        
        for i in range(start_eval, pop_size):
            # Check time periodically to reduce overhead
            if (i % 10 == 0) and is_time_up():
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_x = pop[i].copy()
                
        # Initialize SHADE Memory
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        max_arc_size = int(pop_size * arc_rate)
        archive = np.empty((int(initial_pop_size * arc_rate), dim))
        archive_cnt = 0
        
        # Stagnation Detection
        run_best = np.min(fitness)
        stag_count = 0
        stag_limit = max(25, dim) # Tolerance
        
        # --- B. Evolutionary Loop ---
        while not is_time_up():
            
            # 1. Linear Population Size Reduction (LPSR)
            # Resize target based on global time
            prog = get_progress()
            target_size = int(round((min_pop_size - initial_pop_size) * prog + initial_pop_size))
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
                    # Randomly reduce archive
                    keep_idx = np.random.choice(archive_cnt, max_arc_size, replace=False)
                    archive[:max_arc_size] = archive[keep_idx]
                    archive_cnt = max_arc_size
            
            # 2. Parameter Generation
            # Dynamic 'p' for current-to-pbest (decays 0.18 -> 0.02)
            p_val = 0.18 * (1.0 - prog) + 0.02
            p_val = max(p_val, 2.0 / pop_size)
            
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
                # Check time batch-wise
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
                
                if archive_cnt < max_arc_size:
                    take = min(n_improved, max_arc_size - archive_cnt)
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
                if total_w > 1e-15:
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
