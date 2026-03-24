#The following Python code implements an enhanced **jSO (L-SHADE variant) with Epoch-based Restarts**.
#
#**Key Improvements:**
#1.  **jSO-inspired Parameter Adaptation**: Implements the "High F" heuristic (forcing $F \ge 0.7$ in the first 60% of the search) and specific rules for $p$-best reduction ($0.25 \to 0.05$). This maintains search momentum better than standard L-SHADE.
#2.  **Epoch-based Time Management**: The algorithm divides the `max_time` into flexible epochs. If a run converges early, it restarts and rescales the **Linear Population Size Reduction (LPSR)** schedule to fit the *remaining* time perfectly.
#3.  **Strict Restart Criteria**: Restarts are triggered by stagnation (60 gens without improvement) or complete convergence (minimal population size with zero variance), preventing wasted evaluations on local optima.
#4.  **Vectorized Operations**: Fully vectorized mutation, crossover, and memory updates to maximize throughput.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using jSO (L-SHADE variant) with Epoch-based Restarts.
    
    The algorithm manages 'epochs' of evolution. Each epoch plans its population 
    reduction schedule based on the remaining time. If an epoch converges or 
    stagnates, a new epoch begins (Elitism preserves the global best).
    """

    # --- Configuration ---
    # Population Size Heuristics (jSO suggests larger initial pop than SHADE)
    init_pop_size = int(np.clip(30 * dim, 50, 400))
    min_pop_size = 4
    
    # L-SHADE / jSO Parameters
    H = 5                   # Historical memory size
    arc_rate = 2.0          # Archive size relative to population
    p_max = 0.25            # Initial p-best percentage
    p_min = 0.05            # Final p-best percentage
    
    # Time Management
    start_run = datetime.now()
    end_run = start_run + timedelta(seconds=max_time)
    
    # Pre-process bounds
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
        
        # Track start of this epoch to calculate progress relative to remaining time
        epoch_start = datetime.now()
        
        # --- A. Initialization for New Epoch ---
        pop_size = init_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous epochs
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
                
        # Initialize Memory (H)
        mem_cr = np.full(H, 0.8) # Start CR closer to 1.0
        mem_f = np.full(H, 0.5)  # Start F at 0.5
        k_mem = 0
        archive = []
        
        # Stagnation monitoring
        stag_count = 0
        last_best_val = global_best_val
        
        # --- B. Evolutionary Cycle (Epoch) ---
        while True:
            if check_timeout(): return global_best_val
            
            # 1. Calculate Progress (0.0 to 1.0) based on remaining time
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
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices]
                fitness = fitness[sort_indices]
                
                # Truncate to target size (remove worst)
                pop_size = target_size
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive
                target_arc = int(pop_size * arc_rate)
                if len(archive) > target_arc:
                    random.shuffle(archive)
                    archive = archive[:target_arc]

            # 3. Parameter Adaptation
            # p decreases linearly from p_max to p_min
            p_curr = p_max - (p_max - p_min) * progress
            
            # Assign memory index randomly
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(m_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(m_f, 0.1)
            # Retry if F <= 0 until positive
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            while True:
                bad_mask = f <= 0
                if not np.any(bad_mask): break
                f[bad_mask] = m_f[bad_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_mask)) - 0.5))
            
            f = np.where(f > 1.0, 1.0, f)
            
            # jSO Heuristic: Force high F in the first 60% of search to promote exploration
            if progress < 0.6:
                f = np.where(f < 0.7, 0.7, f)
            
            # 4. Mutation: current-to-pbest/1
            sorted_indices = np.argsort(fitness)
            p_count = max(2, int(pop_size * p_curr))
            pbest_pool = sorted_indices[:p_count]
            
            # Select p-best
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1 (distinct from i)
            shift = np.random.randint(1, pop_size, pop_size)
            r1_idx = (np.arange(pop_size) + shift) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 (distinct from i and r1) from Union(Pop, Archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            
            # Rejection sampling for r2
            r2_idx = np.random.randint(0, len(union_pop), pop_size)
            curr_idx = np.arange(pop_size)
            
            # r2 cannot be i (if r2 is in pop) and r2 cannot be r1
            bad_mask = (r2_idx < pop_size) & ((r2_idx == curr_idx) | (r2_idx == r1_idx))
            
            retry = 0
            while np.any(bad_mask) and retry < 5:
                n_bad = np.sum(bad_mask)
                r2_idx[bad_mask] = np.random.randint(0, len(union_pop), n_bad)
                bad_mask = (r2_idx < pop_size) & ((r2_idx == curr_idx) | (r2_idx == r1_idx))
                retry += 1
                
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
            
            # 6. Bound Handling
            trial = np.clip(trial, min_b, max_b)
            
            # 7. Selection & Evaluation
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_fitness = np.zeros(pop_size)
            
            for i in range(pop_size):
                if check_timeout(): return global_best_val
                
                val = func(trial[i])
                
                if val <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fitness[i] = val
                    
                    if val < fitness[i]:
                        succ_mask[i] = True
                        diff_fitness[i] = fitness[i] - val
                        archive.append(pop[i].copy())
                        
                    if val < global_best_val:
                        global_best_val = val
                        global_best_sol = trial[i].copy()
            
            pop = new_pop
            fitness = new_fitness
            
            # 8. Memory Update (Weighted)
            if np.any(succ_mask):
                w_diff = diff_fitness[succ_mask]
                w_cr = cr[succ_mask]
                w_f = f[succ_mask]
                
                total_diff = np.sum(w_diff)
                weights = w_diff / total_diff if total_diff > 0 else np.ones(len(w_diff))/len(w_diff)
                
                # Update CR Memory (Weighted Mean)
                mem_cr[k_mem] = np.sum(weights * w_cr)
                
                # Update F Memory (Weighted Lehmer Mean)
                denom = np.sum(weights * w_f)
                mean_f = np.sum(weights * (w_f**2)) / denom if denom > 0 else 0.5
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                
                k_mem = (k_mem + 1) % H
                
            # 9. Restart Triggers
            if global_best_val < last_best_val:
                last_best_val = global_best_val
                stag_count = 0
            else:
                stag_count += 1
            
            # Trigger A: Stagnation (no improvement for 60 gens)
            if stag_count >= 60:
                break
                
            # Trigger B: Convergence (min pop size and low variance)
            if pop_size <= min_pop_size:
                if np.var(fitness) < 1e-8:
                    break

    return global_best_val
