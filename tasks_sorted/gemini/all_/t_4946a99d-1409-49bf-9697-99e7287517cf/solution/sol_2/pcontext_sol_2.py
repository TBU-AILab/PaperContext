#The following Python code implements a **Success-History based Adaptive Differential Evolution (SHADE) with Restarts**.
#
#### Key Improvements
#1.  **SHADE Algorithm**: This is a state-of-the-art variant of Differential Evolution that adapts parameters $F$ and $CR$ using a historical memory of successful values. This eliminates the need to manually tune these sensitive hyperparameters.
#2.  **External Archive**: It maintains an archive of inferior solutions recently replaced by better ones. This preserves population diversity and prevents premature convergence when using the greedy `current-to-pbest` mutation strategy.
#3.  **Adaptive Restart**: The algorithm monitors population variance and fitness stagnation. If the search converges to a local optimum or stops improving, it triggers a restart to explore new areas of the search space, while retaining the global best solution found so far.
#4.  **Vectorization**: The implementation leverages NumPy for vectorized mutation, crossover, and bound handling to maximize the number of function evaluations within the `max_time` limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    with an External Archive and Restart mechanism.
    """
    # 1. Setup Timing
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    def is_time_up():
        return (datetime.now() - start_time) >= limit

    # 2. Pre-process Bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # 3. Hyperparameters
    # Population size: SHADE typically uses ~18*D, but we clip it to ensure speed
    # within limited time constraints.
    pop_size = int(np.clip(18 * dim, 40, 100))
    
    # Archive size matches population size
    archive_size = pop_size
    
    # Memory size for historical parameter adaptation (H)
    H = 6
    
    # Global best fitness tracker
    global_best_val = float('inf')
    
    # --- Restart Loop ---
    # Allows the algorithm to escape local optima by restarting the population
    # when convergence or stagnation is detected.
    while not is_time_up():
        
        # 4. Initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population safely checking time
        for i in range(pop_size):
            if is_time_up(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Initialize SHADE Memory (start with 0.5)
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0  # Memory index pointer
        
        # Initialize Archive (pre-allocated for performance)
        archive = np.empty((archive_size, dim))
        archive_cnt = 0
        
        # Stagnation detection
        current_run_best = np.min(fitness)
        stag_count = 0
        
        # 5. Evolutionary Loop
        while not is_time_up():
            
            # --- Parameter Generation ---
            # Randomly select a memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(mean, 0.1), clipped to [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(mean, 0.1)
            # F must be > 0. If <= 0, regenerate. Cap at 1.0.
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Vectorized fix for F <= 0
            while True:
                bad_mask = f <= 0
                if not np.any(bad_mask): break
                count = np.sum(bad_mask)
                # Regenerate using original means for the bad indices
                f[bad_mask] = m_f[bad_mask] + 0.1 * np.random.standard_cauchy(count)
                
            f = np.minimum(f, 1.0)
            
            # --- Mutation: current-to-pbest/1 with Archive ---
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            
            # Select p-best (top p% where p is random in [2/pop, 0.2])
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_p_count = int(max(2, p_val * pop_size))
            top_p_indices = sorted_indices[:top_p_count]
            
            # pbest vectors
            pbest_idx = np.random.choice(top_p_indices, pop_size)
            x_pbest = pop[pbest_idx]
            
            # r1 vectors (random from pop)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_idx]
            
            # r2 vectors (random from Union(Pop, Archive))
            if archive_cnt > 0:
                # Efficiently stack pop and filled archive
                union_pop = np.vstack((pop, archive[:archive_cnt]))
            else:
                union_pop = pop
            
            r2_idx = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2_idx]
            
            # Compute mutant vectors: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # Clip to bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, pop_size)
            rand_u = np.random.rand(pop_size, dim)
            mask = rand_u < cr[:, None]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # --- Selection ---
            trial_fitness = np.zeros(pop_size)
            improved_mask = np.zeros(pop_size, dtype=bool)
            diff_vals = np.zeros(pop_size)
            
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                val = func(trial[i])
                trial_fitness[i] = val
                
                if val <= fitness[i]:
                    improved_mask[i] = True
                    diff_vals[i] = fitness[i] - val
                    if val < global_best_val:
                        global_best_val = val
            
            # Indices of successful individuals
            succ_idx = np.where(improved_mask)[0]
            
            if len(succ_idx) > 0:
                # 1. Update Archive
                # Add parents that are about to be replaced to the archive
                parents = pop[succ_idx].copy()
                n_parents = len(parents)
                
                if archive_cnt + n_parents <= archive_size:
                    archive[archive_cnt : archive_cnt + n_parents] = parents
                    archive_cnt += n_parents
                else:
                    # Fill remaining space
                    rem = archive_size - archive_cnt
                    if rem > 0:
                        archive[archive_cnt:] = parents[:rem]
                        parents = parents[rem:]
                        archive_cnt = archive_size
                    
                    # Randomly replace existing archive members
                    replace_idx = np.random.randint(0, archive_size, len(parents))
                    archive[replace_idx] = parents
                
                # 2. Update Population
                pop[succ_idx] = trial[succ_idx]
                fitness[succ_idx] = trial_fitness[succ_idx]
                
                # 3. Update SHADE Memory
                # Calculate weights based on fitness improvement
                diffs = diff_vals[succ_idx]
                total_diff = np.sum(diffs)
                
                if total_diff > 0:
                    weights = diffs / total_diff
                    
                    s_cr = cr[succ_idx]
                    s_f = f[succ_idx]
                    
                    # Weighted Lehmer Mean for F
                    mean_f = np.sum(weights * (s_f ** 2)) / np.sum(weights * s_f)
                    # Weighted Arithmetic Mean for CR
                    mean_cr = np.sum(weights * s_cr)
                    
                    # Update memory slot
                    mem_cr[k_mem] = mean_cr
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # --- Convergence/Restart Check ---
            new_run_best = np.min(fitness)
            if new_run_best < current_run_best - 1e-8:
                current_run_best = new_run_best
                stag_count = 0
            else:
                stag_count += 1
            
            pop_std = np.std(fitness)
            
            # Restart if variance is extremely low (converged) 
            # or no improvement for 45 generations
            if pop_std < 1e-6 or stag_count > 45:
                break
                
    return global_best_val
