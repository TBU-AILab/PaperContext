#The following Python code implements an improved **L-SHADE-RSP** (Linear Population Size Reduction with Success-History based Adaptive Differential Evolution and **Rescaled Restarts**) algorithm.
#
#**Key Improvements:**
#1.  **Epoch-Based Restarts**: Unlike standard restart mechanisms that simply reset the population, this algorithm treats the remaining time as a new "epoch". It rescales the **Linear Population Size Reduction (LPSR)** schedule to fit the remaining time window perfectly. This ensures that every restart cycle gets a complete exploration-to-exploitation transition, rather than being instantly truncated to a small population.
#2.  **Stagnation Detection**: The algorithm monitors the global best value. If it does not improve for a set number of generations (stagnation) or if the population variance becomes negligible (convergence), it triggers a restart to escape the local optimum while preserving the best-found solution (Elitism).
#3.  **Adaptive Parameters (L-SHADE)**: It uses historical memory ($M_{CR}, M_{F}$) to adapt crossover and mutation rates based on successful updates, and linearly reduces the greediness of the mutation strategy ($p$-best) over the epoch.
#4.  **Vectorized Efficiency**: All evolutionary operators are fully vectorized using NumPy, maximizing the number of function evaluations within the `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Rescaled Restarts (L-SHADE-RSP).
    
    This algorithm manages 'epochs' of evolution. If the population converges 
    or stagnates, it restarts with a new population (keeping the global best) 
    and rescales the population reduction schedule to fit the remaining time.
    """
    
    # --- 1. Configuration & Constants ---
    # Population Size Heuristics
    # Start with a moderate size to ensure speed, scale with dimension
    # Clip ensures we don't start too huge on high dims, nor too small on low dims
    init_pop_size = int(np.clip(25 * dim, 50, 300))
    min_pop_size = 5  # Minimum size to support mutation strategies
    
    # L-SHADE Parameters
    H = 5               # Historical memory size
    arc_rate = 2.0      # Archive size relative to population size
    
    # Stagnation limit (generations without improvement before restart)
    stag_limit = 50

    # --- 2. Setup ---
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
    
    # Helper: Check strictly if time is up
    def check_timeout():
        return datetime.now() >= end_run

    # --- 3. Main Optimization Loop (Restarts) ---
    # epoch_start_time tracks the beginning of the current restart cycle
    epoch_start_time = start_run
    
    while not check_timeout():
        
        # --- A. Initialization for new Epoch ---
        pop_size = init_pop_size
        
        # Generate Random Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best solution from previous epochs
        start_eval_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_eval_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_timeout(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
                
        # Initialize L-SHADE Memory and Archive
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        archive = [] # List to store archived solutions
        
        # Stagnation counters for this epoch
        stag_count = 0
        last_best_val = global_best_val
        
        # --- B. Evolutionary Cycle (Epoch) ---
        # Run until population is minimized AND converged, or stagnated
        while True:
            if check_timeout(): return global_best_val
            
            # 1. Calculate Time Progress for LPSR
            # Progress is relative to the time window [epoch_start, end_run]
            now = datetime.now()
            total_epoch_duration = (end_run - epoch_start_time).total_seconds()
            elapsed_epoch = (now - epoch_start_time).total_seconds()
            
            if total_epoch_duration <= 1e-9:
                progress = 1.0
            else:
                progress = elapsed_epoch / total_epoch_duration
                if progress > 1.0: progress = 1.0
            
            # 2. Linear Population Size Reduction (LPSR)
            # Linearly interpolate target size based on remaining time in this epoch
            target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Reduce Population: sort by fitness, keep best
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices]
                fitness = fitness[sort_indices]
                
                # Truncate
                pop_size = target_size
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive
                target_arc_size = int(pop_size * arc_rate)
                if len(archive) > target_arc_size:
                    # Randomly remove excess to maintain diversity
                    import random
                    random.shuffle(archive)
                    archive = archive[:target_arc_size]

            # 3. Adaptive Parameter Generation
            # Greediness parameter 'p' decreases from 0.2 to 0.05
            p_val = 0.2 - (0.15 * progress)
            p_val = np.clip(p_val, 0.05, 0.2)
            
            # Generate CR and F from memory
            r_idx = np.random.randint(0, H, pop_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1), clipped [0, 1], robust check <= 0
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f = np.where(f > 1.0, 1.0, f)
            f = np.where(f <= 0.0, 0.5, f)
            
            # 4. Mutation: current-to-pbest/1
            # Sort for p-best selection
            sorted_indices = np.argsort(fitness)
            p_count = max(2, int(pop_size * p_val))
            pbest_pool = sorted_indices[:p_count]
            
            # Randomly select p-best for each individual
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1 (distinct from i)
            # Vectorized cyclic shift ensures r1 != i
            shift = np.random.randint(1, pop_size, pop_size)
            r1_idx = (np.arange(pop_size) + shift) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 (distinct from i and r1) from Union(Pop, Archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            
            # Generate r2 indices
            r2_idx = np.random.randint(0, len(union_pop), pop_size)
            
            # Collision handling (r2 != i and r2 != r1)
            curr_indices = np.arange(pop_size)
            bad_mask = (r2_idx == curr_indices) | (r2_idx == r1_idx)
            
            # Retry loop for collisions (usually resolves in 1-2 tries)
            retry_limit = 5
            while np.any(bad_mask) and retry_limit > 0:
                n_bad = np.sum(bad_mask)
                r2_idx[bad_mask] = np.random.randint(0, len(union_pop), n_bad)
                bad_mask = (r2_idx == curr_indices) | (r2_idx == r1_idx)
                retry_limit -= 1
            
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant Vector
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            f_col = f[:, np.newaxis] # Reshape for broadcasting
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, np.newaxis]
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 6. Bound Handling (Clip)
            trial = np.clip(trial, min_b, max_b)
            
            # 7. Selection & Evaluation
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_fitness = np.zeros(pop_size)
            
            for i in range(pop_size):
                if check_timeout(): return global_best_val
                
                trial_val = func(trial[i])
                
                if trial_val <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fitness[i] = trial_val
                    
                    if trial_val < fitness[i]:
                        succ_mask[i] = True
                        diff_fitness[i] = fitness[i] - trial_val
                        archive.append(pop[i].copy())
                        
                    if trial_val < global_best_val:
                        global_best_val = trial_val
                        global_best_sol = trial[i].copy()
            
            pop = new_pop
            fitness = new_fitness
            
            # 8. Memory Update
            if np.any(succ_mask):
                succ_f = f[succ_mask]
                succ_cr = cr[succ_mask]
                succ_diff = diff_fitness[succ_mask]
                
                # Weighted Mean based on fitness improvement
                weights = succ_diff / np.sum(succ_diff)
                
                # Update CR Memory
                mean_cr = np.sum(weights * succ_cr)
                mem_cr[k_mem] = mean_cr
                
                # Update F Memory (Weighted Lehmer Mean)
                mean_f = np.sum(weights * (succ_f**2)) / (np.sum(weights * succ_f) + 1e-10)
                mean_f = np.clip(mean_f, 0, 1)
                mem_f[k_mem] = mean_f
                
                k_mem = (k_mem + 1) % H
            
            # 9. Restart Triggers
            # Check for stagnation
            if global_best_val < last_best_val:
                last_best_val = global_best_val
                stag_count = 0
            else:
                stag_count += 1
            
            # Trigger A: Stagnation limit reached
            if stag_count >= stag_limit:
                break 
            
            # Trigger B: Population minimal and converged (variance is low)
            if pop_size <= min_pop_size:
                if np.std(fitness) < 1e-8:
                    break
        
        # End of Epoch: Update start time for next restart epoch
        epoch_start_time = datetime.now()

    return global_best_val
