#Here is a fully operational, self-contained Python implementation. 
#
#**Improvements Explanation:**
#The previous algorithm (L-SHADE + Coordinate Descent) failed to achieve a near-zero result (scoring ~5.08) likely because it lacked a **Restart Mechanism**. Once it converged to a local optimum (the "basin" of 5.08), it spent the rest of `max_time` polishing that suboptimal solution.
#
#This implementation introduces **L-SHADE-RSP (Restart with Scoping and Polishing)**:
#1.  **Automated Restarts**: The algorithm does not run once. It detects convergence (when the population effectively collapses). Upon convergence, it records the result and **restarts** the search.
#2.  **Smart Restart Strategy**:
#    *   **Global Restart**: Fresh random population to find new basins of attraction.
#    *   **Local Restart**: If a good solution is found, a restart is triggered with a Gaussian cloud centered around the best-so-far solution to refine it (Exploitation).
#3.  **MTS-LS1 (Multiple Trajectory Search - Local Search 1)**: Replaces simple Coordinate Descent. This is a robust local search method often paired with differential evolution to handle non-separable problems better than axis-aligned searches.
#4.  **Optimized Bounds Handling**: Uses "Reflection" (bouncing off the walls) instead of "Clipping". This preserves population diversity better near the boundaries.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Implements L-SHADE with Multi-Start and MTS-LS1 Local Search.
    
    This algorithm overcomes local optima by detecting population stagnation,
    triggering restarts, and applying aggressive local polishing.
    """
    
    # --- Configuration ---
    # Convert bounds to numpy for vectorization
    bounds_np = np.array(bounds)
    lb_glob = bounds_np[:, 0]
    ub_glob = bounds_np[:, 1]
    
    # Global State
    start_time = time.time()
    global_best_val = float('inf')
    global_best_vec = None
    
    def get_remaining_time():
        return max_time - (time.time() - start_time)

    def clip_reflect(vec, low, high):
        """
        Reflects values that go out of bounds back into the domain.
        This maintains diversity better than simple clipping.
        """
        # Reflect
        mask_l = vec < low
        mask_u = vec > high
        # Calculate amount of violation
        # (We use simple reflection: if x < lb, x = lb + (lb - x))
        # Note: multiple reflections are handled by modulo logic or simply clamping after one reflection
        # For speed, we do one reflection then clamp.
        
        vec[mask_l] = 2 * low[mask_l] - vec[mask_l]
        vec[mask_u] = 2 * high[mask_u] - vec[mask_u]
        
        # Clamp strictly in case reflection was still out
        return np.clip(vec, low, high)

    # --- MTS-LS1 Local Search ---
    def mts_ls1_polish(best_vec, best_val, local_time_budget):
        """
        Multiple Trajectory Search (MTS) Local Search 1.
        Explores dimensions with a shrinking step size.
        """
        ls_start = time.time()
        x = best_vec.copy()
        f_x = best_val
        
        # Determine search range (search range acts as step size)
        # We start with a fraction of the domain
        sr = (ub_glob - lb_glob) * 0.4
        
        # Minimum search range
        min_sr = 1e-15
        
        improved = True
        
        while np.any(sr > min_sr):
            if (time.time() - ls_start) > local_time_budget:
                break
            if (time.time() - start_time) > max_time:
                break

            improved = False
            # Randomize dimension order
            dims = np.random.permutation(dim)
            
            for i in dims:
                # Try negative direction
                x_new = x.copy()
                x_new[i] -= sr[i]
                x_new = np.clip(x_new, lb_glob, ub_glob)
                
                v = func(x_new)
                
                if v < f_x:
                    f_x = v
                    x = x_new
                    improved = True
                else:
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] += 0.5 * sr[i] # MTS trick: asymmetric step checks
                    x_new = np.clip(x_new, lb_glob, ub_glob)
                    v = func(x_new)
                    
                    if v < f_x:
                        f_x = v
                        x = x_new
                        improved = True
            
            if not improved:
                sr /= 2.0 # Shrink search range
            
        return x, f_x

    # --- Core L-SHADE Algorithm (Single Epoch) ---
    def run_lshade_epoch(lb, ub, timeout_sec, initial_pop=None):
        nonlocal global_best_val, global_best_vec
        
        # Parameters
        epoch_start = time.time()
        
        # Population Size (Linear Reduction)
        N_init = int(round(max(20, 18 * dim)))
        N_min = 4
        
        # Initialize Population
        if initial_pop is None:
            pop = lb + np.random.rand(N_init, dim) * (ub - lb)
        else:
            # If provided (e.g., from restart cloud), use it, but ensure size matches N_init
            pop = initial_pop
            if len(pop) < N_init:
                fill = lb + np.random.rand(N_init - len(pop), dim) * (ub - lb)
                pop = np.vstack((pop, fill))
            elif len(pop) > N_init:
                pop = pop[:N_init]

        pop_size = N_init
        
        # Evaluate Initial
        fitness = np.array([func(ind) for ind in pop])
        
        # Update Global Best immediately
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < global_best_val:
            global_best_val = fitness[min_idx]
            global_best_vec = pop[min_idx].copy()

        # Archive
        archive = []
        arc_rate = 2.0
        max_arc_size = int(N_init * arc_rate)
        
        # Memory for Adaptive Parameters (H=History size)
        H = 6
        mem_M_cr = np.full(H, 0.5)
        mem_M_f  = np.full(H, 0.5)
        k_mem = 0
        
        # Evaluation Counter estimate for Linear Reduction
        # We estimate max evaluations based on time
        # Run a few loops to gauge speed? No, just use time ratio.
        
        # Loop
        while True:
            t_current = time.time()
            # Stop conditions
            if (t_current - start_time) >= max_time:
                return
            if (t_current - epoch_start) >= timeout_sec:
                return
            
            # Convergence Check (Std Dev of population)
            # If population is very tight, we are stuck or done.
            pop_std = np.std(pop, axis=0)
            if np.mean(pop_std) < 1e-12:
                return 

            # 1. Linear Population Reduction
            # Calculate progress based on time consumed in this epoch vs allocated timeout
            progress = (t_current - epoch_start) / timeout_sec
            progress = min(progress, 1.0)
            
            plan_pop_size = int(round((N_min - N_init) * progress + N_init))
            
            if pop_size > plan_pop_size:
                # Reduction
                # Sort indices
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx]
                fitness = fitness[sort_idx]
                
                num_remove = pop_size - plan_pop_size
                # Remove worst
                pop = pop[:-num_remove]
                fitness = fitness[:-num_remove]
                
                # Resize archive
                curr_arc_size = len(archive)
                req_arc_size = int(plan_pop_size * arc_rate)
                if curr_arc_size > req_arc_size:
                    # Randomly delete
                    del_indices = np.random.choice(curr_arc_size, curr_arc_size - req_arc_size, replace=False)
                    # Use list comprehension for safe deletion (or numpy masking)
                    # Convert archive to numpy for easier handling
                    if curr_arc_size > 0:
                        arc_np = np.array(archive)
                        keep_mask = np.ones(curr_arc_size, dtype=bool)
                        keep_mask[del_indices] = False
                        archive = list(arc_np[keep_mask])

                pop_size = plan_pop_size

            # 2. Parameter Generation
            # Indices for memory
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_M_cr[r_idx]
            m_f = mem_M_f[r_idx]
            
            # Generate CR (Normal)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            # Fold -1 (from box constraint handling in papers) -> 0.0 usually, but clip is fine
            
            # Generate F (Cauchy)
            # f = cauchy(loc=m_f, scale=0.1)
            f = np.random.standard_cauchy(pop_size) * 0.1 + m_f
            # Handle F <= 0 (regenerate) and F > 1 (clip to 1)
            while True:
                bad_f = f <= 0
                if not np.any(bad_f):
                    break
                f[bad_f] = np.random.standard_cauchy(np.sum(bad_f)) * 0.1 + m_f[bad_f]
            f = np.minimum(f, 1.0)
            
            # 3. Mutation: current-to-pbest/1
            # Sort for pbest selection
            sorted_indices = np.argsort(fitness)
            # p is random in [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            p_val = max(p_val, 2.0/pop_size) # Safety
            top_p = int(max(1, p_val * pop_size))
            
            pbest_indices = sorted_indices[np.random.randint(0, top_p, pop_size)]
            x_pbest = pop[pbest_indices]
            
            # r1: random from pop, distinct from i
            r1_indices = np.random.randint(0, pop_size, pop_size)
            # Ensure distinctness (simple cyclic shift if collision)
            collision = (r1_indices == np.arange(pop_size))
            r1_indices[collision] = (r1_indices[collision] + 1) % pop_size
            x_r1 = pop[r1_indices]
            
            # r2: random from Union(pop, archive), distinct from i, r1
            if len(archive) > 0:
                full_pool = np.vstack((pop, np.array(archive)))
            else:
                full_pool = pop
            
            pool_size = len(full_pool)
            r2_indices = np.random.randint(0, pool_size, pop_size)
            # Ignore detailed collision check for r2 speed; DE is robust to minor overlaps
            x_r2 = full_pool[r2_indices]
            
            # Compute Mutant
            # v = x + F*(xp - x) + F*(xr1 - xr2)
            diff1 = x_pbest - pop
            diff2 = x_r1 - x_r2
            F_col = f[:, np.newaxis]
            
            mutant = pop + F_col * diff1 + F_col * diff2
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            rand_uni = np.random.rand(pop_size, dim)
            CR_col = cr[:, np.newaxis]
            
            cross_mask = rand_uni <= CR_col
            # Enforce j_rand
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            
            final_mask = cross_mask | j_rand_mask
            
            trial = np.where(final_mask, mutant, pop)
            trial = clip_reflect(trial, lb, ub)
            
            # 5. Selection
            # Evaluate trials
            # To save time, we can iterate, but vector eval is not possible with func(x)
            new_pop = pop.copy()
            new_fit = fitness.copy()
            
            successful_indices = []
            diff_fitness = []
            
            for i in range(pop_size):
                if (time.time() - start_time) >= max_time:
                    return
                
                y = trial[i]
                fy = func(y)
                
                if fy <= fitness[i]: # Accepted (<= allows movement on plateaus)
                    if fy < fitness[i]:
                        successful_indices.append(i)
                        diff_fitness.append(fitness[i] - fy)
                        # Archive bad solution
                        archive.append(pop[i].copy())
                    
                    new_pop[i] = y
                    new_fit[i] = fy
                    
                    if fy < global_best_val:
                        global_best_val = fy
                        global_best_vec = y.copy()
            
            # Maintain Archive Size
            while len(archive) > max_arc_size:
                # Remove random
                archive.pop(np.random.randint(0, len(archive)))

            # 6. Update Memory
            if len(successful_indices) > 0:
                succ_idx = np.array(successful_indices)
                w = np.array(diff_fitness)
                
                # Weighted Lehmer Mean
                # If all diffs are 0 (plateau movement), use uniform weight
                if np.sum(w) == 0:
                    w = np.ones(len(w))
                
                w = w / np.sum(w)
                
                s_cr = cr[succ_idx]
                s_f = f[succ_idx]
                
                mean_cr = np.sum(w * s_cr)
                mean_f_num = np.sum(w * (s_f**2))
                mean_f_den = np.sum(w * s_f)
                mean_f = mean_f_num / mean_f_den if mean_f_den > 0 else 0.5
                
                mem_M_cr[k_mem] = 0.5 * mem_M_cr[k_mem] + 0.5 * mean_cr
                mem_M_f[k_mem]  = 0.5 * mem_M_f[k_mem]  + 0.5 * mean_f
                
                k_mem = (k_mem + 1) % H
            
            pop = new_pop
            fitness = new_fit

    # --- Main Loop with Restarts ---
    restart_count = 0
    
    # First Epoch: Global Exploration
    # We allocate ~30% of time for the first run. If it finishes early, great.
    # If not, we cut it to allow for restarts/polishing.
    first_timeout = max_time * 0.3
    run_lshade_epoch(lb_glob, ub_glob, first_timeout)
    
    # Check if we have time left
    while get_remaining_time() > 0.1:
        restart_count += 1
        
        # Decide Restart Strategy
        # Strategy A: Polish Global Best (Local Search)
        # Allocate small slice of time for MTS-LS1
        polish_budget = min(get_remaining_time(), 0.5) # 0.5s max per polish step
        if global_best_vec is not None:
             # Use the polish function
            p_vec, p_val = mts_ls1_polish(global_best_vec, global_best_val, polish_budget)
            if p_val < global_best_val:
                global_best_val = p_val
                global_best_vec = p_vec
        
        if get_remaining_time() < 0.1:
            break
            
        # Strategy B: Local Restart (Exploitation) vs Global Restart (Exploration)
        # If we have a decent solution, try to find a better one nearby (Gaussian cloud)
        # 50/50 chance, or adaptive based on improvement?
        # Let's alternate.
        
        epoch_budget = get_remaining_time()
        # Don't spend all remaining time on one restart; split it
        # Assume we might want 2-3 more restarts
        epoch_budget = epoch_budget / 2.0
        
        if restart_count % 2 == 1 and global_best_vec is not None:
            # Local Restart
            # Generate population around global best
            # Sigma = 10% of domain width
            sigma = (ub_glob - lb_glob) * 0.05
            N_init = int(round(max(20, 18 * dim)))
            
            # Create Gaussian cloud
            local_pop = np.random.normal(global_best_vec, sigma, (N_init, dim))
            local_pop = np.clip(local_pop, lb_glob, ub_glob)
            
            run_lshade_epoch(lb_glob, ub_glob, epoch_budget, initial_pop=local_pop)
        else:
            # Global Restart
            run_lshade_epoch(lb_glob, ub_glob, epoch_budget)

    return global_best_val
