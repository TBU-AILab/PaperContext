#An improved algorithm based on **L-SHADE-R (L-SHADE with Adaptive Restarts)** is proposed below.
#
#### Key Improvements
#1.  **Self-Adaptive Scheduling**: The Linear Population Size Reduction (LPSR) and parameter adaptation schedules are dynamically recalculated at the beginning of each restart based on the *remaining* global time. This ensures that every restart cycle is optimized to reach full exploitation (minimum population size) exactly when the time limit expires, maximizing the search efficiency for the remaining budget.
#2.  **Aggressive Restart Triggers**: The algorithm monitors population variance and fitness stagnation. If the population collapses into a single point (variance < tolerance) or fails to improve the local best for a set number of generations, it triggers a restart. This prevents wasting time in local optima.
#3.  **Elitism with Soft Re-initialization**: Upon restarting, the global best solution is preserved (Elitism) to ensure monotonic improvement. The rest of the population is re-initialized to explore new basins of attraction.
#4.  **Midpoint-Target Boundary Handling**: Instead of clipping values to bounds (which biases search to edges), solutions violating bounds are set to the midpoint between the parent and the bound. This preserves the search direction and improves convergence near boundaries.
#5.  **Optimized Vectorization**: The code uses efficient NumPy operations for all evolutionary operators and reduces the overhead of time-checking by batching checks during evaluation loops.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-R (Linear Population Reduction SHADE with Restarts).
    
    Key Features:
    - Adaptive Restarts based on stagnation and convergence.
    - Linear Population Size Reduction (LPSR) scaled to remaining time.
    - Current-to-pbest/1 mutation with external archive.
    - Midpoint-target boundary handling.
    """
    # -------------------------------------------------------------------------
    # Setup and Constants
    # -------------------------------------------------------------------------
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Algorithm Parameters
    # Initial population size: Scaled by dim, clamped for safety/speed
    pop_size_init = int(np.clip(25 * dim, 50, 200))
    pop_size_min = 4
    
    # SHADE Parameters
    H = 6                   # Memory size
    arc_rate = 2.5          # Archive size relative to population
    
    # Restart triggers
    stop_tol = 1e-6         # Population variance tolerance
    stall_limit_base = 50   # Base generations for stagnation check
    stall_limit = max(stall_limit_base, 5 * dim)

    # Global Best Tracking
    global_best_val = float('inf')
    global_best_vec = None
    
    # -------------------------------------------------------------------------
    # Helper: Check Time
    # -------------------------------------------------------------------------
    def get_remaining_seconds():
        elapsed = (datetime.now() - start_time).total_seconds()
        return max_time - elapsed

    # -------------------------------------------------------------------------
    # Main Restart Loop
    # -------------------------------------------------------------------------
    while True:
        # Check if we have enough time to start a meaningful epoch
        rem_time = get_remaining_seconds()
        if rem_time < 0.05: # Buffer to return safely
            return global_best_val
            
        # ---------------------------------------------------------------------
        # Initialization for New Epoch
        # ---------------------------------------------------------------------
        # The epoch targets the REST of the available time.
        # This scales LPSR effectively whether we have 60s or 5s left.
        epoch_start_time = datetime.now()
        epoch_duration_budget = rem_time
        
        # Initialize Population
        pop_size = pop_size_init
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best if it exists
        if global_best_vec is not None:
            pop[0] = global_best_vec.copy()
            fitness[0] = global_best_val
            # We must re-evaluate pop[0] in the loop or skip it. 
            # Simplified: we just skip overwriting it during init eval if calculated.
            # But simpler logic: just keep it in array, evaluation loop handles it.
            
        # Memories for SHADE
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive
        archive = []
        
        # Local (Epoch) tracking
        epoch_best_val = float('inf')
        stall_counter = 0
        
        # ---------------------------------------------------------------------
        # Initial Evaluation (Epoch)
        # ---------------------------------------------------------------------
        for i in range(pop_size):
            # Check time periodically (every 10 evals) to reduce overhead
            if i % 10 == 0:
                if (datetime.now() - start_time) >= time_limit:
                    return global_best_val
            
            # If we injected global best at index 0, we can skip eval if we trust func is deterministic
            # But let's evaluate to be safe and update fitness array
            if i == 0 and global_best_vec is not None:
                # fitness[0] is already set, but we might need to confirm if func is noisy
                # For this problem, assuming deterministic, but re-eval is safer for consistency
                pass 
                
            val = func(pop[i])
            fitness[i] = val
            
            if val < epoch_best_val:
                epoch_best_val = val
                
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
                
        # ---------------------------------------------------------------------
        # Evolutionary Loop (Generations)
        # ---------------------------------------------------------------------
        while True:
            # 1. Time & Progress Check
            now = datetime.now()
            elapsed_total = (now - start_time).total_seconds()
            if elapsed_total >= max_time:
                return global_best_val
            
            # Progress of THIS epoch relative to its specific budget
            # This ensures LPSR finishes exactly when time runs out
            epoch_elapsed = (now - epoch_start_time).total_seconds()
            progress = epoch_elapsed / epoch_duration_budget
            if progress > 1.0: progress = 1.0
            
            # 2. Linear Population Size Reduction (LPSR)
            target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Reduce population: keep best
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices[:target_size]]
                fitness = fitness[sort_indices[:target_size]]
                pop_size = target_size
                
                # Resize Archive
                curr_arc_cap = int(pop_size * arc_rate)
                if len(archive) > curr_arc_cap:
                    # Randomly remove to maintain diversity
                    keep_idxs = np.random.choice(len(archive), curr_arc_cap, replace=False)
                    archive = [archive[k] for k in keep_idxs]
            
            # 3. Restart Checks (Stagnation/Convergence)
            # Trigger A: Stagnation (Best value not improving)
            # Note: We compare epoch_best_val with previous generation's best
            current_best = np.min(fitness)
            if current_best < epoch_best_val - 1e-8:
                epoch_best_val = current_best
                stall_counter = 0
            else:
                stall_counter += 1
                
            # Trigger B: Convergence (Variance too low)
            if np.std(fitness) < stop_tol or stall_counter >= stall_limit:
                # Epoch finished (stuck), break to Restart Outer Loop
                break
                
            # Trigger C: End of Schedule
            if pop_size <= pop_size_min and stall_counter > 10:
                # If we are at min size and not improving, restart
                break

            # 4. Parameter Adaptation
            # p for current-to-pbest: Linear decay 0.2 -> 0.05
            p_val = 0.2 - (0.15 * progress)
            p_val = max(0.05, p_val)
            
            # Generate F and CR (Vectorized)
            r_idxs = np.random.randint(0, H, pop_size)
            mu_cr = mem_cr[r_idxs]
            mu_f = mem_f[r_idxs]
            
            # CR ~ Normal(mu, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(mu, 0.1)
            f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            f[f > 1] = 1.0
            
            # Repair F <= 0
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                n_neg = np.sum(mask_neg)
                f[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(n_neg)
                f[f > 1] = 1.0
                
            # 5. Mutation: current-to-pbest/1
            # V = X + F(Xp - X) + F(Xr1 - Xr2)
            
            # Sort for pbest selection
            sorted_idx = np.argsort(fitness)
            n_pbest = max(2, int(p_val * pop_size))
            pbest_pool = sorted_idx[:n_pbest]
            
            # Select pbest
            pbest_idxs = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idxs]
            
            # Select r1 (!= i)
            # We generate r1 randomly, then shift collisions
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            # Collision with i
            col_i = (r1_idxs == np.arange(pop_size))
            r1_idxs[col_i] = (r1_idxs[col_i] + 1) % pop_size
            x_r1 = pop[r1_idxs]
            
            # Select r2 (!= i, != r1) from Pop + Archive
            if len(archive) > 0:
                pop_all = np.vstack((pop, np.array(archive)))
            else:
                pop_all = pop
            n_all = len(pop_all)
            
            r2_idxs = np.random.randint(0, n_all, pop_size)
            # Fix collisions (iterative is fast enough for small vectors)
            mask_bad = (r2_idxs == np.arange(pop_size)) | (r2_idxs == r1_idxs)
            while np.any(mask_bad):
                n_bad = np.sum(mask_bad)
                r2_idxs[mask_bad] = np.random.randint(0, n_all, n_bad)
                mask_bad = (r2_idxs == np.arange(pop_size)) | (r2_idxs == r1_idxs)
            x_r2 = pop_all[r2_idxs]
            
            # Compute Mutant
            f_v = f[:, None]
            mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
            
            # 6. Crossover (Binomial)
            rand_cr = np.random.rand(pop_size, dim)
            mask_cross = rand_cr < cr[:, None]
            # Force at least one dimension from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trials = np.where(mask_cross, mutant, pop)
            
            # 7. Bound Handling (Midpoint Target)
            # If trial < min, trial = (min + parent)/2
            mask_l = trials < min_b
            if np.any(mask_l):
                trials = np.where(mask_l, (min_b + pop) * 0.5, trials)
            
            mask_u = trials > max_b
            if np.any(mask_u):
                trials = np.where(mask_u, (max_b + pop) * 0.5, trials)
                
            # 8. Evaluation & Selection
            succ_f = []
            succ_cr = []
            diffs = []
            
            for i in range(pop_size):
                # Batch time check
                if i % 10 == 0:
                    if (datetime.now() - start_time) >= time_limit:
                        return global_best_val
                
                f_tri = func(trials[i])
                
                if f_tri <= fitness[i]:
                    # Success
                    if f_tri < fitness[i]:
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diffs.append(fitness[i] - f_tri)
                        archive.append(pop[i].copy())
                    
                    pop[i] = trials[i]
                    fitness[i] = f_tri
                    
                    if f_tri < global_best_val:
                        global_best_val = f_tri
                        global_best_vec = trials[i].copy()
                        stall_counter = 0 # Reset stall if global improves
            
            # 9. Memory Update
            # Maintain archive size
            curr_arc_cap = int(pop_size * arc_rate)
            while len(archive) > curr_arc_cap:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update M_CR, M_F
            if len(succ_f) > 0:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                df = np.array(diffs)
                
                # Weighted Lehmer Mean
                total_diff = np.sum(df)
                w = df / total_diff
                
                # Mean F (Lehmer)
                denom = np.sum(w * s_f)
                if denom == 0: m_f = 0.5
                else: m_f = np.sum(w * s_f**2) / denom
                
                # Mean CR (Weighted Arithmetic)
                m_cr = np.sum(w * s_cr)
                
                mem_f[k_mem] = m_f
                mem_cr[k_mem] = m_cr
                k_mem = (k_mem + 1) % H

    return global_best_val
