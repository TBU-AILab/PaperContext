#The following algorithm implements **L-SHADE-R (L-SHADE with Adaptive Restarts)**.
#
#**Key Improvements:**
#1.  **Adaptive Restart Mechanism**: To address the stagnation observed at the 7.107 value (likely a local optimum or deep basin), this algorithm implements a restart strategy. When the population converges (low variance), stagnates (no improvement over many generations), or completes its LPSR schedule (reaches minimum size), the search is restarted.
#2.  **Elitism Across Epochs**: When restarting, the global best solution found so far is preserved and injected into the new random population. This ensures the algorithm never loses the best known position while gaining fresh diversity to explore other basins.
#3.  **Epoch-Based LPSR and Adaptation**: The Linear Population Size Reduction (LPSR) and parameter adaptation ($p$ for p-best) are scaled based on the time remaining for the *current epoch*, rather than the total global time. This allows the algorithm to perform multiple full "schedules" of exploration-to-exploitation within the `max_time`.
#4.  **Midpoint-Target Constraint Handling**: Maintains the superior boundary handling strategy from the previous best attempt, which pulls solutions into the feasible space towards the parent rather than clipping them to the edge.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-R (Linear Population Reduction SHADE with Restarts).
    
    This algorithm splits the available time into 'epochs'. If the population 
    converges or stagnates within an epoch, a restart is triggered. 
    The best solution is preserved (Elitism) across restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Initial Population Size: 
    # Scaled by dimension but capped to ensure speed in Python.
    pop_size_init = max(40, min(150, 20 * dim))
    
    # Minimum Population Size (for LPSR)
    pop_size_min = 5
    
    # SHADE Parameters
    H = 6               # Memory size
    arc_rate = 2.0      # Archive size ratio
    
    # Restart triggers
    # If std dev of fitness < tol, restart.
    tol = 1e-6          
    # If no improvement for stall_limit generations, restart.
    stall_limit = max(50, 5 * dim)
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global Best tracking
    best_global_val = float('inf')
    best_global_vec = None
    
    # -------------------------------------------------------------------------
    # Main Optimization Loop (Handles Restarts)
    # -------------------------------------------------------------------------
    while True:
        # Check remaining time to decide if we can start a new epoch
        elapsed_total = (datetime.now() - start_time).total_seconds()
        remaining_time = max_time - elapsed_total
        
        # If very little time is left (< 0.1s or < 2%), return best result
        if remaining_time < 0.1 or remaining_time < 0.02 * max_time:
            return best_global_val
            
        # ---------------------------------------------------------------------
        # Epoch Setup
        # ---------------------------------------------------------------------
        epoch_start_time = datetime.now()
        
        # Reset Population
        pop_size = pop_size_init
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best into the new population to preserve progress
        if best_global_vec is not None:
            pop[0] = best_global_vec.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Reset Memories for the new epoch
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Epoch-local tracking
        best_epoch_val = float('inf')
        stall_counter = 0
        last_best_epoch = float('inf')
        
        # ---------------------------------------------------------------------
        # Initial Epoch Evaluation
        # ---------------------------------------------------------------------
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_global_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_epoch_val:
                best_epoch_val = val
            
            if val < best_global_val:
                best_global_val = val
                best_global_vec = pop[i].copy()
        
        # ---------------------------------------------------------------------
        # Evolutionary Cycle (One Epoch)
        # ---------------------------------------------------------------------
        while True:
            # Global Time Check
            if (datetime.now() - start_time) >= time_limit:
                return best_global_val
            
            # Calculate Progress relative to this epoch's remaining budget
            # We assume this epoch utilizes the rest of the time unless it converges early.
            epoch_now = datetime.now()
            epoch_elapsed = (epoch_now - epoch_start_time).total_seconds()
            
            # Progress goes from 0.0 to 1.0 based on the time remaining when the epoch STARTED.
            progress = epoch_elapsed / remaining_time
            if progress > 1.0: progress = 1.0
            
            # -----------------------------------------------------------------
            # 1. Linear Population Size Reduction (LPSR)
            # -----------------------------------------------------------------
            target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Reduce Population (Keep best)
                sort_indices = np.argsort(fitness)
                pop = pop[sort_indices[:target_size]]
                fitness = fitness[sort_indices[:target_size]]
                pop_size = target_size
                
                # Reduce Archive
                curr_arc_cap = int(pop_size * arc_rate)
                if len(archive) > curr_arc_cap:
                    # Random removal preserves diversity better
                    keep_idxs = np.random.choice(len(archive), curr_arc_cap, replace=False)
                    archive = [archive[k] for k in keep_idxs]
            
            # -----------------------------------------------------------------
            # 2. Parameter Adaptation
            # -----------------------------------------------------------------
            # p for p-best selection: Linear decay from 0.2 to 0.05
            p_val = 0.2 - (0.15 * progress)
            p_val = max(0.05, p_val)
            
            # Generate Adaptive Parameters (Vectorized)
            r_idxs = np.random.randint(0, H, pop_size)
            mu_cr = mem_cr[r_idxs]
            mu_f = mem_f[r_idxs]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            f[f > 1] = 1.0
            
            # Repair F <= 0
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                n_neg = np.sum(mask_neg)
                f[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(n_neg)
                f[f > 1] = 1.0
            
            # -----------------------------------------------------------------
            # 3. Mutation: current-to-pbest/1 (Vectorized)
            # -----------------------------------------------------------------
            sorted_idx = np.argsort(fitness)
            n_pbest = max(2, int(p_val * pop_size))
            pbest_pool = sorted_idx[:n_pbest]
            
            # Select p-best
            x_pbest = pop[np.random.choice(pbest_pool, pop_size)]
            
            # Select r1 (distinct from i)
            # Shift indices to avoid self-selection
            r1_idxs = (np.arange(pop_size) + np.random.randint(1, pop_size, pop_size)) % pop_size
            x_r1 = pop[r1_idxs]
            
            # Select r2 (distinct from i, r1) from Union(Pop, Archive)
            if len(archive) > 0:
                pop_all = np.vstack((pop, np.array(archive)))
            else:
                pop_all = pop
            n_all = len(pop_all)
            
            r2_idxs = np.random.randint(0, n_all, pop_size)
            
            # Fix collisions for r2
            mask_bad = (r2_idxs == np.arange(pop_size)) | (r2_idxs == r1_idxs)
            while np.any(mask_bad):
                n_bad = np.sum(mask_bad)
                r2_idxs[mask_bad] = np.random.randint(0, n_all, n_bad)
                mask_bad = (r2_idxs == np.arange(pop_size)) | (r2_idxs == r1_idxs)
            
            x_r2 = pop_all[r2_idxs]
            
            # Compute Mutant
            f_v = f[:, None]
            mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
            
            # -----------------------------------------------------------------
            # 4. Crossover (Binomial)
            # -----------------------------------------------------------------
            rand_cr = np.random.rand(pop_size, dim)
            mask_cross = rand_cr < cr[:, None]
            # Ensure at least one parameter comes from mutant
            mask_cross[np.arange(pop_size), np.random.randint(0, dim, pop_size)] = True
            
            trials = np.where(mask_cross, mutant, pop)
            
            # -----------------------------------------------------------------
            # 5. Boundary Handling (Midpoint Target)
            # -----------------------------------------------------------------
            # If out of bounds, place halfway between parent and bound.
            mask_l = trials < min_b
            if np.any(mask_l):
                trials = np.where(mask_l, (min_b + pop) / 2.0, trials)
            
            mask_u = trials > max_b
            if np.any(mask_u):
                trials = np.where(mask_u, (max_b + pop) / 2.0, trials)
                
            # -----------------------------------------------------------------
            # 6. Evaluation & Selection
            # -----------------------------------------------------------------
            succ_f = []
            succ_cr = []
            diffs = []
            
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_global_val
                
                f_tri = func(trials[i])
                
                if f_tri <= fitness[i]:
                    if f_tri < fitness[i]:
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diffs.append(fitness[i] - f_tri)
                        archive.append(pop[i].copy())
                    
                    pop[i] = trials[i]
                    fitness[i] = f_tri
                    
                    if f_tri < best_epoch_val:
                        best_epoch_val = f_tri
                    
                    if f_tri < best_global_val:
                        best_global_val = f_tri
                        best_global_vec = trials[i].copy()
                        # Reset stall counter if we found a new global best
                        # But for restart logic, we care about local stagnation too
            
            # -----------------------------------------------------------------
            # 7. Memory Update
            # -----------------------------------------------------------------
            curr_arc_cap = int(pop_size * arc_rate)
            while len(archive) > curr_arc_cap:
                archive.pop(np.random.randint(0, len(archive)))
            
            if len(succ_f) > 0:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                df = np.array(diffs)
                w = df / np.sum(df)
                
                # Weighted Lehmer Mean F
                denom = np.sum(w * s_f)
                if denom == 0:
                    mean_f = 0.5
                else:
                    mean_f = np.sum(w * s_f**2) / denom
                
                mem_f[k_mem] = mean_f
                mem_cr[k_mem] = np.sum(w * s_cr)
                k_mem = (k_mem + 1) % H
                
            # -----------------------------------------------------------------
            # 8. Restart Triggers
            # -----------------------------------------------------------------
            # Check stagnation (Epoch best hasn't improved)
            if best_epoch_val < last_best_epoch - 1e-10:
                last_best_epoch = best_epoch_val
                stall_counter = 0
            else:
                stall_counter += 1
                
            # Trigger A: LPSR Schedule Finished (Pop size reached minimum)
            if pop_size <= pop_size_min:
                break 
                
            # Trigger B: Population Convergence (Variance collapsed)
            if np.std(fitness) < tol:
                break
                
            # Trigger C: Stagnation (No improvement for too long)
            if stall_counter >= stall_limit:
                break

    return best_global_val
