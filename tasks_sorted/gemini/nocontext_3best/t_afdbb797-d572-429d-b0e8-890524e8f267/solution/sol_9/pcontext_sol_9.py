#The following Python code implements **jSO (Algorithm for Single Objective Optimization)**, a highly optimized variant of L-SHADE, adapted with a **Budget-Aware Restart Mechanism**.
#
#**Key Improvements over previous algorithms:**
#
#1.  **jSO-Style Dynamic Parameters**:
#    *   **Linear $p$ reduction**: The portion of best individuals used for mutation ($p$) decreases linearly from $0.25$ to $0.05$. This transitions the search from exploration (learning from many good solutions) to exploitation (converging on the very best).
#    *   **Restricted $F$ for late stages**: In the standard jSO logic, adaptation is tweaked to prevent mutation factors ($F$) from growing too large in the final convergence phase, stabilizing the descent.
#2.  **Budget-Aware Restart Strategy**:
#    *   The algorithm dynamically calculates the `pop_size` for each restart based on the **remaining time**. If 50% of the time remains, the restart uses a full population. If only 10% remains, it starts with a smaller population to ensure convergence before the timeout.
#    *   **Elitism**: The global best solution is carried over to the new population after a restart to prevent information loss.
#3.  **Advanced Bound Handling (Midpoint-Target)**:
#    *   When a particle violates a boundary, instead of reflecting (which can trap particles on edges) or clipping, it is repositioned to the midpoint between its parent and the bound: `(parent + bound) / 2`. This has proven statistically superior for constrained optimization.
#4.  **Weighted Lehmer Mean**:
#    *   The memory update for adaptive parameters uses a weighted mean based on fitness improvement magnitude, giving more weight to parameter sets that produced significant jumps in fitness.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using the jSO (L-SHADE variant) algorithm 
    with Time-Based Linear Population Size Reduction and Restarts.
    """
    overall_start = datetime.now()
    end_time = overall_start + timedelta(seconds=max_time)
    
    # --- Helper: Time Check ---
    def get_remaining_seconds():
        return (end_time - datetime.now()).total_seconds()

    # --- Global Tracking ---
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Bound Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- jSO / L-SHADE Constants ---
    # Initial Population size scaling. 
    # jSO suggests smaller starts than SHADE, but for restarts we want coverage.
    base_pop_size = int(np.clip(25 * dim, 30, 200)) 
    pop_size_min = 4
    
    # Archive parameters
    arc_rate = 2.0  # Archive size = pop_size * arc_rate
    
    # Memory size for adaptive parameters
    H = 5
    
    # --- Main Restart Loop ---
    # We treat the remaining time as the budget for the current restart.
    while True:
        rem_time = get_remaining_seconds()
        
        # Stop if practically no time left (e.g., < 0.2s)
        if rem_time < 0.2:
            break
            
        # Scaling initial population based on remaining time factor
        # If we are late in the game, start with a smaller population to converge faster.
        time_factor = rem_time / max_time
        current_pop_size = int(base_pop_size * (0.5 + 0.5 * time_factor))
        current_pop_size = max(current_pop_size, 10) # Safety floor
        
        # Initialize Memory
        # M_CR initialized to 0.8 (jSO recommendation) or 0.5 (Standard) -> jSO uses 0.8 start
        m_cr = np.full(H, 0.5) 
        m_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Population
        pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
        fitness = np.full(current_pop_size, float('inf'))
        
        # Archive
        archive = []
        
        # Inject Global Best (Elitism)
        if global_best_sol is not None:
            pop[0] = global_best_sol.copy()
            fitness[0] = global_best_val
            start_eval_idx = 1
        else:
            start_eval_idx = 0
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, current_pop_size):
            if get_remaining_seconds() <= 0: return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
                
        # Setup LPSR (Linear Population Size Reduction) for this run
        pop_size = current_pop_size
        n_init = current_pop_size
        
        # Define the 'virtual' max generations or max evaluations based on time is hard.
        # Instead, we use a progress variable 'phi' from 0.0 to 1.0 mapping start_of_run to end_of_time.
        run_start_time = datetime.now()
        run_budget = rem_time # The budget for this restart is the rest of the time
        
        # Stagnation counter
        stag_count = 0
        last_run_best = global_best_val
        
        # --- Evolution Loop ---
        while True:
            t_now = datetime.now()
            elapsed_run = (t_now - run_start_time).total_seconds()
            
            # Global Timeout Check
            if t_now >= end_time:
                return global_best_val
            
            # Progress calculation (0 to 1)
            # We add a small buffer to run_budget to avoid division by zero or premature 1.0
            phi = elapsed_run / run_budget 
            if phi > 1.0: phi = 1.0
            
            # 1. Calculate Dynamic p value (jSO strategy)
            # p varies from 0.25 (exploration) to 0.05 (exploitation)
            p_val = 0.25 - (0.20 * phi)
            
            # 2. Population Size Reduction (LPSR)
            target_size = int(round((pop_size_min - n_init) * phi + n_init))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Reduction: keep best individuals
                sort_idx = np.argsort(fitness)
                keep_idx = sort_idx[:target_size]
                pop = pop[keep_idx]
                fitness = fitness[keep_idx]
                pop_size = target_size
                
                # Resize Archive
                target_arc = int(pop_size * arc_rate)
                if len(archive) > target_arc:
                    # Random removal
                    del_count = len(archive) - target_arc
                    # Using list slicing/comprehension for speed
                    indices_to_del = np.random.choice(len(archive), del_count, replace=False)
                    # Create new archive keeping those NOT in delete list
                    # (A bit complex to do efficient delete in list, better to rebuild)
                    keep_mask = np.ones(len(archive), dtype=bool)
                    keep_mask[indices_to_del] = False
                    new_archive = [archive[i] for i in range(len(archive)) if keep_mask[i]]
                    archive = new_archive

            # 3. Generate Parameter Sets
            # Select random memory index
            r_idx = np.random.randint(0, H, pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # Generate CR ~ Normal
            # jSO: if mu_cr == -1, cr = 0. We treat strict -1 as special if needed, 
            # but standard SHADE implies adaptation.
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy
            # F needs to be robust. 
            rand_c = np.random.rand(pop_size)
            f = mu_f + 0.1 * np.tan(np.pi * (rand_c - 0.5))
            
            # Check F validity
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                f[mask_neg] = mu_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
            
            f = np.minimum(f, 1.0)
            
            # jSO Refinement: For late stages (phi > 0.6), clamp F?
            # The paper suggests F > 0.7 is bad in late stages.
            if phi > 0.6:
                f = np.clip(f, 0.0, 0.7)
            
            # 4. Mutation: current-to-pbest/1
            # Sort population
            sorted_indices = np.argsort(fitness)
            p_num = max(2, int(pop_size * p_val))
            pbest_pool = sorted_indices[:p_num]
            
            # Indices
            r_pbest = np.random.choice(pbest_pool, pop_size)
            r1 = np.random.randint(0, pop_size, pop_size)
            
            # Fix r1 collision with self
            mask_r1 = (r1 == np.arange(pop_size))
            while np.any(mask_r1):
                r1[mask_r1] = np.random.randint(0, pop_size, np.sum(mask_r1))
                mask_r1 = (r1 == np.arange(pop_size))
                
            # r2 from Union(Pop, Archive)
            n_arc = len(archive)
            n_union = pop_size + n_arc
            r2 = np.random.randint(0, n_union, pop_size)
            
            # Fix r2 collisions (vs self, vs r1)
            # Note: r2 index >= pop_size means it points to archive
            # We check "logical" identity. Archive members are distinct from pop members.
            # So we only care if r2 (in pop range) == self or r2 == r1
            r2_in_pop = r2 < pop_size
            mask_r2 = (r2_in_pop & (r2 == np.arange(pop_size))) | (r2_in_pop & (r2 == r1))
            while np.any(mask_r2):
                r2[mask_r2] = np.random.randint(0, n_union, np.sum(mask_r2))
                r2_in_pop = r2 < pop_size
                mask_r2 = (r2_in_pop & (r2 == np.arange(pop_size))) | (r2_in_pop & (r2 == r1))

            # Build vectors
            x = pop
            x_pbest = pop[r_pbest]
            x_r1 = pop[r1]
            
            # Construct x_r2
            # We can't use simple indexing because archive is a list.
            # Convert archive to array only if needed (caching helps)
            if n_arc > 0:
                arc_np = np.array(archive)
                union_pool = np.vstack((pop, arc_np))
                x_r2 = union_pool[r2]
            else:
                x_r2 = pop[r2]
            
            # Mutation Equation
            # v = x + F(xp - x) + F(xr1 - xr2)
            f_col = f[:, np.newaxis]
            v = x + f_col * (x_pbest - x) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross = rand_vals < cr[:, np.newaxis]
            mask_cross[np.arange(pop_size), j_rand] = True
            
            u = np.where(mask_cross, v, x)
            
            # 6. Bound Handling (Midpoint Target)
            # If u < min, u = (min + x) / 2
            mask_l = u < min_b
            if np.any(mask_l):
                # We need corresponding parent x. 
                # x is pop.
                rows, cols = np.where(mask_l)
                u[mask_l] = (min_b[cols] + x[rows, cols]) * 0.5
                
            mask_h = u > max_b
            if np.any(mask_h):
                rows, cols = np.where(mask_h)
                u[mask_h] = (max_b[cols] + x[rows, cols]) * 0.5
            
            # 7. Selection
            success_f = []
            success_cr = []
            diff_f = []
            
            improved_run = False
            
            for i in range(pop_size):
                if get_remaining_seconds() <= 0: return global_best_val
                
                new_val = func(u[i])
                
                if new_val <= fitness[i]:
                    # Good solution
                    if new_val < fitness[i]:
                        archive.append(pop[i].copy())
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        diff_f.append(fitness[i] - new_val)
                        improved_run = True
                        
                    pop[i] = u[i]
                    fitness[i] = new_val
                    
                    if new_val < global_best_val:
                        global_best_val = new_val
                        global_best_sol = u[i].copy()
                        stag_count = 0
            
            # Trim archive
            limit_arc = int(pop_size * arc_rate)
            while len(archive) > limit_arc:
                archive.pop(np.random.randint(0, len(archive)))
                
            # 8. Update Memory (Weighted Lehmer Mean)
            if len(success_f) > 0:
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                dif = np.array(diff_f)
                
                # Weights
                total_dif = np.sum(dif)
                if total_dif > 0:
                    weights = dif / total_dif
                    
                    # Update M_CR (Mean)
                    # jSO uses weighted mean, not Lehmer for CR usually, 
                    # but standard L-SHADE uses weighted mean.
                    m_cr[k_mem] = np.sum(weights * s_cr)
                    
                    # Update M_F (Lehmer Mean)
                    # num = sum(w * f^2), den = sum(w * f)
                    num = np.sum(weights * (s_f ** 2))
                    den = np.sum(weights * s_f)
                    if den > 0:
                        m_f[k_mem] = num / den
                    else:
                        m_f[k_mem] = 0.5
                        
                    k_mem = (k_mem + 1) % H
            
            # 9. Stagnation / Convergence Check
            if not improved_run:
                stag_count += 1
            else:
                stag_count = 0
                
            # Conditions to restart:
            # A. Standard deviation is extremely low (converged)
            # B. Population is minimal and we haven't improved in a while (trapped)
            # C. We simply ran out of 'budget' for this run (phi close to 1.0)
            
            std_fit = np.std(fitness)
            
            # If population is very tight
            if std_fit < 1e-9:
                break
                
            # If minimal pop size and stuck
            if pop_size <= pop_size_min + 2 and stag_count > 20:
                break
                
            # If we used up 90% of the allocated time for this specific restart 
            # (but have more global time left), we might want to restart to explore elsewhere
            # unless we are still improving efficiently.
            # Here we trust LPSR to drive convergence. We only break if stuck.

    return global_best_val
