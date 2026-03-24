#The following Python code implements a **Restarting Memetic SHADE (Success-History Adaptive Differential Evolution)** algorithm. 
#
#This algorithm combines the robust global search capabilities of **SHADE** (which uses historical memory to adapt mutation and crossover parameters) with a **Coordinate Descent Local Search** to refine solutions. It also employs an **External Archive** to maintain diversity and a **Restart Mechanism** to escape local optima when the population stagnates or converges.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Memetic SHADE.
    
    Features:
    1. SHADE: Adapts F and CR parameters using a history of successful updates.
    2. External Archive: Preserves diversity by storing recently replaced good solutions.
    3. Coordinate Descent Local Search: Refines the best solution upon convergence/stagnation.
    4. Midpoint Bound Handling: Preserves evolutionary momentum near boundaries.
    5. Restart Mechanism: Resets population if converged to maximize time usage.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Population size: SHADE typically uses ~18*dim. 
    # Clamped to [30, 80] to balance exploration with generation speed.
    NP = int(np.clip(18 * dim, 30, 80))
    
    # Pre-process bounds for efficient operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracker
    global_best_val = float('inf')
    global_best_vec = None
    
    # -------------------------------------------------------------------------
    # Helper: Check Time
    # -------------------------------------------------------------------------
    def check_time():
        return datetime.now() - start_time >= time_limit

    # -------------------------------------------------------------------------
    # Local Search: Coordinate Descent
    # -------------------------------------------------------------------------
    def local_search(x_input, f_input):
        """
        Performs a lightweight coordinate descent (Pattern Search) 
        to polish the best solution found so far.
        """
        x_curr = x_input.copy()
        f_curr = f_input
        
        # Initial step sizes per dimension (5% of domain)
        steps = diff_b * 0.05
        
        # Limit max passes to ensure we don't use too much time
        max_passes = 20
        
        for _ in range(max_passes):
            if check_time(): break
            
            improved_pass = False
            # Randomize order of dimensions
            dims = np.random.permutation(dim)
            
            for j in dims:
                if check_time(): break
                
                original_x = x_curr[j]
                step = steps[j]
                
                # 1. Try moving in negative direction
                x_curr[j] = np.clip(original_x - step, min_b[j], max_b[j])
                val = func(x_curr)
                
                if val < f_curr:
                    f_curr = val
                    improved_pass = True
                    # Keep the change
                    continue
                
                # 2. Try moving in positive direction
                x_curr[j] = np.clip(original_x + step, min_b[j], max_b[j])
                val = func(x_curr)
                
                if val < f_curr:
                    f_curr = val
                    improved_pass = True
                    # Keep the change
                    continue
                
                # 3. No improvement: Revert and shrink step size
                x_curr[j] = original_x
                steps[j] *= 0.5
            
            # Termination: If steps are negligible relative to domain
            if np.max(steps / (diff_b + 1e-15)) < 1e-8:
                break
                
            if not improved_pass:
                # Early exit if a full pass provides no improvement
                break
                
        return x_curr, f_curr

    # -------------------------------------------------------------------------
    # Main Restart Loop
    # -------------------------------------------------------------------------
    while not check_time():
        
        # --- 1. Initialization ---
        pop = min_b + np.random.rand(NP, dim) * diff_b
        fitness = np.full(NP, float('inf'))
        
        # Evaluate Initial Population
        for i in range(NP):
            if check_time(): return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
        
        # SHADE History Memory (H=6)
        H = 6
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive (stores replaced parent vectors)
        archive = [] 
        
        # Stagnation Counter
        stag_count = 0
        max_stag = 20 # Generations without global improvement to trigger restart
        
        # --- 2. Evolutionary Loop ---
        while not check_time():
            
            # Sort population by fitness (needed for current-to-pbest)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Check for Convergence or Stagnation
            pop_converged = (np.abs(fitness[-1] - fitness[0]) < 1e-8)
            stagnated = (stag_count >= max_stag)
            
            if pop_converged or stagnated:
                # Before restarting, polish the best individual
                best_loc, best_f = local_search(pop[0], fitness[0])
                if best_f < global_best_val:
                    global_best_val = best_f
                    global_best_vec = best_loc
                # Break inner loop to restart
                break
            
            # --- Parameter Generation ---
            # Randomly select memory index for each individual
            r_idx = np.random.randint(0, H, NP)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.random.standard_cauchy(NP)
            f = np.where(f > 1, 1.0, f)
            f = np.where(f <= 0, 0.5, f) # If <= 0, reset to 0.5
            
            # --- Mutation: current-to-pbest/1 ---
            # p is random in [2/NP, 0.2]
            p_val = np.random.uniform(2/NP, 0.2)
            top_p_cnt = int(max(2, NP * p_val))
            
            # x_pbest
            pbest_idxs = np.random.randint(0, top_p_cnt, NP)
            x_pbest = pop[pbest_idxs]
            
            # x_r1 (random from pop)
            r1_idxs = np.random.randint(0, NP, NP)
            x_r1 = pop[r1_idxs]
            
            # x_r2 (random from Pop U Archive)
            if len(archive) > 0:
                archive_arr = np.array(archive)
                pop_archive = np.vstack((pop, archive_arr))
            else:
                pop_archive = pop
            
            r2_idxs = np.random.randint(0, len(pop_archive), NP)
            x_r2 = pop_archive[r2_idxs]
            
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_u = np.random.rand(NP, dim)
            mask = rand_u < cr[:, None]
            j_rand = np.random.randint(0, dim, NP)
            mask[np.arange(NP), j_rand] = True # Ensure at least one parameter changes
            
            trial = np.where(mask, mutant, pop)
            
            # --- Bound Handling (Midpoint) ---
            # If out of bounds, set to midpoint between parent and bound
            min_broad = np.tile(min_b, (NP, 1))
            max_broad = np.tile(max_b, (NP, 1))
            
            lower_viol = trial < min_broad
            if np.any(lower_viol):
                trial[lower_viol] = (min_broad[lower_viol] + pop[lower_viol]) * 0.5
                
            upper_viol = trial > max_broad
            if np.any(upper_viol):
                trial[upper_viol] = (max_broad[upper_viol] + pop[upper_viol]) * 0.5
            
            trial = np.clip(trial, min_broad, max_broad)
            
            # --- Selection and Memory Update ---
            new_fitness = np.zeros(NP)
            succ_mask = np.zeros(NP, dtype=bool)
            diff_f = []
            succ_cr = []
            succ_f = []
            
            improved_global = False
            
            for i in range(NP):
                if check_time(): return global_best_val
                
                f_trial = func(trial[i])
                new_fitness[i] = f_trial
                
                if f_trial <= fitness[i]:
                    succ_mask[i] = True
                    diff_f.append(fitness[i] - f_trial)
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        improved_global = True
            
            if improved_global:
                stag_count = 0
            else:
                stag_count += 1
            
            # Update Population and Archive
            success_indices = np.where(succ_mask)[0]
            if len(success_indices) > 0:
                # Add parents to archive
                for idx in success_indices:
                    archive.append(pop[idx].copy())
                
                # Limit archive size to NP
                while len(archive) > NP:
                    archive.pop(np.random.randint(0, len(archive)))
                
                # Update survivors
                pop[success_indices] = trial[success_indices]
                fitness[success_indices] = new_fitness[success_indices]
                
                # Update SHADE Memory (Weighted Lehmer Mean)
                diff_f_arr = np.array(diff_f)
                total_diff = np.sum(diff_f_arr)
                
                if total_diff > 0:
                    weights = diff_f_arr / total_diff
                    succ_f_arr = np.array(succ_f)
                    succ_cr_arr = np.array(succ_cr)
                    
                    # Mean wL for F
                    mean_f_l = np.sum(weights * succ_f_arr**2) / (np.sum(weights * succ_f_arr) + 1e-15)
                    # Mean wA for CR
                    mean_cr_a = np.sum(weights * succ_cr_arr)
                    
                    M_F[k_mem] = np.clip(mean_f_l, 0, 1)
                    M_CR[k_mem] = np.clip(mean_cr_a, 0, 1)
                    k_mem = (k_mem + 1) % H

    return global_best_val
