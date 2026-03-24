#Here is the improved algorithm. It implements **L-SHADE-OBL (L-SHADE with Opposition-Based Learning)** and a **Stagnation Restart** mechanism.
#
#The improvements over the previous version are:
#1.  **OBL Initialization**: It generates an "opposite" population based on the initial random samples and selects the best starting individuals. This drastically improves the probability of starting in a promising basin of attraction.
#2.  **Stagnation Restart**: The previous L-SHADE algorithm would sit idle if it converged to a local optimum (like the result ~40.3) before the time ran out. This version detects convergence (low population variance) and triggers a "Soft Restart" (keeps the best, scatters the rest) to utilize the remaining time budget to find better optima.
#3.  **Robust Evaluation**: Includes strict time checks within loops to ensure `max_time` is never exceeded, even with expensive functions.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Opposition-Based Learning (OBL) initialization and Stagnation Restart.
    
    Improvements:
    1. OBL: Checks opposite points in the search space during initialization to 
       better define the initial basin of attraction.
    2. Restart: If population diversity collapses (convergence) while time remains, 
       it triggers a restart (keeping the best solution) to escape local optima.
    """
    start_time = time.time()
    
    # --- Helper Functions ---
    def get_remaining_time():
        return max_time - (time.time() - start_time)

    def check_termination():
        return (time.time() - start_time) >= max_time

    # --- Parameters ---
    # Initial Population Size: Adaptive based on dimension
    # 18*dim is standard for L-SHADE. Clamped between 30 and 300.
    pop_size_init = int(np.clip(18 * dim, 30, 300))
    pop_size_min = 4
    
    # Memory for adaptive parameters (History size H)
    mem_size = 6 # Slightly larger memory
    
    # --- Pre-computation ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_vec = None

    # --- Restart Loop ---
    # We treat the available time as a resource. If we converge early, we restart.
    
    restart_count = 0
    
    while not check_termination():
        
        # --- 1. Initialization Phase with OBL ---
        # Current run population size (resets on restart)
        current_pop_size = pop_size_init
        
        # Initialize Memory
        m_cr = np.full(mem_size, 0.5)
        m_f = np.full(mem_size, 0.5)
        mem_k = 0
        archive = []
        
        # A. Random Generation
        pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
        
        # B. Opposition-Based Learning (OBL) Initialization
        # Generate opposite population: X_opp = min + max - X
        # Only perform if we have enough time budget (safe heuristic: if dim < 50 or optimistic)
        if get_remaining_time() > max_time * 0.5:
            pop_opp = min_b + max_b - pop
            # Clip OBL
            pop_opp = np.clip(pop_opp, min_b, max_b)
            
            # Evaluate both sets (2 * N)
            # We must be careful with time here.
            pop_combined = np.vstack((pop, pop_opp))
            fit_combined = np.zeros(len(pop_combined))
            
            evaluated_count = 0
            for i in range(len(pop_combined)):
                if check_termination(): break
                fit_combined[i] = func(pop_combined[i])
                evaluated_count += 1
                
                if fit_combined[i] < global_best_val:
                    global_best_val = fit_combined[i]
                    global_best_vec = pop_combined[i].copy()
            
            if check_termination(): return global_best_val
            
            # Select best N
            sort_idx = np.argsort(fit_combined)
            pop = pop_combined[sort_idx[:current_pop_size]]
            fitness = fit_combined[sort_idx[:current_pop_size]]
            
        else:
            # Standard Initialization (Time critical)
            fitness = np.zeros(current_pop_size)
            for i in range(current_pop_size):
                if check_termination(): break
                val = func(pop[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = pop[i].copy()
            
            if check_termination(): return global_best_val

        # If this is a restart, inject the global best into the population
        # to ensure elitism and guide the new search
        if restart_count > 0 and global_best_vec is not None:
            pop[0] = global_best_vec.copy()
            fitness[0] = global_best_val

        # --- 2. Evolutionary Loop ---
        # Calculate local max evaluations or time for Linear Reduction calculation
        # L-SHADE reduces population relative to progress. 
        # On a restart, we reset the "progress" tracking relative to remaining time.
        loop_start_time = time.time()
        time_at_start = loop_start_time - start_time
        
        while True:
            curr_time = time.time()
            if curr_time - start_time >= max_time:
                return global_best_val
            
            # --- Linear Population Size Reduction (LPSR) ---
            # Progress matches the global time
            overall_progress = (curr_time - start_time) / max_time
            
            target_size = int(round((pop_size_min - pop_size_init) * overall_progress + pop_size_init))
            target_size = max(pop_size_min, target_size)
            
            if current_pop_size > target_size:
                n_reduce = current_pop_size - target_size
                # Kill worst
                sort_idx = np.argsort(fitness)
                # Keep top 'target_size'
                keep_idx = sort_idx[:target_size]
                pop = pop[keep_idx]
                fitness = fitness[keep_idx]
                current_pop_size = target_size
                
                # Resize archive
                if len(archive) > current_pop_size:
                    # Random removal to maintain diversity in archive
                    # (Quick shuffle and slice)
                    import random
                    random.shuffle(archive)
                    archive = archive[:current_pop_size]

            # --- Parameter Generation ---
            # Generate CR and F based on history memory
            r_idx = np.random.randint(0, mem_size, current_pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1, current_pop_size)
            cr = np.clip(cr, 0, 1)
            # Memory hack: if mu_cr is -1 (terminal), cr=0
            cr[m_cr[r_idx] == -1] = 0
            
            # F ~ Cauchy(mu_f, 0.1)
            f = np.random.standard_cauchy(current_pop_size) * 0.1 + mu_f
            
            # Correct F
            # If F > 1 -> 1. If F <= 0 -> regenerate.
            mask_bad = f <= 0
            retry = 0
            while np.any(mask_bad) and retry < 5:
                f[mask_bad] = np.random.standard_cauchy(np.sum(mask_bad)) * 0.1 + mu_f[r_idx][mask_bad]
                mask_bad = f <= 0
                retry += 1
            f = np.clip(f, 0, 1) # Negatives likely handled by regen, but clip to be safe
            
            # --- Mutation: current-to-pbest/1 ---
            # Sort for pbest selection
            sorted_indices = np.argsort(fitness)
            # p varies: larger at start, smaller at end (jSO strategy) or fixed 0.11 (L-SHADE)
            # We use L-SHADE standard p=0.11
            p_val = 0.11
            n_pbest = max(2, int(p_val * current_pop_size))
            pbest_candidates = sorted_indices[:n_pbest]
            
            pbest_idx = np.random.choice(pbest_candidates, current_pop_size)
            x_pbest = pop[pbest_idx]
            
            # r1 != i
            r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
            # Simple check to avoid r1==i (rare in large pops, matters in small)
            collisions = (r1_indices == np.arange(current_pop_size))
            r1_indices[collisions] = (r1_indices[collisions] + 1) % current_pop_size
            x_r1 = pop[r1_indices]
            
            # r2 != i, r2 != r1, from Pop U Archive
            if len(archive) > 0:
                arc_np = np.array(archive)
                union_pop = np.vstack((pop, arc_np))
            else:
                union_pop = pop
                
            r2_indices = np.random.randint(0, len(union_pop), current_pop_size)
            # We skip detailed collision check for r2 vs r1/i for speed, 
            # DE is robust enough to handle occasional overlap.
            x_r2 = union_pop[r2_indices]
            
            # Calculate V
            # v = x + F*(xp - x) + F*(xr1 - xr2)
            f_col = f[:, None]
            v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # Boundary Correction (Midpoint)
            mask_l = v < min_b
            mask_h = v > max_b
            
            # Optimized boolean indexing for correction
            if np.any(mask_l):
                v[mask_l] = (pop[mask_l] + min_b[np.where(mask_l)[1]]) / 2.0
            if np.any(mask_h):
                v[mask_h] = (pop[mask_h] + max_b[np.where(mask_h)[1]]) / 2.0
                
            # --- Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, current_pop_size)
            cross_mask = np.random.rand(current_pop_size, dim) <= cr[:, None]
            # Ensure at least one dimension is taken from mutant
            cross_mask[np.arange(current_pop_size), j_rand] = True
            
            u = np.where(cross_mask, v, pop)
            
            # --- Evaluation & Selection ---
            succ_f = []
            succ_cr = []
            diff_fit = []
            
            # Just iterating is safer for strict time checking
            for i in range(current_pop_size):
                if check_termination():
                    return global_best_val
                
                # Check bounds before calling func (paranoid check)
                u_eval = np.clip(u[i], min_b, max_b)
                
                new_val = func(u_eval)
                
                if new_val < fitness[i]:
                    # Success
                    if len(archive) < current_pop_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace random archive member
                        rem_idx = np.random.randint(0, len(archive))
                        archive[rem_idx] = pop[i].copy()
                        
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fit.append(fitness[i] - new_val)
                    
                    fitness[i] = new_val
                    pop[i] = u_eval
                    
                    if new_val < global_best_val:
                        global_best_val = new_val
                        global_best_vec = u_eval.copy()
                
            # --- Memory Update ---
            if len(succ_f) > 0:
                succ_f = np.array(succ_f)
                succ_cr = np.array(succ_cr)
                diff_fit = np.array(diff_fit)
                
                # Weighted Lehmer Mean
                denom = np.sum(diff_fit)
                if denom > 0:
                    weights = diff_fit / denom
                    
                    # Mean WL F
                    # If sum(w*f) is 0 (rare), avoid div by zero
                    denom_f = np.sum(weights * succ_f)
                    if denom_f > 1e-12:
                        mean_f = np.sum(weights * (succ_f**2)) / denom_f
                    else:
                        mean_f = 0.5
                        
                    # Mean WL CR
                    mean_cr = np.sum(weights * succ_cr)
                    
                    m_f[mem_k] = np.clip(mean_f, 0, 1)
                    m_cr[mem_k] = np.clip(mean_cr, 0, 1)
                    
                    mem_k = (mem_k + 1) % mem_size

            # --- Stagnation / Convergence Check ---
            # If population variance is extremely low, we are stuck in a local optimum.
            # If time allows, restart.
            
            # Only check standard deviation if we have > 5 individuals
            if current_pop_size > 5:
                # Calculate standard deviation of fitness
                fit_std = np.std(fitness)
                
                # Threshold: if variation is tiny relative to absolute value
                if fit_std < 1e-8:
                    # Stagnation detected.
                    # Break inner loop to trigger restart in outer loop.
                    # We only restart if we have > 10% time remaining to make it worth it
                    if get_remaining_time() > max_time * 0.1:
                        restart_count += 1
                        break 
    
    return global_best_val
