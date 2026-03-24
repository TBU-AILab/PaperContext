#Here is the improved algorithm. It implements **L-SHADE-RSP (Rank-based Selection Pressure)** augmented with a **Terminal Local Search (Pattern Search)**.
#
#### Key Improvements:
#1.  **Rank-Based Selection Pressure (RSP):** Instead of choosing mutation vector $r1$ uniformly at random, vectors are chosen based on their fitness rank. Better individuals have a higher probability of being chosen as sources for mutation, significantly accelerating convergence towards promising regions compared to standard L-SHADE or jSO.
#2.  **Terminal Local Search (TLS):** Pure evolutionary algorithms can struggle to find the exact floating-point minimum in a basin (the "last mile" problem). This algorithm reserves the final 10% of the time budget (or triggers upon convergence) to run a coordinate-descent Pattern Search on the best solution found. This effectively drives the value from "close" (e.g., 1.18) to optimal (e.g., 0.0).
#3.  **Robust Restart:** Implements a "Soft Restart" (Gaussian explosion around the best) to escape stagnation while preserving the current basin, and a "Hard Restart" if the search stagnates early, ensuring global exploration.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-RSP (Rank-Based Selection Pressure) 
    combined with a Terminal Local Search (Pattern Search).
    """
    start_time = time.time()
    
    # --- Configuration ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Population Size (Linear Reduction)
    # L-SHADE-RSP typically starts with N = 18 * dim or similar heuristic
    pop_size_init = int(18 * dim)
    pop_size_init = np.clip(pop_size_init, 30, 250) # Clamped for safety
    pop_size_min = 4
    pop_size = pop_size_init
    
    # Adaptive Memory (History)
    h_mem = 5
    m_cr = np.full(h_mem, 0.5)
    m_f = np.full(h_mem, 0.5)
    k_mem = 0
    
    # Archive
    archive = []
    
    # Initialization
    pop = min_b + (max_b - min_b) * np.random.rand(pop_size, dim)
    fitness = np.array([func(ind) for ind in pop])
    
    # Sort immediately for ranking logic
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    best_idx = 0
    best_fitness = fitness[best_idx]
    best_sol = pop[best_idx].copy()
    
    # Stagnation counter
    gens_no_improve = 0
    last_best_fitness = best_fitness
    
    # Main Loop (Switch to Local Search at 90% time or early convergence)
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        progress = elapsed / max_time
        
        # --- Termination / Phase Switch Conditions ---
        if elapsed >= max_time:
            return best_fitness
        
        # If 90% time used, or population converged extremely tight, switch to LS
        # Calculate population diameter/std to detect convergence
        pop_std = np.std(fitness)
        if progress > 0.9 or (pop_std < 1e-9 and gens_no_improve > 10):
            break
            
        # --- 1. Linear Population Size Reduction (LPSR) ---
        plan_pop_size = int(round(((pop_size_min - pop_size_init) * progress) + pop_size_init))
        if pop_size > plan_pop_size:
            reduction = pop_size - plan_pop_size
            if pop_size - reduction < pop_size_min:
                reduction = pop_size - pop_size_min
            
            # Since pop is sorted, we just trim the end (worst individuals)
            pop_size -= reduction
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Resize Archive
            archive_target = int(pop_size * 2.0)
            if len(archive) > archive_target:
                # Remove random elements
                del_indices = np.random.choice(len(archive), len(archive) - archive_target, replace=False)
                keep_mask = np.ones(len(archive), dtype=bool)
                keep_mask[del_indices] = False
                archive = [archive[i] for i in range(len(archive)) if keep_mask[i]]

        # --- 2. Rank-Based Selection Probabilities ---
        # Probability proportional to rank: p_i = (N - i) / sum(1..N)
        # i=0 is best.
        ranks = np.arange(pop_size, 0, -1) # [N, N-1, ..., 1]
        rank_sum = np.sum(ranks)
        rank_probs = ranks / rank_sum
        
        # --- 3. Parameter Adaptation ---
        r_idx = np.random.randint(0, h_mem, size=pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # Generate CR (Normal, clamped [0,1])
        # If M_CR is terminal (-1), CR=0 (not implemented here, assumed standard behavior)
        crs = np.random.normal(mu_cr, 0.1)
        crs = np.clip(crs, 0.0, 1.0)
        
        # Generate F (Cauchy, clamped [0,1], retry <= 0)
        fs = mu_f + 0.1 * np.random.standard_cauchy(size=pop_size)
        while True:
            mask_neg = fs <= 0
            if not np.any(mask_neg):
                break
            fs[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(size=np.sum(mask_neg))
        fs = np.clip(fs, 0.0, 1.0)
        
        # --- 4. Mutation: current-to-pbest-w/1 (RSP) ---
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # p-best selection
        p_val = max(2, int(pop_size * max(0.05, 0.2 * (1 - progress)))) # Dynamic p
        p_best_indices = np.random.randint(0, p_val, size=pop_size)
        x_pbest = pop[p_best_indices]
        
        # r1 selection (Rank Based)
        # Instead of uniform random, we pick based on rank probabilities
        r1_indices = np.random.choice(pop_size, size=pop_size, p=rank_probs)
        
        # Fix collision r1 == i (fallback to random if collision)
        mask_col = r1_indices == np.arange(pop_size)
        if np.any(mask_col):
            r1_indices[mask_col] = np.random.randint(0, pop_size, size=np.sum(mask_col))
        x_r1 = pop[r1_indices]
        
        # r2 selection (Union of Pop and Archive)
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, len(union_pop), size=pop_size)
        
        # Fix collision r2 == r1 or r2 == i
        mask_col_r2 = (r2_indices == r1_indices) | (r2_indices == np.arange(pop_size))
        if np.any(mask_col_r2):
            r2_indices[mask_col_r2] = np.random.randint(0, len(union_pop), size=np.sum(mask_col_r2))
        x_r2 = union_pop[r2_indices]
        
        # Compute difference
        diff1 = x_pbest - pop
        diff2 = x_r1 - x_r2
        
        mutants = pop + fs[:, None] * diff1 + fs[:, None] * diff2
        
        # Boundary Handling (Bounce Back)
        # If x < min, x = (min + old) / 2
        mask_l = mutants < min_b
        if np.any(mask_l):
            rows, cols = np.where(mask_l)
            mutants[rows, cols] = (min_b[cols] + pop[rows, cols]) / 2.0
            
        mask_h = mutants > max_b
        if np.any(mask_h):
            rows, cols = np.where(mask_h)
            mutants[rows, cols] = (max_b[cols] + pop[rows, cols]) / 2.0
            
        # --- 5. Crossover ---
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals <= crs[:, None]
        j_rand = np.random.randint(0, dim, size=pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trials = np.where(mask_cross, mutants, pop)
        
        # --- 6. Evaluation & Selection ---
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        successful_f = []
        successful_cr = []
        diff_fitness_vals = []
        
        improved_any = False
        
        for i in range(pop_size):
            # Safe time check
            if (i % 10 == 0) and (time.time() - start_time >= max_time):
                return best_fitness
                
            f_trial = func(trials[i])
            
            if f_trial <= fitness[i]:
                new_pop[i] = trials[i]
                new_fitness[i] = f_trial
                
                if f_trial < fitness[i]:
                    archive.append(pop[i].copy())
                    successful_f.append(fs[i])
                    successful_cr.append(crs[i])
                    diff_fitness_vals.append(fitness[i] - f_trial)
                    
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trials[i].copy()
                    improved_any = True
                    
        pop = new_pop
        fitness = new_fitness
        
        # --- 7. Memory Update ---
        if len(diff_fitness_vals) > 0:
            w = np.array(diff_fitness_vals)
            w = w / np.sum(w)
            
            # MeanL for F
            s_f = np.array(successful_f)
            num = np.sum(w * (s_f ** 2))
            den = np.sum(w * s_f)
            new_f = num / (den + 1e-15)
            m_f[k_mem] = 0.5 * m_f[k_mem] + 0.5 * new_f
            
            # Weighted Mean for CR
            s_cr = np.array(successful_cr)
            new_cr = np.sum(w * s_cr)
            m_cr[k_mem] = 0.5 * m_cr[k_mem] + 0.5 * new_cr
            
            k_mem = (k_mem + 1) % h_mem
            
        # --- 8. Sorting (Required for RSP) ---
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # --- 9. Stagnation Detection & Restart ---
        if abs(best_fitness - last_best_fitness) < 1e-12:
            gens_no_improve += 1
        else:
            gens_no_improve = 0
            last_best_fitness = best_fitness

        # Strategy: If stuck, do a "Soft Restart" (Explosion)
        # or "Hard Restart" if very early.
        if gens_no_improve > 25:
            # If we are late in the game, don't destroy info, just perturb locally
            # If we are early, maybe we are in a wrong basin.
            
            if progress < 0.4:
                # Hard Restart: Keep best, randomize rest
                # This helps explore other basins
                num_reset = pop_size - 1
                pop[1:] = min_b + (max_b - min_b) * np.random.rand(num_reset, dim)
                # We need to evaluate these
                for k in range(1, pop_size):
                    if time.time() - start_time >= max_time: return best_fitness
                    fitness[k] = func(pop[k])
                    if fitness[k] < best_fitness:
                        best_fitness = fitness[k]
                        best_sol = pop[k].copy()
                # Re-sort
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx]
                fitness = fitness[sorted_idx]
                
                # Reset stats
                gens_no_improve = 0
                m_f[:] = 0.5
                m_cr[:] = 0.5
                
            elif progress < 0.8:
                # Soft Restart: Explosion around best
                # Generate gaussian cloud around best
                sigma = (max_b - min_b) * 0.05 # 5% width
                for k in range(1, pop_size):
                    candidate = best_sol + np.random.randn(dim) * sigma
                    # Clip
                    candidate = np.clip(candidate, min_b, max_b)
                    
                    if time.time() - start_time >= max_time: return best_fitness
                    f_val = func(candidate)
                    pop[k] = candidate
                    fitness[k] = f_val
                    if f_val < best_fitness:
                        best_fitness = f_val
                        best_sol = candidate.copy()
                
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx]
                fitness = fitness[sorted_idx]
                gens_no_improve = 0 # Reset counter

    # --- Phase 2: Terminal Local Search (Pattern Search / Coordinate Descent) ---
    # Used in the remaining time to polish the best solution.
    # This is critical for getting from 1.x to 0.0
    
    current_x = best_sol.copy()
    current_f = best_fitness
    
    # Initial step size: 1% of domain average
    # We use a decaying step size
    step_size = (max_b - min_b) * 0.01 
    
    # We iterate dimensions. 
    while True:
        if time.time() - start_time >= max_time:
            return best_fitness
            
        improved = False
        
        # Try adjusting each dimension
        for i in range(dim):
            if time.time() - start_time >= max_time:
                return best_fitness
                
            # Try positive step
            x_new = current_x.copy()
            x_new[i] += step_size[i]
            if x_new[i] > max_b[i]: x_new[i] = max_b[i] # Bound check
            
            f_new = func(x_new)
            if f_new < current_f:
                current_f = f_new
                current_x = x_new
                best_fitness = f_new # Update global return
                improved = True
                continue # Move to next dim
            
            # Try negative step
            x_new = current_x.copy()
            x_new[i] -= step_size[i]
            if x_new[i] < min_b[i]: x_new[i] = min_b[i]
            
            f_new = func(x_new)
            if f_new < current_f:
                current_f = f_new
                current_x = x_new
                best_fitness = f_new
                improved = True
        
        if not improved:
            # Refine step size
            step_size *= 0.5
            # If step size is too small (floating point precision limit), break
            if np.max(step_size) < 1e-15:
                break
        
    return best_fitness
