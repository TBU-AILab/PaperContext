#To improve upon the previous result, I propose the **jSO (JavaScript-based Self-Organizing)** algorithm, which is a highly optimized variant of L-SHADE that consistently ranks at the top of evolutionary computation benchmarks (e.g., CEC 2017).
#
#**Key Improvements over standard L-SHADE:**
#1.  **Forced Exploration Phase:** Specifically forces the Scale Factor ($F$) to remain high (0.7) during the first 60% of the search. This prevents the "early convergence" trap (likely the cause of the ~31.9 result in a multimodal landscape).
#2.  **Mid-Point Boundary Handling:** Instead of simply clipping values to bounds (which sticks particles to the edges), it places out-of-bound particles halfway between the parent and the bound. This preserves the search distribution.
#3.  **Weighted Lehmer Mean:** A more mathematically robust update strategy for the historical memory, giving higher weight to successful mutations that generated larger fitness improvements.
#4.  **Dynamic P-Best:** The number of "top" individuals used for mutation decreases over time, shifting from exploration to exploitation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Algorithm: jSO (L-SHADE variant with epistemic parameter adaptation).
    Optimized for finding global minima in limited time.
    """
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def get_progress():
        """Returns normalized time progress [0.0, 1.0]."""
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    # --- Initialization ---
    bounds = np.array(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    
    # jSO Population Sizing Strategy: N_init = 25 * log(D) * sqrt(D)
    # This scales better than fixed 18*D for various dimensions
    pop_size_init = int(25 * np.log(dim) * np.sqrt(dim))
    pop_size_init = max(30, pop_size_init) # Minimum safeguard
    pop_size = pop_size_init
    pop_size_min = 4

    # Allocate Population
    pop = np.random.uniform(min_b, max_b, (pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    # Using a safe loop to ensure we don't overrun time immediately
    best_val = float('inf')
    best_idx = -1
    
    for i in range(pop_size):
        if (datetime.now() - start_time).total_seconds() >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i
            
    # --- Memory Initialization ---
    # Memory size H
    H = 5
    mem_cr = np.full(H, 0.8) # Start with high crossover assumption
    mem_f = np.full(H, 0.5)
    k_mem = 0
    archive = [] # External archive for diversity

    # --- Main Loop ---
    while True:
        # 1. Time & Progress Check
        progress = get_progress()
        if progress >= 1.0:
            return best_val
            
        # 2. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on time progress
        plan_pop_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        if pop_size > plan_pop_size:
            # Reduce population: discard worst individuals
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices[:plan_pop_size]]
            fitness = fitness[sort_indices[:plan_pop_size]]
            pop_size = plan_pop_size
            
            # Reduce archive size to match capacity (usually A = P)
            archive_target = pop_size
            if len(archive) > archive_target:
                # Randomly remove excess
                del_count = len(archive) - archive_target
                # Safe random removal from list
                for _ in range(del_count):
                    archive.pop(np.random.randint(0, len(archive)))

        # 3. Parameter Generation (jSO Specifics)
        # Generate Memory Indices
        r_idxs = np.random.randint(0, H, pop_size)
        
        # Generate CR: Normal(M_cr, 0.1)
        cr = np.random.normal(mem_cr[r_idxs], 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        # In later stages, if CR is suspiciously low, nudge it (optional in jSO, implies separability)
        
        # Generate F: Cauchy(M_f, 0.1)
        # Cauchy = standard_cauchy * scale + loc
        f = np.random.standard_cauchy(pop_size) * 0.1 + mem_f[r_idxs]
        f = np.clip(f, 0.0, 1.0) # Clip high
        
        # Fix invalid F (negative/zero)
        # If F <= 0, regenerate until > 0 (simplified here as usually F > 0.1 is effective)
        retry_f_mask = f <= 0
        while np.any(retry_f_mask):
            f[retry_f_mask] = np.random.standard_cauchy(np.sum(retry_f_mask)) * 0.1 + mem_f[r_idxs][retry_f_mask]
            f = np.clip(f, -10, 1.0) # Temp clip to prevent infinity loop
            retry_f_mask = f <= 0
        
        # --- jSO Exploration Rule ---
        # If progress < 0.6 (First 60% of time), force F to be at least 0.7
        # This prevents premature convergence in local optima (like the 31.9 result)
        if progress < 0.6:
            f[f < 0.7] = 0.7

        # 4. Mutation: current-to-pbest/1
        # Dynamic p-value: shrinks from 0.25 to 0.05 over time
        p_val = 0.25 - (0.25 - 0.05) * progress
        p_count = max(2, int(p_val * pop_size))
        
        # Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        top_p_indices = sorted_indices[:p_count]
        
        # Select r_pbest
        pbest_indices = np.random.choice(top_p_indices, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Resolve collisions
        collision_r1 = (r1_indices == np.arange(pop_size))
        r1_indices[collision_r1] = (r1_indices[collision_r1] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Union of Pop and Archive)
        if len(archive) > 0:
            arr_archive = np.array(archive)
            pop_union = np.vstack((pop, arr_archive))
        else:
            pop_union = pop
            
        r2_indices = np.random.randint(0, len(pop_union), pop_size)
        # Resolve r2 collision against r1 and self
        # (Simplified: just ensure r2 != r1 is the critical one for differential)
        # Note: In standard DE, r2 != r1 != i.
        # With Union, checking strict uniqueness is costly. Statistical diversity usually suffices.
        x_r2 = pop_union[r2_indices]
        
        # Calculate Mutant Vector V
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        # Reshape F for broadcasting
        F_col = f.reshape(-1, 1)
        v = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        cross_mask[np.arange(pop_size), j_rand] = True # Ensure at least one
        
        u = np.where(cross_mask, v, pop)
        
        # 6. Bound Handling: Mid-Point Correction
        # Instead of clipping, put it halfway between parent and bound.
        # This helps escape bounds if the global optimum is near them.
        lower_mask = u < min_b
        upper_mask = u > max_b
        
        # u = (u + min_b) / 2 where lower violated
        u[lower_mask] = (pop[lower_mask] + min_b[np.where(lower_mask)[1]]) / 2.0
        # u = (u + max_b) / 2 where upper violated
        u[upper_mask] = (pop[upper_mask] + max_b[np.where(upper_mask)[1]]) / 2.0
        
        # 7. Selection (Evaluation)
        new_fitness = np.zeros(pop_size)
        success_mask = np.zeros(pop_size, dtype=bool)
        diff_f = []
        
        # Prepare batches for successful params
        win_cr = []
        win_f = []
        
        # Evaluate loop
        for i in range(pop_size):
            # Strict time check inside loop to maximize usage
            if (datetime.now() - start_time).total_seconds() >= max_time:
                return best_val
            
            f_new = func(u[i])
            new_fitness[i] = f_new
            
            if f_new < best_val:
                best_val = f_new
            
            if f_new < fitness[i]:
                success_mask[i] = True
                diff_f.append(fitness[i] - f_new)
                win_cr.append(cr[i])
                win_f.append(f[i])
                
                # Add parent to archive before replacement
                archive.append(pop[i].copy())
                
                # Update population
                pop[i] = u[i]
                fitness[i] = f_new
                
        # 8. Memory Update (Weighted Lehmer Mean)
        if len(win_cr) > 0:
            diff_f = np.array(diff_f)
            win_cr = np.array(win_cr)
            win_f = np.array(win_f)
            
            # Weights based on fitness improvement
            total_diff = np.sum(diff_f)
            weights = diff_f / total_diff
            
            # Update M_CR (Weighted Mean)
            # Avoid divide by zero
            m_cr_new = np.sum(weights * win_cr)
            
            # Update M_F (Weighted Lehmer Mean: sum(w*f^2)/sum(w*f))
            f_sum = np.sum(weights * win_f)
            if f_sum > 1e-10:
                m_f_new = np.sum(weights * (win_f ** 2)) / f_sum
            else:
                m_f_new = 0.5
                
            mem_cr[k_mem] = m_cr_new
            mem_f[k_mem] = m_f_new
            k_mem = (k_mem + 1) % H
            
        # 9. Convergence Check / Restart
        # If population collapses to a point (variance ~ 0), we are stuck.
        # If we have time left, we must restart to find other basins.
        if pop_size > 5:
            fitness_range = np.max(fitness) - np.min(fitness)
            if fitness_range < 1e-9 and progress < 0.9:
                # Keep best
                idx_best = np.argmin(fitness)
                best_ind = pop[idx_best].copy()
                best_fit_local = fitness[idx_best]
                
                # Scatter rest
                pop = np.random.uniform(min_b, max_b, (pop_size, dim))
                pop[0] = best_ind # Preserve best at index 0
                
                # Reset memory slightly to allow new adaptation
                mem_cr.fill(0.5)
                mem_f.fill(0.5)
                
                # Re-eval all (except preserved best, but simpler to just re-eval loop)
                # Need to update fitness array
                for i in range(pop_size):
                    if (datetime.now() - start_time).total_seconds() >= max_time: return best_val
                    if i == 0: 
                        fitness[i] = best_fit_local
                    else:
                        fitness[i] = func(pop[i])
                        if fitness[i] < best_val: best_val = fitness[i]

    return best_val
