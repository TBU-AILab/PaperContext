#The following algorithm implements **iL-SHADE (Restart-based L-SHADE)**.
#
#### Reasoning for Improvements
#1.  **Restart-Based L-SHADE**: Standard L-SHADE reduces the population size linearly to force convergence. However, in a fixed-time scenario, it might converge too early (stagnation) or too late. This algorithm runs "sessions" of L-SHADE. If the population stagnates (low variance) or the session ends, it **restarts** with a fresh population (injecting the global best for elitism). This continually utilizes the remaining time to explore new basins of attraction.
#2.  **Adaptive Cooling Schedule**: The Linear Population Size Reduction (LPSR) and mutation parameter $p$ (top percentage) are calculated based on the *remaining time* of the current session, not the global time. This ensures that every restart attempts a full exploration-to-exploitation trajectory within its available window.
#3.  **Midpoint Bound Handling**: Instead of simple clipping (which biases the search to the edges), particles violating bounds are placed halfway between the parent and the bound. This maintains diversity near constraints.
#4.  **Weighted Lehmer Mean**: The adaptation of parameters $F$ and $CR$ uses the weighted Lehmer mean based on fitness improvement magnitude, prioritizing parameters that yield significant gains.
#
#### Algorithm Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using iL-SHADE (Restart-based L-SHADE).
    
    Strategy:
    1. Runs L-SHADE (Success-History Adaptive Differential Evolution with Linear Reduction).
    2. Adapts population reduction (LPSR) to the specific time budget of the current session.
    3. Detects stagnation (low variance) to trigger early restarts.
    4. Preserves the global best solution (Elitism) across restarts.
    """
    
    # --- Time Management ---
    t_global_start = time.time()
    # Reserve 5% buffer to ensure strictly valid return
    t_max = max_time * 0.95
    
    def get_elapsed():
        return time.time() - t_global_start

    # --- Problem Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracking ---
    best_val = float('inf')
    best_sol = None
    
    # --- Main Restart Loop ---
    # We treat the optimization as a sequence of "sessions".
    # Each session is a full L-SHADE run scaled to the remaining time.
    while True:
        elapsed = get_elapsed()
        remaining = t_max - elapsed
        
        # If very little time remains (< 0.2s), overhead outweighs benefit. Return.
        if remaining < 0.2:
            return best_val
            
        # --- Session Configuration ---
        
        # Population Sizing: 18 * dim is standard for SHADE.
        # We clamp it to [30, 250] to balance exploration speed vs evaluation cost.
        pop_size = int(round(18 * dim))
        pop_size = max(30, min(250, pop_size))
        
        # If time is critically short (< 1s), start with a smaller population to ensure progress.
        if remaining < 1.0:
            pop_size = min(50, pop_size)
            
        N_init = pop_size
        N_min = 4
        
        # Initialize Population (Uniform Random)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject the global best solution into the new population
        if best_sol is not None:
            pop[0] = best_sol.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if get_elapsed() > t_max: return best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
                
        # Sort population by fitness (required for L-SHADE logic)
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        
        # SHADE Memory Parameters (History size H=5)
        H = 5
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive (Stores successful parents to maintain diversity)
        # Size factor 2.0 * Initial Population
        archive = np.zeros((int(2.0 * N_init), dim))
        arc_count = 0
        
        # Session Time Tracking
        t_session_start = time.time()
        # The session plans to use ALL remaining time.
        # If it stagnates early, we break and restart.
        t_session_budget = t_max - (t_session_start - t_global_start)
        
        # --- Session Loop (Generational) ---
        while True:
            # 1. Global Time Check
            if get_elapsed() > t_max:
                return best_val
            
            # 2. Calculate Progress (Relative to Session Budget)
            # This drives the cooling schedules (LPSR and p-best)
            s_elapsed = time.time() - t_session_start
            progress = s_elapsed / t_session_budget
            
            if progress >= 1.0:
                break # Session finished
                
            # 3. Linear Population Size Reduction (LPSR)
            # Reduce population from N_init to N_min over the session duration
            N_next = int(round(N_init + (N_min - N_init) * progress))
            N_next = max(N_min, N_next)
            
            if pop_size > N_next:
                n_remove = pop_size - N_next
                # Population is sorted, so we remove the worst (end of array)
                pop = pop[:-n_remove]
                fitness = fitness[:-n_remove]
                pop_size = N_next
                
                # Resize Archive to maintain roughly 2.0 * pop_size
                target_arc = int(2.0 * pop_size)
                if arc_count > target_arc:
                    # Randomly discard excess archive members
                    keep_idxs = np.random.choice(arc_count, target_arc, replace=False)
                    archive[:target_arc] = archive[keep_idxs]
                    arc_count = target_arc
            
            # 4. Stagnation Check
            # If population variance is negligible, we are stuck in a local optimum.
            # Restart early to save time.
            if progress > 0.05: # Allow initial convergence phase
                fit_range = fitness[-1] - fitness[0]
                # If fitness range is extremely small relative to value
                if fit_range < 1e-8:
                    break # Trigger restart
            
            # 5. Parameter Generation (Vectorized)
            r_idxs = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idxs]
            m_f = mem_f[r_idxs]
            
            # CR: Normal distribution, clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            np.clip(cr, 0, 1, out=cr)
            
            # F: Cauchy distribution
            # F = m_f + 0.1 * tan(pi * (rand - 0.5))
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Repair F
            f[f > 1] = 1.0
            # If F <= 0, regenerate (Retry loop)
            while True:
                mask_bad = f <= 0
                if not np.any(mask_bad): break
                n_bad = np.sum(mask_bad)
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
                f[f > 1] = 1.0
            
            # 6. Mutation: current-to-pbest/1
            # Adaptive p: starts at 0.2 (exploration), ends at 0.11 (exploitation)
            p_val = 0.2 - (0.09 * progress)
            p_val = max(p_val, 2.0 / pop_size) # Ensure at least 2 individuals
            n_pbest = int(p_val * pop_size)
            n_pbest = max(2, n_pbest)
            
            # Select p-best (from top sorted individuals)
            pbest_idxs = np.random.randint(0, n_pbest, pop_size)
            x_pbest = pop[pbest_idxs]
            
            # Select r1 (distinct from i)
            r1 = np.random.randint(0, pop_size, pop_size)
            mask_self = r1 == np.arange(pop_size)
            r1[mask_self] = (r1[mask_self] + 1) % pop_size
            x_r1 = pop[r1]
            
            # Select r2 (from Population U Archive)
            n_union = pop_size + arc_count
            r2 = np.random.randint(0, n_union, pop_size)
            
            # Construct x_r2 array
            x_r2 = np.empty((pop_size, dim))
            mask_pop = r2 < pop_size
            x_r2[mask_pop] = pop[r2[mask_pop]]
            mask_arc = ~mask_pop
            if np.any(mask_arc):
                # Archive indices map from [pop_size, pop_size + arc_count) -> [0, arc_count)
                x_r2[mask_arc] = archive[r2[mask_arc] - pop_size]
                
            # Compute Mutant Vector V
            f_col = f[:, None]
            v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 7. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cr = np.random.rand(pop_size, dim) < cr[:, None]
            mask_cr[np.arange(pop_size), j_rand] = True
            u = np.where(mask_cr, v, pop)
            
            # 8. Bound Handling (Midpoint Method)
            # Place violated particles halfway between parent and bound
            mask_l = u < min_b
            mask_h = u > max_b
            
            if np.any(mask_l):
                cols = np.where(mask_l)[1]
                u[mask_l] = (pop[mask_l] + min_b[cols]) / 2.0
            if np.any(mask_h):
                cols = np.where(mask_h)[1]
                u[mask_h] = (pop[mask_h] + max_b[cols]) / 2.0
            
            # 9. Evaluation and Selection
            new_fitness = np.empty(pop_size)
            s_cr, s_f, s_df = [], [], []
            
            for i in range(pop_size):
                if get_elapsed() > t_max: return best_val
                
                val = func(u[i])
                new_fitness[i] = val
                
                if val <= fitness[i]:
                    # Improvement or Neutral
                    if val < fitness[i]:
                        # Store success data
                        s_cr.append(cr[i])
                        s_f.append(f[i])
                        s_df.append(fitness[i] - val)
                        
                        # Add replaced parent to archive
                        if arc_count < archive.shape[0]:
                            archive[arc_count] = pop[i].copy()
                            arc_count += 1
                        else:
                            # Replace random archive member
                            rand_idx = np.random.randint(0, arc_count)
                            archive[rand_idx] = pop[i].copy()
                    
                    if val < best_val:
                        best_val = val
                        best_sol = u[i].copy()
                else:
                    # Revert to parent
                    u[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # Update Population
            pop = u
            fitness = new_fitness
            
            # 10. Update SHADE Memory (Weighted Lehmer Mean)
            if len(s_df) > 0:
                s_df_np = np.array(s_df)
                s_cr_np = np.array(s_cr)
                s_f_np = np.array(s_f)
                
                total_df = np.sum(s_df_np)
                if total_df > 1e-15:
                    weights = s_df_np / total_df
                    
                    # Update M_CR
                    m_cr_new = np.sum(weights * s_cr_np)
                    mem_cr[k_mem] = m_cr_new
                    
                    # Update M_F (Lehmer Mean: sum(w*F^2) / sum(w*F))
                    num = np.sum(weights * (s_f_np ** 2))
                    den = np.sum(weights * s_f_np)
                    m_f_new = num / den if den > 1e-9 else 0.5
                    mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                    
                    k_mem = (k_mem + 1) % H
                    
            # 11. Sort Population (Critical for LPSR and p-best in next generation)
            sort_order = np.argsort(fitness)
            pop = pop[sort_order]
            fitness = fitness[sort_order]

    return best_val
