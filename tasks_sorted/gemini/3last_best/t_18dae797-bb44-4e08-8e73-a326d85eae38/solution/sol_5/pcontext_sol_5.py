#The output value of the best generated algorithm is: 1.1088871283573951
#
#The best generated algorithm code:
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using L-SHADE (Linear Population Size Reduction
    Success-History Adaptive Differential Evolution) with Adaptive Restarts.
    """
    
    # --- Time Management ---
    # Use time.time() for low overhead checks
    t_start = time.time()
    # Reserve a small buffer (2%) to ensure strictly valid return within limit
    t_limit = max_time * 0.98
    t_end = t_start + t_limit

    def is_timeout():
        return time.time() >= t_end

    # --- Problem Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    global_best_val = float('inf')
    
    # --- Main Loop (Handles Restarts) ---
    # We restart if the population converges or if the L-SHADE schedule completes 
    # (reaches min population) and stagnates, provided time remains.
    while not is_timeout():
        
        # --- Session Setup ---
        session_start_time = time.time()
        remaining_time = t_end - session_start_time
        
        # If remaining time is too short to be useful, stop.
        if remaining_time < 0.05:
            break
            
        # --- L-SHADE Configuration ---
        # Initial Population Size: 18 * dim (standard L-SHADE), clamped for performance
        N_init = int(round(18 * dim))
        N_init = max(30, min(200, N_init))
        N_min = 4
        
        curr_pop_size = N_init
        
        # Initialize Population
        pop = min_b + np.random.rand(curr_pop_size, dim) * diff_b
        fitness = np.full(curr_pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(curr_pop_size):
            if is_timeout(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Sort population (required for LPSR and p-best)
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        
        # SHADE Memory Initialization
        H = 5
        mem_M_CR = np.full(H, 0.5)
        mem_M_F = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive (stores successful parents replaced by offspring)
        archive = []
        
        # --- Evolution Session ---
        while not is_timeout():
            
            # 1. Linear Population Size Reduction (LPSR)
            # Adapt reduction schedule to the *remaining* time for this restart
            t_now = time.time()
            elapsed = t_now - session_start_time
            # Progress 0.0 -> 1.0
            progress = elapsed / remaining_time if remaining_time > 1e-9 else 1.0
            
            # Calculate Target Population Size
            N_target = int(round(N_init + (N_min - N_init) * progress))
            N_target = max(N_min, N_target)
            
            # Reduce Population if needed
            if curr_pop_size > N_target:
                n_keep = N_target
                # Population is already sorted at end of loop
                pop = pop[:n_keep]
                fitness = fitness[:n_keep]
                curr_pop_size = n_keep
                
                # Resize Archive (maintain |A| <= |P|)
                while len(archive) > curr_pop_size:
                    archive.pop(np.random.randint(0, len(archive)))

            # 2. Convergence / Stagnation Check
            # If population variance is negligible, restart to explore elsewhere
            if curr_pop_size >= N_min:
                if np.std(fitness) < 1e-9:
                    break
            
            # 3. Parameter Generation
            r_idx = np.random.randint(0, H, curr_pop_size)
            m_cr = mem_M_CR[r_idx]
            m_f = mem_M_F[r_idx]
            
            # Generate CR (Normal dist)
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0, 1)
            
            # Generate F (Cauchy dist)
            F = m_f + 0.1 * np.tan(np.pi * (np.random.rand(curr_pop_size) - 0.5))
            
            # Repair F
            F[F > 1] = 1.0
            # Retry if <= 0
            while True:
                mask_bad = F <= 0
                if not np.any(mask_bad): break
                n_bad = np.sum(mask_bad)
                # Resample bad values
                F[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
                F[F > 1] = 1.0

            # 4. Mutation: current-to-pbest/1
            # Sort is needed for p-best selection
            # (Population is sorted at end of loop, so we are good)
            
            # Dynamic p-value (top %)
            p_min_val = 2.0 / curr_pop_size
            p_val = np.random.uniform(p_min_val, 0.2) if p_min_val < 0.2 else p_min_val
            n_pbest = int(max(2, np.ceil(curr_pop_size * p_val)))
            n_pbest = min(n_pbest, curr_pop_size)
            
            # Select p-best
            pbest_indices = np.random.randint(0, n_pbest, curr_pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (!= i)
            r1_indices = np.random.randint(0, curr_pop_size, curr_pop_size)
            # Handle collision r1 == i
            mask_col = r1_indices == np.arange(curr_pop_size)
            r1_indices[mask_col] = (r1_indices[mask_col] + 1) % curr_pop_size
            x_r1 = pop[r1_indices]
            
            # Select r2 (!= r1, != i) from Population U Archive
            if len(archive) > 0:
                pop_arc = np.vstack((pop, np.array(archive)))
            else:
                pop_arc = pop
            
            r2_indices = np.random.randint(0, len(pop_arc), curr_pop_size)
            # Note: Vectorized check for r2!=r1!=i is omitted for speed; 
            # DE is robust to occasional collision.
            x_r2 = pop_arc[r2_indices]
            
            # Compute Mutant Vector V
            F_exp = F[:, None]
            V = pop + F_exp * (x_pbest - pop) + F_exp * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            mask_cr = np.random.rand(curr_pop_size, dim) < CR[:, None]
            j_rand = np.random.randint(0, dim, curr_pop_size)
            mask_cr[np.arange(curr_pop_size), j_rand] = True
            
            U = np.where(mask_cr, V, pop)
            
            # 6. Bound Constraint Handling (Clip)
            U = np.clip(U, min_b, max_b)
            
            # 7. Selection and Evaluation
            new_fitness = np.empty(curr_pop_size)
            improved_indices = []
            diff_vals = []
            
            for i in range(curr_pop_size):
                if is_timeout(): return global_best_val
                
                f_new = func(U[i])
                new_fitness[i] = f_new
                
                if f_new <= fitness[i]:
                    improved_indices.append(i)
                    diff_vals.append(fitness[i] - f_new)
                    
                    if f_new < global_best_val:
                        global_best_val = f_new
            
            # 8. Update Population, Memory, Archive
            if improved_indices:
                imp_idx = np.array(improved_indices)
                diffs = np.array(diff_vals)
                
                # Add parents to Archive
                for ii in imp_idx:
                    archive.append(pop[ii].copy())
                # Maintain Archive size
                while len(archive) > curr_pop_size:
                    archive.pop(np.random.randint(0, len(archive)))
                
                # Update Population
                pop[imp_idx] = U[imp_idx]
                fitness[imp_idx] = new_fitness[imp_idx]
                
                # Update Memory (Weighted Lehmer Mean)
                total_diff = np.sum(diffs)
                if total_diff > 0:
                    w = diffs / total_diff
                    
                    # Update M_CR
                    s_cr = CR[imp_idx]
                    m_cr_new = np.sum(w * s_cr)
                    mem_M_CR[k_mem] = m_cr_new
                    
                    # Update M_F
                    s_f = F[imp_idx]
                    num = np.sum(w * (s_f ** 2))
                    den = np.sum(w * s_f)
                    m_f_new = num / den if den > 1e-9 else 0.5
                    mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                    
                    k_mem = (k_mem + 1) % H
            
            # 9. Sort Population (Critical for next iteration's LPSR and p-best)
            sort_order = np.argsort(fitness)
            pop = pop[sort_order]
            fitness = fitness[sort_order]

    return global_best_val
