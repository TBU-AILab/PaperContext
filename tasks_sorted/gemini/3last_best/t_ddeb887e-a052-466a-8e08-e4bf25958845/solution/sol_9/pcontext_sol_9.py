#The following algorithm implements **L-SHADE-R-OBL (Linear Success-History Adaptive Differential Evolution with Restarts, Opposition-Based Learning, and MTS-Local Search)**.
#
#It improves upon previous iterations by:
#1.  **Corrected Restart Budgeting**: Unlike previous versions that used global time for L-SHADE population reduction (causing instant population collapse upon restart), this version calculates the reduction curve based on the *remaining* time allocated to the current restart. This ensures a full exploration-exploitation cycle even in late restarts.
#2.  **OBL Initialization**: Uses Opposition-Based Learning to sample the search space more effectively during initialization.
#3.  **Midpoint Bound Handling**: Instead of simple clipping, it uses the midpoint between the violation and the bound, preserving population distribution near boundaries.
#4.  **Integrated MTS-LS1**: A local search (Modified Multiple Trajectory Search) is applied to the global best solution to rapidly refine the solution within the identified basin of attraction.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-R-OBL (Linear SHADE with Restarts, OBL, and MTS-LS1).
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Parameters ---
    # Population sizing: Linear reduction from max_pop to min_pop
    max_pop = int(np.clip(20 * dim, 60, 200))
    min_pop = 5
    
    # SHADE Memory parameters
    H = 6
    
    # Global Best Tracking
    best_fit = float('inf')
    best_sol = None
    
    # Restart Timer
    restart_start_time = start_time
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check remaining global time
        current_time = time.time()
        remaining_time = max_time - (current_time - start_time)
        if remaining_time <= 1e-4:
            return best_fit
            
        # 1. Initialization (OBL + Random)
        # We start with max_pop for every restart to ensure exploration
        pop_size = max_pop
        
        # Random Population
        p_rand = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best from previous restart
        if best_sol is not None:
            p_rand[0] = best_sol.copy()
            
        # Opposite Population
        p_opp = min_b + max_b - p_rand
        
        # Bound Correction for Opposite (Midpoint/Random fix)
        mask_out = (p_opp < min_b) | (p_opp > max_b)
        if np.any(mask_out):
            # Replace out-of-bounds opposites with random values
            p_opp[mask_out] = min_b[mask_out] + np.random.rand(np.sum(mask_out)) * diff_b[mask_out]
            
        # Combine and Evaluate
        combined_pop = np.vstack((p_rand, p_opp))
        combined_fit = np.zeros(len(combined_pop))
        
        for i in range(len(combined_pop)):
            if time.time() - start_time >= max_time: return best_fit
            val = func(combined_pop[i])
            combined_fit[i] = val
            if val < best_fit:
                best_fit = val
                best_sol = combined_pop[i].copy()
                
        # Select best N individuals
        sorted_idx = np.argsort(combined_fit)
        pop = combined_pop[sorted_idx[:pop_size]]
        fitness = combined_fit[sorted_idx[:pop_size]]
        
        # --- Run Configuration ---
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Local Search Step Size (MTS-LS1)
        # Initialize search radius relative to domain size
        sr = diff_b * 0.4
        
        # --- Generation Loop ---
        while True:
            t_now = time.time()
            if t_now - start_time >= max_time: return best_fit
            
            # 2. Linear Population Size Reduction (L-SHADE)
            # Calculated based on the "Lifetime" of this restart.
            # We treat the remaining time at the start of the restart as the budget.
            total_restart_budget = (start_time + max_time) - restart_start_time
            if total_restart_budget <= 1e-5: return best_fit
            
            progress = (t_now - restart_start_time) / total_restart_budget
            
            target_size = int(round((min_pop - max_pop) * progress + max_pop))
            target_size = max(min_pop, target_size)
            
            if pop_size > target_size:
                # Remove worst individuals
                # (Population is usually sorted or we sort now)
                s_idx = np.argsort(fitness)
                pop = pop[s_idx]
                fitness = fitness[s_idx]
                
                pop = pop[:target_size]
                fitness = fitness[:target_size]
                pop_size = target_size
                
                # Resize Archive
                if len(archive) > pop_size:
                    del archive[pop_size:]
            
            # 3. Parameter Generation
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # CR ~ Normal
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Fix F <= 0
            mask_bad = f <= 0
            while np.any(mask_bad):
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_bad)) - 0.5))
                mask_bad = f <= 0
            f = np.minimum(f, 1.0)
            
            # 4. Mutation: current-to-pbest/1
            # Sort for pbest selection
            s_idx = np.argsort(fitness)
            pop = pop[s_idx]
            fitness = fitness[s_idx]
            
            # Select p-best (random from top p%)
            p_val = np.random.uniform(2.0/pop_size, 0.2, pop_size)
            p_idx = (p_val * pop_size).astype(int)
            p_idx = np.maximum(p_idx, 1)
            
            rand_ranks = np.floor(np.random.rand(pop_size) * p_idx).astype(int)
            x_pbest = pop[rand_ranks]
            
            # Select r1 (!= i)
            r1 = np.random.randint(0, pop_size, pop_size)
            mask_s = (r1 == np.arange(pop_size))
            r1[mask_s] = (r1[mask_s] + 1) % pop_size
            x_r1 = pop[r1]
            
            # Select r2 (!= i, != r1, from Union)
            if len(archive) > 0:
                arr_arch = np.array(archive)
                union = np.vstack((pop, arr_arch))
            else:
                union = pop
            n_union = len(union)
            
            r2 = np.random.randint(0, n_union, pop_size)
            mask_c = (r2 == np.arange(pop_size)) | (r2 == r1)
            if np.any(mask_c):
                r2[mask_c] = np.random.randint(0, n_union, np.sum(mask_c))
            x_r2 = union[r2]
            
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # 5. Crossover
            mask_cr = np.random.rand(pop_size, dim) < cr[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cr[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cr, mutant, pop)
            
            # 6. Bound Handling (Midpoint Back-Projection)
            mask_l = trial < min_b
            mask_u = trial > max_b
            trial[mask_l] = (min_b[mask_l] + pop[mask_l]) / 2.0
            trial[mask_u] = (max_b[mask_u] + pop[mask_u]) / 2.0
            
            # 7. Selection
            success_diff = []
            success_cr = []
            success_f = []
            improved_global = False
            
            for i in range(pop_size):
                if time.time() - start_time >= max_time: return best_fit
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        # Add to archive
                        if len(archive) < pop_size:
                            archive.append(pop[i].copy())
                        else:
                            archive[np.random.randint(0, pop_size)] = pop[i].copy()
                        
                        success_diff.append(fitness[i] - f_trial)
                        success_cr.append(cr[i])
                        success_f.append(f[i])
                    
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    if f_trial < best_fit:
                        best_fit = f_trial
                        best_sol = trial[i].copy()
                        improved_global = True
            
            # 8. Memory Update
            if len(success_diff) > 0:
                w = np.array(success_diff)
                w /= np.sum(w)
                
                mean_cr = np.sum(w * np.array(success_cr))
                sf = np.array(success_f)
                mean_f = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-15)
                
                mem_cr[k_mem] = mean_cr
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                k_mem = (k_mem + 1) % H
            
            # 9. Local Search (MTS-LS1)
            # Trigger if we found a new best, or with low probability
            if improved_global or np.random.rand() < 0.02:
                # Search along dimensions
                dims = np.random.permutation(dim)
                ls_improved = False
                
                for d in dims:
                    if time.time() - start_time >= max_time: return best_fit
                    x_curr = best_sol[d]
                    
                    # Negative step
                    best_sol[d] = np.clip(x_curr - sr[d], min_b[d], max_b[d])
                    val = func(best_sol)
                    if val < best_fit:
                        best_fit = val
                        ls_improved = True
                        # Sync with population
                        # (Pop might have changed index of best, just overwrite index 0 is unsafe without sort
                        # but finding argmin is safe)
                        pop[np.argmin(fitness)] = best_sol.copy()
                        fitness[np.argmin(fitness)] = best_fit
                    else:
                        best_sol[d] = x_curr # Revert
                        
                        # Positive step (half size)
                        best_sol[d] = np.clip(x_curr + 0.5 * sr[d], min_b[d], max_b[d])
                        val = func(best_sol)
                        if val < best_fit:
                            best_fit = val
                            ls_improved = True
                            pop[np.argmin(fitness)] = best_sol.copy()
                            fitness[np.argmin(fitness)] = best_fit
                        else:
                            best_sol[d] = x_curr # Revert
                
                if not ls_improved:
                    sr *= 0.5 # Reduce search radius
                
                # Reset radius if too small to avoid underflow
                mask_sr = sr < 1e-15
                if np.any(mask_sr):
                    sr[mask_sr] = diff_b[mask_sr] * 0.4

            # 10. Stagnation Check
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break # Trigger Restart
                
        # Reset restart timer for next loop
        restart_start_time = time.time()
