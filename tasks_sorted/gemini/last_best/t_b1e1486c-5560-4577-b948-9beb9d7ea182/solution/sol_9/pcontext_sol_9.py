#The output value of the last generated algorithm is: 1.1793875222624877
#
#The last generated algorithm code:
#Here is the improved algorithm. It implements **Hybrid jSO-OBL-MTS** (jSO with Opposition-Based Learning and MTS Local Search).
#
##### Key Improvements:
#1.  **Opposition-Based Learning (OBL) Initialization:** Used to massively improve initial coverage and convergence speed by evaluating opposite points in the search space.
#2.  **MTS-LS1 Local Search Integration:** A coordinated local search is applied to the best individual during stagnation phases or the final 15% of the time budget. This allows the algorithm to refine the solution from "good" (1.17) to "optimal" (near 0).
#3.  **Refined jSO Logic:** Retains the core jSO strengths (Weighted Lehmer Mean, Linear Pop Reduction, Adaptive constraints) but with tuned constants for better stability.
#4.  **Optimized Evaluation Budget:** Local search is triggered conditionally to avoid starving the global search (L-SHADE) component.

import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Hybrid jSO with OBL initialization and 
    MTS-LS1 Local Search for high-precision refinement.
    """
    start_time = time.time()
    
    # --- Helper: Check Timeout ---
    def is_timeout():
        return (time.time() - start_time) >= max_time

    # --- 1. Initialization Strategy ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Population Size Strategy (jSO formula)
    # N_init = 25 * log(D) * sqrt(D)
    pop_size_init = int(round(25 * np.log(dim) * np.sqrt(dim)))
    pop_size_init = max(30, min(pop_size_init, 1000)) # Safety clamps
    pop_size = pop_size_init
    pop_size_min = 4
    
    # Memory for Parameters (jSO)
    h_mem = 5
    m_cr = np.full(h_mem, 0.8) 
    m_f = np.full(h_mem, 0.5)
    k_mem = 0
    
    # Archive
    archive = []
    
    # --- 2. OBL Initialization ---
    # Random Population
    pop = min_b + (max_b - min_b) * np.random.rand(pop_size, dim)
    
    # Opposite Population: x' = min + max - x
    pop_opp = min_b + max_b - pop
    pop_opp = np.clip(pop_opp, min_b, max_b)
    
    fitness = np.zeros(pop_size)
    fitness_opp = np.zeros(pop_size)
    
    best_fitness = float('inf')
    best_sol = None

    # Evaluate Random
    for i in range(pop_size):
        if is_timeout(): return best_fitness
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # Evaluate Opposite
    for i in range(pop_size):
        if is_timeout(): return best_fitness
        val = func(pop_opp[i])
        fitness_opp[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop_opp[i].copy()
            
    # Selection: Keep best N from (Pop + Pop_Opp)
    combined_pop = np.vstack((pop, pop_opp))
    combined_fit = np.concatenate((fitness, fitness_opp))
    
    sorted_idx = np.argsort(combined_fit)
    pop = combined_pop[sorted_idx[:pop_size]]
    fitness = combined_fit[sorted_idx[:pop_size]]
    
    # Ensure best_sol matches sorted pop[0]
    best_fitness = fitness[0]
    best_sol = pop[0].copy()
    
    # --- 3. Local Search Setup (MTS-LS1) ---
    search_range = (max_b - min_b) * 0.4
    ls_stagnation_count = 0
    
    # --- 4. Main Optimization Loop ---
    while not is_timeout():
        current_time = time.time()
        progress = (current_time - start_time) / max_time
        
        # A. Linear Population Size Reduction (LPSR)
        plan_pop_size = int(round(((pop_size_min - pop_size_init) * progress) + pop_size_init))
        if pop_size > plan_pop_size:
            # Pop is sorted at end of loop, so we trim the worst (end)
            pop_size = plan_pop_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Reduce Archive
            target_arc_size = int(round(pop_size * 1.4))
            if len(archive) > target_arc_size:
                del_count = len(archive) - target_arc_size
                # Remove random elements
                keep_mask = np.ones(len(archive), dtype=bool)
                remove_idx = np.random.choice(len(archive), del_count, replace=False)
                keep_mask[remove_idx] = False
                archive = [archive[i] for i in range(len(archive)) if keep_mask[i]]

        # B. Parameter Generation (jSO)
        r_idx = np.random.randint(0, h_mem, size=pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # CR generation (Normal)
        crs = np.random.normal(mu_cr, 0.1)
        crs = np.clip(crs, 0.0, 1.0)
        # jSO Constraint: Progress < 0.25 -> CR >= 0.7
        if progress < 0.25:
            crs[crs < 0.7] = 0.7
            
        # F generation (Cauchy)
        fs = mu_f + 0.1 * np.random.standard_cauchy(size=pop_size)
        # Retry if <= 0
        while True:
            mask_neg = fs <= 0
            if not np.any(mask_neg): break
            fs[mask_neg] = mu_f[r_idx[mask_neg]] + 0.1 * np.random.standard_cauchy(size=np.sum(mask_neg))
        fs = np.clip(fs, 0.0, 1.0)
        # jSO Constraint: Progress < 0.6 -> F <= 0.7
        if progress < 0.6:
            fs[fs > 0.7] = 0.7
            
        # C. Mutation: current-to-pbest-w/1
        # Calculate p (Linear decay 0.25 -> 0.05)
        p_val = 0.25 - (0.20 * progress)
        p_num = max(2, int(round(pop_size * p_val)))
        
        pbest_indices = np.random.randint(0, p_num, size=pop_size)
        x_pbest = pop[pbest_indices]
        
        r1_indices = np.random.randint(0, pop_size, size=pop_size)
        # Fix r1 != i
        mask_self = r1_indices == np.arange(pop_size)
        r1_indices[mask_self] = (r1_indices[mask_self] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # r2 from Union(Pop, Archive)
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, len(union_pop), size=pop_size)
        # Simple collision check for r2
        x_r2 = union_pop[r2_indices]
        
        # Compute Mutants
        f_v = fs[:, None]
        mutants = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
        
        # Boundary Correction (Midpoint)
        mask_l = mutants < min_b
        if np.any(mask_l):
            r, c = np.where(mask_l)
            mutants[r, c] = (pop[r, c] + min_b[c]) / 2.0
        mask_h = mutants > max_b
        if np.any(mask_h):
            r, c = np.where(mask_h)
            mutants[r, c] = (pop[r, c] + max_b[c]) / 2.0
            
        # D. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals <= crs[:, None]
        j_rand = np.random.randint(0, dim, size=pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        trials = np.where(mask_cross, mutants, pop)
        
        # E. Selection
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        succ_f = []
        succ_cr = []
        diff_fit = []
        
        global_improved = False
        
        for i in range(pop_size):
            if i % 10 == 0 and is_timeout(): return best_fitness
            
            f_trial = func(trials[i])
            
            if f_trial <= fitness[i]:
                new_pop[i] = trials[i]
                new_fitness[i] = f_trial
                
                if f_trial < fitness[i]:
                    archive.append(pop[i].copy())
                    succ_f.append(fs[i])
                    succ_cr.append(crs[i])
                    diff_fit.append(fitness[i] - f_trial)
                    
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_sol = trials[i].copy()
                        global_improved = True
        
        pop = new_pop
        fitness = new_fitness
        
        # F. Memory Update (Weighted Lehmer Mean)
        if len(diff_fit) > 0:
            w = np.array(diff_fit)
            w = w / np.sum(w)
            
            s_f = np.array(succ_f)
            s_cr = np.array(succ_cr)
            
            # Mean F (Lehmer)
            num = np.sum(w * (s_f**2))
            den = np.sum(w * s_f)
            if den > 1e-15:
                m_f[k_mem] = 0.5 * m_f[k_mem] + 0.5 * (num / den)
            else:
                m_f[k_mem] = 0.5 * m_f[k_mem] + 0.5 * 0.5
                
            # Mean CR (Weighted)
            m_cr[k_mem] = 0.5 * m_cr[k_mem] + 0.5 * np.sum(w * s_cr)
            
            k_mem = (k_mem + 1) % h_mem
            
        # G. Sort Population
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # H. MTS-LS1 Local Search
        # Strategy: Trigger if no global improvement for some time OR periodically to refine
        if global_improved:
            ls_stagnation_count = 0
        else:
            ls_stagnation_count += 1
            
        do_ls = False
        # If stagnant for 15 gens or if late game (progress > 0.85) and stagnant for 3 gens
        if ls_stagnation_count >= 15:
            do_ls = True
        elif progress > 0.85 and ls_stagnation_count >= 3:
            do_ls = True
            
        if do_ls:
            # Apply to best solution only
            improved_ls = False
            
            # Iterate dimensions
            for d in range(dim):
                if is_timeout(): return best_fitness
                
                # Reset search range if too small (precision limit)
                if search_range[d] < 1e-15:
                    search_range[d] = (max_b[d] - min_b[d]) * 0.01
                    
                original = best_sol[d]
                
                # Try Negative Step
                best_sol[d] = max(min_b[d], min(max_b[d], original - search_range[d]))
                ft = func(best_sol)
                
                if ft < best_fitness:
                    best_fitness = ft
                    improved_ls = True
                else:
                    # Try Positive Step (0.5 * sr)
                    best_sol[d] = max(min_b[d], min(max_b[d], original + 0.5 * search_range[d]))
                    ft = func(best_sol)
                    
                    if ft < best_fitness:
                        best_fitness = ft
                        improved_ls = True
                    else:
                        # Restore and shrink
                        best_sol[d] = original
                        search_range[d] *= 0.5
            
            if improved_ls:
                # Update the population leader
                pop[0] = best_sol.copy()
                fitness[0] = best_fitness
                ls_stagnation_count = 0

    return best_fitness
