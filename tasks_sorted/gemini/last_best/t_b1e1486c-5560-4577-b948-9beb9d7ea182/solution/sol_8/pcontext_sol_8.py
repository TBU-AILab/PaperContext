#Here is the improved algorithm. It implements **L-SHADE-MTS** (L-SHADE with Multiple Trajectory Search).
#
##### Key Improvements:
#1.  **MTS-LS1 Local Search:** This is the most critical addition to bridge the gap from a "good" solution (like 1.17) to the global optimum (near 0). It applies a coordinate-descent-style local search (MTS-LS1) to the best individual when the population stagnates or in the final phase of optimization.
#2.  **Opposition-Based Learning (OBL) Initialization:** Instead of a purely random start, it generates a population and its mathematical opposite, selecting the best halves. This provides a massive initial boost by covering the search space more effectively.
#3.  **jSO Adaptive Parameters:** It retains the sophisticated parameter adaptation logic (Weighted Lehmer Mean) from the jSO algorithm, which was identified as the strongest component of the previous best attempt.
#4.  **Linear Population Size Reduction (LPSR):** Linearly reduces the population size to shift from exploration to exploitation, but with a robust minimum size clamp.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-MTS (L-SHADE with Multiple Trajectory Search).
    Combines Opposition-Based Learning, jSO-style adaptation, and 
    MTS-LS1 local search for high-precision refinement.
    """
    start_time = time.time()
    
    # --- Configuration ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Population Size Strategy (jSO heuristic)
    # Start large for exploration
    pop_size_init = int(round(18 * dim))
    pop_size_init = max(30, min(pop_size_init, 500)) # Safety clamps
    pop_size_min = 4
    
    pop_size = pop_size_init
    
    # Memory for Adaptive Parameters (H=5)
    h_mem = 5
    m_cr = np.full(h_mem, 0.8) # Mean Crossover Rate
    m_f = np.full(h_mem, 0.5)  # Mean Scaling Factor
    k_mem = 0
    
    # Archive for preserving diversity
    archive = []
    
    # --- 1. Opposition-Based Learning (OBL) Initialization ---
    # Generate random population
    pop = min_b + (max_b - min_b) * np.random.rand(pop_size, dim)
    
    # Generate opposite population: x_opp = min + max - x
    pop_opp = min_b + max_b - pop
    pop_opp = np.clip(pop_opp, min_b, max_b)
    
    # Evaluate Random
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        if time.time() - start_time > max_time: return float('inf')
        fitness[i] = func(pop[i])
        
    # Evaluate Opposite
    fitness_opp = np.zeros(pop_size)
    for i in range(pop_size):
        if time.time() - start_time > max_time: return np.min(fitness)
        fitness_opp[i] = func(pop_opp[i])
        
    # Selection: Combine and keep best N
    combined_pop = np.vstack((pop, pop_opp))
    combined_fit = np.concatenate((fitness, fitness_opp))
    
    sorted_idx = np.argsort(combined_fit)
    pop = combined_pop[sorted_idx[:pop_size]]
    fitness = combined_fit[sorted_idx[:pop_size]]
    
    best_idx = 0
    best_fitness = fitness[best_idx]
    best_sol = pop[best_idx].copy()
    
    # --- Local Search Init (MTS-LS1) ---
    # Search range (step size) for each dimension
    sr = (max_b - min_b) * 0.4
    gens_no_improve = 0
    
    # --- Main Loop ---
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed >= max_time:
            return best_fitness
            
        progress = elapsed / max_time
        
        # --- 2. Linear Population Size Reduction (LPSR) ---
        plan_pop_size = int(round(((pop_size_min - pop_size_init) * progress) + pop_size_init))
        if pop_size > plan_pop_size:
            # Reduce population (remove worst individuals)
            pop_size = plan_pop_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Reduce archive (target size ~1.4 * pop_size)
            target_arc_size = int(pop_size * 1.4)
            if len(archive) > target_arc_size:
                # Random removal
                remove_count = len(archive) - target_arc_size
                idxs = np.random.choice(len(archive), remove_count, replace=False)
                # Create mask to keep
                keep_mask = np.ones(len(archive), dtype=bool)
                keep_mask[idxs] = False
                archive = [archive[i] for i in range(len(archive)) if keep_mask[i]]

        # --- 3. Parameter Generation ---
        r_idx = np.random.randint(0, h_mem, size=pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # CR: Normal Distribution
        crs = np.random.normal(mu_cr, 0.1)
        crs = np.clip(crs, 0.0, 1.0)
        # Constraint: Early stage prefers higher crossover
        if progress < 0.25:
            crs[crs < 0.7] = 0.7
            
        # F: Cauchy Distribution
        fs = mu_f + 0.1 * np.random.standard_cauchy(size=pop_size)
        while True:
            mask_neg = fs <= 0
            if not np.any(mask_neg): break
            fs[mask_neg] = mu_f[r_idx][mask_neg] + 0.1 * np.random.standard_cauchy(size=np.sum(mask_neg))
        fs = np.clip(fs, 0.0, 1.0)
        # Constraint: Early stage prevents overly aggressive steps
        if progress < 0.6:
            fs[fs > 0.7] = 0.7
            
        # --- 4. Mutation: current-to-pbest-w/1 ---
        # p linearly reduces from 0.25 to 0.05
        p_val = 0.25 - (0.20 * progress)
        p_num = max(2, int(pop_size * p_val))
        
        pbest_indices = np.random.randint(0, p_num, size=pop_size)
        x_pbest = pop[pbest_indices]
        
        r1_indices = np.random.randint(0, pop_size, size=pop_size)
        mask_s = r1_indices == np.arange(pop_size)
        r1_indices[mask_s] = (r1_indices[mask_s] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, len(union_pop), size=pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Calculate mutation vectors
        diff1 = x_pbest - pop
        diff2 = x_r1 - x_r2
        mutants = pop + fs[:, None] * diff1 + fs[:, None] * diff2
        
        # Boundary Handling (Midpoint)
        mask_l = mutants < min_b
        if np.any(mask_l):
            r, c = np.where(mask_l)
            mutants[r, c] = (pop[r, c] + min_b[c]) / 2.0
        mask_h = mutants > max_b
        if np.any(mask_h):
            r, c = np.where(mask_h)
            mutants[r, c] = (pop[r, c] + max_b[c]) / 2.0
            
        # --- 5. Crossover ---
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals <= crs[:, None]
        j_rand = np.random.randint(0, dim, size=pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        trials = np.where(mask_cross, mutants, pop)
        
        # --- 6. Selection ---
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        succ_f = []
        succ_cr = []
        diff_fit = []
        
        improved_global = False
        
        for i in range(pop_size):
            if i % 10 == 0 and (time.time() - start_time >= max_time):
                return best_fitness
                
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
                    improved_global = True
                    
        pop = new_pop
        fitness = new_fitness
        
        # --- 7. Memory Update (Weighted Lehmer Mean) ---
        if len(diff_fit) > 0:
            w = np.array(diff_fit)
            w = w / np.sum(w)
            
            # F update
            num = np.sum(w * (np.array(succ_f)**2))
            den = np.sum(w * np.array(succ_f))
            m_f[k_mem] = 0.5 * m_f[k_mem] + 0.5 * (num / (den + 1e-15))
            
            # CR update
            m_cr[k_mem] = 0.5 * m_cr[k_mem] + 0.5 * np.sum(w * np.array(succ_cr))
            
            k_mem = (k_mem + 1) % h_mem
            
        # Sort for next iteration
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # --- 8. Local Search Trigger (MTS-LS1) ---
        # Trigger if stagnant OR in final refinement phase
        if improved_global:
            gens_no_improve = 0
        else:
            gens_no_improve += 1
            
        pop_std = np.std(fitness)
        do_ls = False
        
        # Trigger conditions
        if gens_no_improve > 15: do_ls = True
        elif progress > 0.85 and gens_no_improve > 2: do_ls = True
        elif pop_std < 1e-6 and gens_no_improve > 5: do_ls = True
        
        if do_ls:
            # Run Local Search on best_sol
            improved_ls = False
            
            for j in range(dim):
                if time.time() - start_time >= max_time: return best_fitness
                
                # If Search Range is too small, reset or skip
                if sr[j] < 1e-15:
                    sr[j] = (max_b[j] - min_b[j]) * 0.005 # Small reset
                    
                original_x = best_sol[j]
                current_f = best_fitness
                
                # Try Negative Step
                best_sol[j] -= sr[j]
                best_sol[j] = max(min_b[j], min(max_b[j], best_sol[j]))
                
                f_new = func(best_sol)
                
                if f_new < current_f:
                    best_fitness = f_new
                    improved_ls = True
                    # Update population leader
                    pop[0] = best_sol.copy()
                    fitness[0] = best_fitness
                else:
                    # Revert and Try Positive Step (Half size)
                    best_sol[j] = original_x + 0.5 * sr[j]
                    best_sol[j] = max(min_b[j], min(max_b[j], best_sol[j]))
                    
                    f_new = func(best_sol)
                    
                    if f_new < current_f:
                        best_fitness = f_new
                        improved_ls = True
                        pop[0] = best_sol.copy()
                        fitness[0] = best_fitness
                    else:
                        # Revert and Shrink Range
                        best_sol[j] = original_x
                        sr[j] *= 0.5
                        
            if improved_ls:
                gens_no_improve = 0

    return best_fitness
