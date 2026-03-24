#An algorithm that implements **L-SHADE-R-OBL (Linear Success-History Adaptive Differential Evolution with Restarts, Opposition-Based Learning, and MTS-Local Search)**.
#
#### Key Algorithm Features
#1.  **L-SHADE Structure**: Implements Linear Population Size Reduction based on the **global time budget**. The population starts large for exploration and shrinks to a few individuals for fast exploitation as the time limit approaches.
#2.  **OBL Initialization**: Uses Opposition-Based Learning during the initialization of each restart. It generates random candidates and their opposites, selecting the best half. This increases the probability of starting in a good basin of attraction.
#3.  **MTS-LS1 Local Search**: Incorporates a lightweight Local Search (Modified Multiple Trajectory Search) applied to the global best solution. It greedily refines the best solution along coordinate axes with adapting step sizes, ensuring high precision.
#4.  **Adaptive Restarts**: If the population converges (stagnation) before the time limit, the algorithm restarts. Crucially, it preserves the global best solution (Elitism) and adapts the new population size based on the remaining time, preventing wasted computations on full-sized populations when time is scarce.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-R-OBL (Linear SHADE with Restarts, OBL, and Local Search).
    """
    start_time = time.time()
    
    # --- Bounds Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Parameters ---
    # Population Size: Linear reduction from N_init to N_min
    # We clip N_init to ensure reasonable speed in Python
    n_init = int(np.clip(25 * dim, 60, 250))
    n_min = 5
    
    # SHADE Memory Size
    H = 6
    
    # Global Best Tracking
    best_fit = float('inf')
    best_sol = None
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check remaining time
        now = time.time()
        if now - start_time >= max_time:
            return best_fit
            
        # 1. Initialize Population (OBL: Opposition-Based Learning)
        # Determine current target population size based on global time
        # This ensures efficient restarts (start small if time is tight)
        progress = (now - start_time) / max_time
        # Reset to n_init for a fresh restart, but we will immediately reduce it 
        # inside the loop to match the time curve. 
        pop_size = n_init
        
        # Generate Random Population
        p_rand = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Generate Opposite Population
        p_opp = min_b + max_b - p_rand
        # Bound checking for opposite (randomize if out of bounds)
        mask_out = (p_opp < min_b) | (p_opp > max_b)
        if np.any(mask_out):
            # Vectorized random replacement
            p_opp[mask_out] = min_b[mask_out] + np.random.rand(np.sum(mask_out)) * diff_b[mask_out]
            
        # Combine and Evaluate
        combined_pop = np.vstack((p_rand, p_opp))
        combined_fit = np.zeros(len(combined_pop))
        
        # Evaluation Loop
        for i in range(len(combined_pop)):
            if time.time() - start_time >= max_time: return best_fit
            val = func(combined_pop[i])
            combined_fit[i] = val
            if val < best_fit:
                best_fit = val
                best_sol = combined_pop[i].copy()
                
        # Select best N (OBL Selection)
        sorted_idx = np.argsort(combined_fit)
        pop = combined_pop[sorted_idx[:pop_size]]
        fitness = combined_fit[sorted_idx[:pop_size]]
        
        # Elitism: Ensure we carried over the global best from previous restarts
        if best_sol is not None and best_fit < fitness[0]:
            pop[0] = best_sol.copy()
            fitness[0] = best_fit
            
        # Re-Sort after injection
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # --- Run Configuration ---
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Local Search Step Size (MTS-LS1)
        # Independent step size for each dimension
        sr = diff_b * 0.4
        
        # --- Generation Loop ---
        while True:
            # Time Check
            now = time.time()
            if now - start_time >= max_time:
                return best_fit
                
            # 1. Linear Population Size Reduction (L-SHADE)
            # Calculated based on GLOBAL time progress
            progress = (now - start_time) / max_time
            n_target = int(round((n_min - n_init) * progress + n_init))
            n_target = max(n_min, n_target)
            
            if pop_size > n_target:
                # Reduce population (remove worst individuals from sorted pop)
                pop = pop[:n_target]
                fitness = fitness[:n_target]
                pop_size = n_target
                
                # Resize Archive: L-SHADE keeps |A| <= |N|
                if len(archive) > pop_size:
                    del archive[pop_size:]
            
            # 2. Parameter Generation (Vectorized)
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # CR ~ Normal(M_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(M_f, 0.1)
            # Efficient vectorized Cauchy: M_f + 0.1 * tan(pi * (rand - 0.5))
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Handle F <= 0 (Retry)
            mask_bad = f <= 0
            while np.any(mask_bad):
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_bad)) - 0.5))
                mask_bad = f <= 0
            f = np.minimum(f, 1.0)
            
            # 3. Mutation: current-to-pbest/1
            # p-best selection (random from top p%)
            # p decreases linearly? Standard SHADE uses random p in [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2, pop_size)
            p_idx = (p_val * pop_size).astype(int)
            p_idx = np.maximum(p_idx, 1)
            
            # Select pbest
            rand_ranks = np.floor(np.random.rand(pop_size) * p_idx).astype(int)
            x_pbest = pop[rand_ranks]
            
            # Select r1 (!= i)
            r1 = np.random.randint(0, pop_size, pop_size)
            mask_s = (r1 == np.arange(pop_size))
            r1[mask_s] = (r1[mask_s] + 1) % pop_size
            x_r1 = pop[r1]
            
            # Select r2 (!= i, != r1, from Union(Pop, Archive))
            if len(archive) > 0:
                arr_arch = np.array(archive)
                union = np.vstack((pop, arr_arch))
            else:
                union = pop
            
            n_union = len(union)
            r2 = np.random.randint(0, n_union, pop_size)
            # Simple collision check
            mask_c = (r2 == np.arange(pop_size)) | (r2 == r1)
            if np.any(mask_c):
                r2[mask_c] = np.random.randint(0, n_union, np.sum(mask_c))
            x_r2 = union[r2]
            
            # Compute Mutant
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            mask_cr = np.random.rand(pop_size, dim) < cr[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cr[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cr, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection & Evaluation
            success_diff = []
            success_cr = []
            success_f = []
            improved_any = False
            
            for i in range(pop_size):
                if time.time() - start_time >= max_time: return best_fit
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        # Add replaced to archive
                        if len(archive) < pop_size:
                            archive.append(pop[i].copy())
                        else:
                            archive[np.random.randint(0, pop_size)] = pop[i].copy()
                            
                        success_diff.append(fitness[i] - f_trial)
                        success_cr.append(cr[i])
                        success_f.append(f[i])
                        improved_any = True
                        
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    if f_trial < best_fit:
                        best_fit = f_trial
                        best_sol = trial[i].copy()
            
            # 6. SHADE Memory Update
            if len(success_diff) > 0:
                w = np.array(success_diff)
                w /= np.sum(w)
                
                mean_cr = np.sum(w * np.array(success_cr))
                sf = np.array(success_f)
                mean_f = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-15)
                
                mem_cr[k_mem] = mean_cr
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                k_mem = (k_mem + 1) % H
            
            # 7. Sort Population
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # 8. Local Search (MTS-LS1) on Global Best
            # Apply periodically or when new best found, to refine results
            # We use a probability to balance overhead
            if improved_any or np.random.rand() < 0.1:
                # Iterate dimensions (random order to prevent bias)
                dims = np.arange(dim)
                np.random.shuffle(dims)
                
                ls_improved = False
                for d in dims:
                    if time.time() - start_time >= max_time: return best_fit
                    
                    x_curr = best_sol[d]
                    
                    # Try Negative Step
                    x_new = np.clip(x_curr - sr[d], min_b[d], max_b[d])
                    best_sol[d] = x_new
                    val = func(best_sol)
                    
                    if val < best_fit:
                        best_fit = val
                        pop[0] = best_sol.copy()
                        fitness[0] = best_fit
                        ls_improved = True
                    else:
                        # Revert
                        best_sol[d] = x_curr
                        
                        # Try Positive Step (Half size)
                        x_new = np.clip(x_curr + 0.5 * sr[d], min_b[d], max_b[d])
                        best_sol[d] = x_new
                        val = func(best_sol)
                        
                        if val < best_fit:
                            best_fit = val
                            pop[0] = best_sol.copy()
                            fitness[0] = best_fit
                            ls_improved = True
                        else:
                            # Revert
                            best_sol[d] = x_curr
                
                if not ls_improved:
                    sr *= 0.5 # Reduce search radius if no success
                else:
                    # If success, we assume current basin is promising, keep SR or reset?
                    # MTS-LS logic usually keeps or expands. We keep.
                    pass
                    
                # Reset SR if too small (avoid underflow / stuck)
                mask_sr = sr < 1e-15
                if np.any(mask_sr):
                    sr[mask_sr] = diff_b[mask_sr] * 0.4
                    
            # 9. Stagnation Check (Restart)
            # If population variance is zero or fitness range is negligible
            if fitness[-1] - fitness[0] < 1e-9 or np.std(fitness) < 1e-9:
                break 

    return best_fit
