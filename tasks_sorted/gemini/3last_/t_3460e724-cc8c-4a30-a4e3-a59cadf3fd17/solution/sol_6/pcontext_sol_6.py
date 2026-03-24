#The following algorithm implements **O-IPO-jDE (Opposition-based Increasing Population Optimization with Self-Adaptive Differential Evolution)**.
#
#**Rationale for Improvement:**
#1.  **Opposition-Based Learning (OBL) Initialization**: The previous best algorithm (IPO-jDE) used random initialization. This improved version generates both random solutions and their "opposite" solutions ($x_{opp} = min + max - x$) during the initialization of each restart. It selects the fittest half from this combined pool. This allows the algorithm to cover the search space twice as effectively initially, often landing immediately in better basins of attraction.
#2.  **Local Search Polish**: Before abandoning a converged population to restart, the algorithm now performs a high-intensity "polishing" step on the global best solution using a Gaussian walk with shrinking variance. This helps squeeze out the final decimal points of precision that pure DE might struggle to resolve due to crossover limitations.
#3.  **Retained Strengths**: It keeps the successful "Increasing Population" strategy (starting fast/small, ending slow/massive) and the robust jDE parameter adaptation from the best-performing previous submission.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using O-IPO-jDE:
    Opposition-based Increasing Population Optimization with Self-Adaptive DE.
    Includes Local Search polishing before restarts.
    """
    start_time = time.time()
    
    # --- Helper: Time Check ---
    # Returns True if we have exceeded max_time (with a small buffer)
    def is_time_up():
        return (time.time() - start_time) >= max_time
    
    # --- Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global State
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Configuration ---
    # Start small to solve simple problems fast. 
    # Scale up on restarts to handle complex multimodal landscapes.
    pop_size = max(20, int(5 * dim))
    MAX_POP_SIZE = 500  # Cap to ensure generations don't become too slow
    
    # --- Main Restart Loop ---
    while True:
        # Check time before starting a new optimization cycle
        if (time.time() - start_time) > max_time - 0.05:
            return global_best_val
            
        # 1. Initialization Phase (Opposition-Based)
        N = pop_size
        
        # A. Generate Random Population
        pop_rand = min_b + np.random.rand(N, dim) * diff_b
        
        # B. Generate Opposite Population
        # X_opp = Min + Max - X_rand
        # This checks the "mirror" side of the search space
        pop_opp = min_b + max_b - pop_rand
        
        # Merge and handle bounds for OPP (random is already bounded)
        # Note: We combine them into a pool of 2*N candidates
        pop_pool = np.vstack((pop_rand, pop_opp))
        
        # Clip strictly to bounds
        pop_pool = np.maximum(pop_pool, min_b)
        pop_pool = np.minimum(pop_pool, max_b)
        
        # Inject Elitism: If we have a global best from previous runs, 
        # replace the first candidate with it.
        start_idx = 0
        if global_best_sol is not None:
            pop_pool[0] = global_best_sol
            # We already know its fitness, but for simplicity of vectorization,
            # we might re-eval or cache. Let's cache.
            # However, simpler to just eval unless it's index 0.
            pass

        # Evaluate Pool
        pool_fitness = np.full(len(pop_pool), float('inf'))
        
        # We evaluate candidates. 
        # Optimization: if N is large, OBL might be expensive. 
        # But for the first gen, it's worth it.
        for i in range(len(pop_pool)):
            if is_time_up(): return global_best_val
            
            # Optimization: Skip re-eval of injected global best at index 0
            if i == 0 and global_best_sol is not None:
                val = global_best_val
            else:
                val = func(pop_pool[i])
            
            pool_fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop_pool[i].copy()
                
        # Select best N individuals to form the actual population
        sorted_pool_idx = np.argsort(pool_fitness)
        pop = pop_pool[sorted_pool_idx[:N]]
        fitness = pool_fitness[sorted_pool_idx[:N]]
        
        # --- jDE Parameter Initialization ---
        # F ~ U(0.1, 1.0), CR ~ U(0.0, 1.0)
        F = 0.1 + 0.9 * np.random.rand(N)
        CR = np.random.rand(N)
        
        # Archive for mutation strategy (stores replaced individuals)
        archive = np.zeros((int(N * 2), dim))
        arc_count = 0
        
        # --- Evolution Loop ---
        # Run until convergence or time limit
        while True:
            if is_time_up(): return global_best_val
            
            # Sort population by fitness (needed for current-to-pbest)
            # Sorting also helps in detecting convergence
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            F = F[sort_idx]
            CR = CR[sort_idx]
            
            # Convergence Check
            # If the difference between best and median/worst is tiny, we are stuck.
            fit_range = fitness[-1] - fitness[0]
            if fit_range < 1e-8:
                break
                
            # --- jDE Parameter Adaptation ---
            # Randomly reset F and CR with small probabilities
            tau1, tau2 = 0.1, 0.1
            
            rand_f = np.random.rand(N)
            F_new = np.where(rand_f < tau1, 0.1 + 0.9 * np.random.rand(N), F)
            
            rand_cr = np.random.rand(N)
            CR_new = np.where(rand_cr < tau2, np.random.rand(N), CR)
            
            # --- Mutation: current-to-pbest/1 ---
            # V = X + F(Xpbest - X) + F(Xr1 - Xr2)
            
            # p-best selection: top 5% to 20% (randomized per gen for diversity)
            p_val = np.random.uniform(0.05, 0.2)
            p_num = max(2, int(N * p_val))
            
            pbest_indices = np.random.randint(0, p_num, N)
            x_pbest = pop[pbest_indices]
            
            # r1 selection: random from pop, != current
            r1_indices = np.random.randint(0, N, N)
            # Shift collision
            r1_indices = np.where(r1_indices == np.arange(N), (r1_indices + 1) % N, r1_indices)
            x_r1 = pop[r1_indices]
            
            # r2 selection: random from Pop U Archive
            if arc_count > 0:
                pool = np.vstack((pop, archive[:arc_count]))
            else:
                pool = pop
            
            pool_size = len(pool)
            r2_indices = np.random.randint(0, pool_size, N)
            x_r2 = pool[r2_indices]
            
            # Compute Mutant
            # Reshape F for broadcasting
            F_col = F_new[:, None]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_mat = np.random.rand(N, dim)
            j_rand = np.random.randint(0, dim, N)
            j_mask = np.zeros((N, dim), dtype=bool)
            j_mask[np.arange(N), j_rand] = True
            
            cross_mask = (rand_mat < CR_new[:, None]) | j_mask
            trial = np.where(cross_mask, mutant, pop)
            
            # --- Bound Handling ---
            # Midpoint fallback: if out of bounds, place between parent and bound
            lower_viol = trial < min_b
            upper_viol = trial > max_b
            
            trial = np.where(lower_viol, (pop + min_b) * 0.5, trial)
            trial = np.where(upper_viol, (pop + max_b) * 0.5, trial)
            
            # --- Evaluation and Selection ---
            # Prepared next gen arrays
            next_pop = pop.copy()
            next_fitness = fitness.copy()
            next_F = F.copy()
            next_CR = CR.copy()
            
            archive_candidates = []
            
            for i in range(N):
                if is_time_up(): return global_best_val
                
                f_trial = func(trial[i])
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_sol = trial[i].copy()
                
                # Selection
                if f_trial <= fitness[i]:
                    # Improvement
                    archive_candidates.append(pop[i].copy())
                    next_pop[i] = trial[i]
                    next_fitness[i] = f_trial
                    next_F[i] = F_new[i]
                    next_CR[i] = CR_new[i]
            
            # Update Population
            pop = next_pop
            fitness = next_fitness
            F = next_F
            CR = next_CR
            
            # Update Archive
            if len(archive_candidates) > 0:
                cands = np.array(archive_candidates)
                num_cands = len(cands)
                capacity = len(archive)
                
                if arc_count + num_cands <= capacity:
                    archive[arc_count:arc_count+num_cands] = cands
                    arc_count += num_cands
                else:
                    # Fill remainder
                    space = capacity - arc_count
                    if space > 0:
                        archive[arc_count:] = cands[:space]
                        cands = cands[space:]
                        arc_count = capacity
                    
                    # Random replacement
                    if len(cands) > 0:
                        idx = np.random.randint(0, capacity, len(cands))
                        archive[idx] = cands

        # --- End of Evolution (Restart Triggered) ---
        
        # --- Local Search Polish ---
        # Before restarting (discarding population), try to polish the best solution
        # using a Gaussian walk. This helps if DE got "close enough" but lacks precision.
        if global_best_sol is not None:
            # 50 iterations of polishing
            ls_iter = 50
            # Sigma decreases from 1% of domain to 0.001%
            current_sol = global_best_sol.copy()
            current_val = global_best_val
            
            for k in range(ls_iter):
                if is_time_up(): return global_best_val
                
                # Decay scale
                scale = 0.01 * (1.0 - k/ls_iter)
                step = np.random.normal(0, scale, dim) * diff_b
                
                cand = current_sol + step
                # Clip
                cand = np.maximum(cand, min_b)
                cand = np.minimum(cand, max_b)
                
                val = func(cand)
                
                if val < current_val:
                    current_val = val
                    current_sol = cand
                    if val < global_best_val:
                        global_best_val = val
                        global_best_sol = cand.copy()
        
        # --- Prepare for Next Restart ---
        # Increase population size to explore more thoroughly (IPO strategy)
        # Factor 1.5 instead of 2 to allow more restarts with intermediate sizes
        pop_size = int(pop_size * 1.5)
        if pop_size > MAX_POP_SIZE:
            pop_size = MAX_POP_SIZE

    return global_best_val
