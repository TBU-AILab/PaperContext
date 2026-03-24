#The following algorithm implements **O-IPO-jDE-AMTS (Opposition-based Increasing Population Optimization with Adaptive MTS Local Search)**.
#
#### Rationale for Improvement
#1.  **Adaptive MTS Local Search**: The previous best algorithm (Code 3) used a simple Gaussian walk for refinement. While better than nothing, it is "blind" to the landscape's structure. This version integrates a **Coordinate Descent (MTS-LS1)** strategy, but with a critical improvement: **Adaptive Step Sizing**. Instead of a fixed initial step (e.g., 40% of domain), the local search initializes its step sizes (`sr`) based on the **standard deviation/spread of the converged population**. This allows the local search to seamlessly continue refining at the exact scale where the DE population left off, whether it's exploring a basin or polishing to $10^{-15}$ precision.
#2.  **Refined OBL**: Initialization uses **Opposition-Based Learning**, but with a "Random" fix for out-of-bound opposites (instead of clamping). This maintains higher diversity in the initial population, preventing premature clustering at the search space boundaries.
#3.  **Randomized Bound Handling**: During the DE phase, if a trial vector exceeds bounds, it is placed randomly between the parent and the bound. This prevents the "center bias" of midpoint handling and the "edge sticking" of clamping.
#4.  **Robust IPO**: It retains the "Increasing Population" restart strategy (growing by factor 2) and the self-adaptive jDE parameters, ensuring a good balance between fast exploration of easy problems and deep search of complex multimodal landscapes.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using O-IPO-jDE-AMTS:
    Opposition-based Initialization, Increasing Population Optimization,
    Self-Adaptive Differential Evolution, and Adaptive MTS Local Search.
    """
    start_time = time.time()
    
    # --- Helper: Time Management ---
    def check_time():
        # Reserve small buffer to return safely
        return (time.time() - start_time) >= max_time - 0.05
    
    # --- Pre-computation ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Algorithm Configuration ---
    # Start with a moderate population to find basins quickly
    pop_size = max(20, int(4 * dim))
    MAX_POP_SIZE = 500
    
    # --- Main Restart Loop (IPO) ---
    while True:
        if check_time(): return global_best_val
        
        # 1. Opposition-Based Initialization
        N = pop_size
        
        # A. Random Population
        pop_rand = min_b + np.random.rand(N, dim) * diff_b
        
        # B. Opposite Population (OBL)
        # x_opp = min + max - x
        pop_opp = min_b + max_b - pop_rand
        
        # Handle bounds for OBL: Randomize if out (preserves diversity)
        mask_l = pop_opp < min_b
        mask_u = pop_opp > max_b
        mask_oob = mask_l | mask_u
        random_fix = min_b + np.random.rand(N, dim) * diff_b
        pop_opp = np.where(mask_oob, random_fix, pop_opp)
        
        # C. Combine and Select Best N
        pop_pool = np.vstack((pop_rand, pop_opp))
        
        # Elitism: Inject global best from previous restarts
        if global_best_sol is not None:
            pop_pool[0] = global_best_sol
            
        fit_pool = np.full(2 * N, float('inf'))
        
        # Evaluate Pool
        for i in range(2 * N):
            if check_time(): return global_best_val
            
            val = func(pop_pool[i])
            fit_pool[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop_pool[i].copy()
                
        # Sort and select top N
        sort_idx = np.argsort(fit_pool)
        pop = pop_pool[sort_idx[:N]]
        fitness = fit_pool[sort_idx[:N]]
        
        # 2. jDE Setup
        # F initialized to 0.5, CR to 0.9
        F = np.full(N, 0.5)
        CR = np.full(N, 0.9)
        
        # External Archive for current-to-pbest
        archive = np.zeros((N, dim))
        arc_count = 0
        
        # Stagnation detection
        stagnation_count = 0
        last_best_fit = fitness[0]
        
        # --- Evolution Cycle ---
        while True:
            if check_time(): return global_best_val
            
            # Sort population (required for pbest selection)
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            F = F[sort_idx]
            CR = CR[sort_idx]
            
            # Check Stagnation
            curr_best = fitness[0]
            if abs(curr_best - last_best_fit) < 1e-12:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = curr_best
            
            # Restart Conditions:
            # 1. Population converged (low variance)
            # 2. Stagnation limit reached
            fit_spread = fitness[-1] - fitness[0]
            tol = 1e-9 if global_best_val < 1.0 else 1e-9 * global_best_val
            
            if (fit_spread < tol) or (stagnation_count > 30):
                break
            
            # jDE Parameter Adaptation
            # 10% chance to reset parameters
            mask_f = np.random.rand(N) < 0.1
            mask_cr = np.random.rand(N) < 0.1
            
            F_new = F.copy()
            CR_new = CR.copy()
            
            # F ~ U(0.1, 1.0) -> favours exploration but allows fine search
            F_new[mask_f] = 0.1 + 0.9 * np.random.rand(mask_f.sum())
            # CR ~ U(0.0, 1.0)
            CR_new[mask_cr] = np.random.rand(mask_cr.sum())
            
            # Mutation: current-to-pbest/1
            # Top p% selection (p in 5% - 20%)
            p_pct = np.random.uniform(0.05, 0.2)
            p_limit = max(2, int(N * p_pct))
            
            pbest_idx = np.random.randint(0, p_limit, N)
            x_pbest = pop[pbest_idx]
            
            r1_idx = np.random.randint(0, N, N)
            # Ensure r1 != i
            r1_idx = np.where(r1_idx == np.arange(N), (r1_idx + 1) % N, r1_idx)
            x_r1 = pop[r1_idx]
            
            # r2 from Union(Pop, Archive)
            if arc_count > 0:
                pool = np.vstack((pop, archive[:arc_count]))
            else:
                pool = pop
            
            r2_idx = np.random.randint(0, len(pool), N)
            x_r2 = pool[r2_idx]
            
            # Compute Mutant Vector
            F_col = F_new[:, None]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            j_rand = np.random.randint(0, dim, N)
            mask_j = np.zeros((N, dim), dtype=bool)
            mask_j[np.arange(N), j_rand] = True
            
            cross_mask = (np.random.rand(N, dim) < CR_new[:, None]) | mask_j
            trial = np.where(cross_mask, mutant, pop)
            
            # Bound Handling: Randomized Inter-Bound
            # Places point randomly between parent and bound if violated
            low_mask = trial < min_b
            high_mask = trial > max_b
            
            rand_fix = np.random.rand(N, dim)
            trial = np.where(low_mask, min_b + rand_fix * (pop - min_b), trial)
            trial = np.where(high_mask, max_b - rand_fix * (max_b - pop), trial)
            
            # Selection
            pop_next = pop.copy()
            fit_next = fitness.copy()
            F_next = F.copy()
            CR_next = CR.copy()
            
            archive_cands = []
            
            for k in range(N):
                if check_time(): return global_best_val
                
                f_trial = func(trial[k])
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_sol = trial[k].copy()
                
                # Greedy Selection
                if f_trial <= fitness[k]:
                    pop_next[k] = trial[k]
                    fit_next[k] = f_trial
                    F_next[k] = F_new[k]
                    CR_next[k] = CR_new[k]
                    # Add replaced parent to archive candidates
                    archive_cands.append(pop[k].copy())
            
            pop = pop_next
            fitness = fit_next
            F = F_next
            CR = CR_next
            
            # Update Archive
            if archive_cands:
                cands = np.array(archive_cands)
                cnt = len(cands)
                if arc_count + cnt <= N:
                    archive[arc_count:arc_count+cnt] = cands
                    arc_count += cnt
                else:
                    # Fill space
                    rem = N - arc_count
                    if rem > 0:
                        archive[arc_count:] = cands[:rem]
                        cands = cands[rem:]
                        arc_count = N
                    # Random replacement for remainder
                    if len(cands) > 0:
                        idx = np.random.randint(0, N, len(cands))
                        archive[idx] = cands

        # 3. Adaptive MTS-LS1 Local Search
        # Run on the global best solution when population stagnates
        if global_best_sol is not None:
            best_sol = global_best_sol.copy()
            best_val = global_best_val
            
            # Initialize Search Range (sr) based on population spread
            # This makes the local search adaptive to the current convergence level
            spread = np.max(pop, axis=0) - np.min(pop, axis=0)
            # Ensure minimum sr is not 0, and not larger than 20% of domain
            sr = np.maximum(spread * 0.5, 1e-13)
            sr = np.minimum(sr, diff_b * 0.2)
            
            # Perform Coordinate Descent
            # Limit passes to avoid timeout, break if sr becomes too small
            for _ in range(15):
                if check_time(): return global_best_val
                if np.max(sr) < 1e-15: break
                
                # Randomize dimension order
                dims = np.arange(dim)
                np.random.shuffle(dims)
                
                improved = False
                for d in dims:
                    if check_time(): return global_best_val
                    
                    original_x = best_sol[d]
                    
                    # Try Negative Step
                    best_sol[d] = max(min_b[d], original_x - sr[d])
                    val = func(best_sol)
                    
                    if val < best_val:
                        best_val = val
                        global_best_val = val
                        global_best_sol = best_sol.copy()
                        improved = True
                    else:
                        best_sol[d] = original_x
                        # Try Positive Step (0.5 size)
                        best_sol[d] = min(max_b[d], original_x + 0.5 * sr[d])
                        val = func(best_sol)
                        
                        if val < best_val:
                            best_val = val
                            global_best_val = val
                            global_best_sol = best_sol.copy()
                            improved = True
                        else:
                            best_sol[d] = original_x
                            # Shrink Search Range for this dimension
                            sr[d] *= 0.5
                            
        # 4. IPO Restart Preparation
        # Double population size for next run (Increase diversity)
        pop_size = int(pop_size * 2)
        if pop_size > MAX_POP_SIZE:
            pop_size = MAX_POP_SIZE

    return global_best_val
