#The following algorithm implements **O-IPO-jDE-MTS (Opposition-based Increasing Population Optimization with jDE and MTS-LS1 Local Search)**.
#
##### Rationale for Improvement
#1.  **MTS-LS1 (Refined Coordinate Descent)**: Previous algorithms struggled to refine the final solution. This implementation integrates a robust **MTS-LS1** (Multiple Trajectory Search - Local Search 1) logic. Unlike a random walk, MTS-LS1 systematically probes each dimension with an adaptive step size (`sr`), halving the step only when exploration in that dimension fails. This effectively descends into the local minimum's deepest point with high precision.
#2.  **Opposition-Based Learning (OBL)**: To ensure high-quality initial basins, the algorithm generates a random population and its mathematical opposite ($min + max - x$). It evaluates both sets and selects the fittest half, drastically increasing the probability of starting near the global optimum.
#3.  **Adaptive Population Sizing (IPO)**: The algorithm starts with a small population to quickly solve simple problems or find initial basins. Upon stagnation, it restarts with **double the population**, ensuring that if the first run gets stuck in a local optimum due to lack of diversity, subsequent runs have the capacity to explore complex multimodal landscapes.
#4.  **jDE with Archive**: It uses Self-Adaptive Differential Evolution (jDE) with `current-to-pbest/1` mutation and an external archive. This combination balances convergence speed (via pbest) and diversity (via archive and parameter self-adaptation).
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using O-IPO-jDE-MTS:
    Opposition-based Initialization, Increasing Population Optimization,
    Self-Adaptive DE, and MTS Local Search.
    """
    start_time = time.time()
    
    # --- Time Management ---
    # Return best solution if we are within 0.05s of the limit
    def check_time():
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
    # Start with a small population for efficiency
    # Scale up on restarts to handle complexity
    pop_size = max(20, int(3 * dim))
    MAX_POP_SIZE = 500
    
    # --- Main Restart Loop ---
    while True:
        if check_time(): return global_best_val
        
        # 1. Opposition-Based Initialization
        N = pop_size
        
        # A. Generate Random Population
        pop_rand = min_b + np.random.rand(N, dim) * diff_b
        
        # B. Generate Opposite Population
        # x_opp = min + max - x
        pop_opp = min_b + max_b - pop_rand
        
        # Fix bounds for opposite population (Randomize if out of bounds)
        # This preserves diversity better than clamping
        mask_out = (pop_opp < min_b) | (pop_opp > max_b)
        random_fix = min_b + np.random.rand(N, dim) * diff_b
        pop_opp = np.where(mask_out, random_fix, pop_opp)
        
        # C. Combine and Select Best N
        # We perform 2*N evaluations to pick the best starting set
        pop_pool = np.vstack((pop_rand, pop_opp))
        fit_pool = np.zeros(2 * N)
        
        # Elitism: Inject global best from previous runs
        if global_best_sol is not None:
            pop_pool[0] = global_best_sol
            
        for i in range(2 * N):
            if check_time(): return global_best_val
            val = func(pop_pool[i])
            fit_pool[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop_pool[i].copy()
                
        # Select best N
        sorted_idx = np.argsort(fit_pool)
        pop = pop_pool[sorted_idx[:N]]
        fitness = fit_pool[sorted_idx[:N]]
        
        # 2. jDE Setup
        # F ~ 0.5, CR ~ 0.9 initially
        F = np.full(N, 0.5)
        CR = np.full(N, 0.9)
        
        # Archive to store improved-upon solutions (maintains diversity)
        archive = np.zeros((N, dim))
        arc_count = 0
        
        # Stagnation counters
        last_best_fit = fitness[0]
        stagnation_count = 0
        
        # --- Evolution Cycle ---
        while True:
            if check_time(): return global_best_val
            
            # Sort population by fitness (needed for current-to-pbest)
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            F = F[sort_idx]
            CR = CR[sort_idx]
            
            # Update Global Best
            if fitness[0] < global_best_val:
                global_best_val = fitness[0]
                global_best_sol = pop[0].copy()
            
            # Stagnation Detection
            if abs(fitness[0] - last_best_fit) < 1e-12:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = fitness[0]
                
            # Exit Conditions for Restart:
            # 1. Population converged (fitness spread is tiny)
            # 2. No improvement for too many generations
            if (fitness[-1] - fitness[0] < 1e-8) or (stagnation_count > 30):
                break
                
            # jDE Parameter Adaptation
            # 10% chance to reset F or CR
            mask_f = np.random.rand(N) < 0.1
            mask_cr = np.random.rand(N) < 0.1
            
            F_new = F.copy()
            CR_new = CR.copy()
            # F ~ U(0.1, 1.0)
            F_new[mask_f] = 0.1 + 0.9 * np.random.rand(mask_f.sum())
            # CR ~ U(0.0, 1.0)
            CR_new[mask_cr] = np.random.rand(mask_cr.sum())
            
            # Mutation: current-to-pbest/1
            # Top p% selection (p between 2 individuals and 10% of pop)
            p_limit = max(2, int(N * 0.1))
            pbest_idx = np.random.randint(0, p_limit, N)
            x_pbest = pop[pbest_idx]
            
            # r1: Random from pop, distinct from current
            r1_idx = np.random.randint(0, N, N)
            r1_idx = np.where(r1_idx == np.arange(N), (r1_idx + 1) % N, r1_idx)
            x_r1 = pop[r1_idx]
            
            # r2: Random from Union(Pop, Archive)
            if arc_count > 0:
                pool = np.vstack((pop, archive[:arc_count]))
            else:
                pool = pop
            r2_idx = np.random.randint(0, len(pool), N)
            x_r2 = pool[r2_idx]
            
            # Mutant Vector Calculation
            # v = x + F*(pbest - x) + F*(r1 - r2)
            F_col = F_new[:, None]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            j_rand = np.random.randint(0, dim, N)
            mask_j = np.zeros((N, dim), dtype=bool)
            mask_j[np.arange(N), j_rand] = True
            
            cross_mask = (np.random.rand(N, dim) < CR_new[:, None]) | mask_j
            trial = np.where(cross_mask, mutant, pop)
            
            # Bound Handling: Weighted Midpoint
            # If out of bounds, place between parent and bound.
            # (parent + bound) / 2
            mask_l = trial < min_b
            mask_u = trial > max_b
            trial = np.where(mask_l, (pop + min_b) * 0.5, trial)
            trial = np.where(mask_u, (pop + max_b) * 0.5, trial)
            
            # Selection
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            parents_to_archive = []
            
            # Evaluate trials
            for i in range(N):
                if check_time(): return global_best_val
                
                f_trial = func(trial[i])
                
                # Global best update
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_sol = trial[i].copy()
                    
                # Greedy Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fitness[i] = f_trial
                    parents_to_archive.append(pop[i].copy())
                    F[i] = F_new[i]
                    CR[i] = CR_new[i]
            
            pop = new_pop
            fitness = new_fitness
            
            # Archive Maintenance
            if parents_to_archive:
                cands = np.array(parents_to_archive)
                num = len(cands)
                if arc_count + num <= N:
                    archive[arc_count:arc_count+num] = cands
                    arc_count += num
                else:
                    # Fill space
                    rem = N - arc_count
                    if rem > 0:
                        archive[arc_count:] = cands[:rem]
                        cands = cands[rem:]
                        arc_count = N
                    # Random replacement
                    if len(cands) > 0:
                        idx = np.random.randint(0, N, len(cands))
                        archive[idx] = cands

        # --- 3. MTS Local Search (MTS-LS1) ---
        # Polishing the best solution found so far using Coordinate Descent.
        if global_best_sol is not None:
            best_sol = global_best_sol.copy()
            best_val = global_best_val
            
            # Search Range (sr) initialization
            # Start wide (40% of domain) and shrink individually per dimension
            sr = (max_b - min_b) * 0.4
            
            # Run for a limited number of passes or until precision limit
            for _ in range(15):
                if check_time(): return global_best_val
                if np.max(sr) < 1e-13: break
                
                # Iterate over dimensions
                for d in range(dim):
                    if check_time(): return global_best_val
                    
                    original_x = best_sol[d]
                    
                    # Try Negative Step
                    best_sol[d] = max(min_b[d], original_x - sr[d])
                    val = func(best_sol)
                    
                    if val < best_val:
                        best_val = val
                        global_best_val = val
                        global_best_sol = best_sol.copy()
                    else:
                        # Restore
                        best_sol[d] = original_x
                        
                        # Try Positive Step (0.5 * sr)
                        best_sol[d] = min(max_b[d], original_x + 0.5 * sr[d])
                        val = func(best_sol)
                        
                        if val < best_val:
                            best_val = val
                            global_best_val = val
                            global_best_sol = best_sol.copy()
                        else:
                            # Restore and Shrink Range
                            best_sol[d] = original_x
                            sr[d] *= 0.5
        
        # --- 4. Prepare for Restart (IPO) ---
        # Double population size for next run
        pop_size *= 2
        if pop_size > MAX_POP_SIZE:
            pop_size = MAX_POP_SIZE

    return global_best_val
