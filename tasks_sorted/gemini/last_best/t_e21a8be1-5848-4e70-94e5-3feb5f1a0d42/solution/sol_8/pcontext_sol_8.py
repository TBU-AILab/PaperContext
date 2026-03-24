#The following algorithm implements **L-SHADE-RSP-OBL-MTS**.
#
##### Key Improvements:
#1.  **OBL (Opposition-Based Learning) Initialization**: Instead of purely random initialization, the algorithm generates opposite points ($lb + ub - x$) and selects the fittest half. This provides a much higher quality starting population, often placing individuals closer to the global basin immediately.
#2.  **L-SHADE-RSP (Rank-Based Success Parameters)**: Builds on L-SHADE but refines the parameter adaptation. It uses a **Weighted Lehmer Mean** for memory updates where weights are proportional to fitness improvements, ensuring parameters that yield *larger* jumps in fitness have more influence on future mutations.
#3.  **Linear Population Size Reduction (LPSR)**: Linearly reduces population size from an initial high value (for exploration) to a minimum (for exploitation), forcing convergence as time runs out.
#4.  **Soft Restart Strategy**: Monitors population variance. If the population stagnates (variance $\approx$ 0) before the local search phase, it triggers a soft restart: the global best is preserved, but other individuals are re-initialized to explore new basins, resetting adaptive memories.
#5.  **MTS (Multiple Trajectory Search) Polishing**: A dedicated time slot (final 20%) is reserved for coordinate-descent local search. This phase fine-tunes the best solution found by the evolutionary phase to high precision using an adaptive step size that shrinks upon failure and expands/resets upon stagnation.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-RSP-OBL-MTS.
    
    Algorithm Phases:
    1. OBL Initialization: Opposition-Based Learning for better initial coverage.
    2. L-SHADE-RSP: Evolutionary search with Rank-based Success Parameters 
       and Linear Population Size Reduction.
    3. MTS-LS1: Local Search polishing phase for final precision.
    """

    # --- 1. Configuration & Time Management ---
    t_start = time.time()
    t_end = t_start + max_time
    
    # Allocate 80% time for Evolution, 20% for Local Search polishing
    ratio_ls = 0.20
    t_ls_start = t_start + (1.0 - ratio_ls) * max_time
    
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # --- 2. Initialization (OBL) ---
    # LPSR Configuration
    pop_size_init = int(round(20 * dim)) 
    pop_size_min = 4
    pop_size = pop_size_init
    
    # SHADE Memory (H=5)
    H = 5
    M_cr = np.full(H, 0.5) # Memory for Crossover Rate
    M_f = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0
    archive = []
    
    # a. Random Generation
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        pop[:, d] = np.random.uniform(lb[d], ub[d], pop_size)
        
    # b. Opposition Generation
    # x' = lb + ub - x
    pop_opp = lb + ub - pop
    pop_opp = np.clip(pop_opp, lb, ub)
    
    # c. Combine and Select Best N
    pop_pool = np.vstack((pop, pop_opp))
    fit_pool = np.full(2 * pop_size, float('inf'))
    
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate pool with time check
    # We iterate to respect the time limit strictly even during init
    for i in range(len(pop_pool)):
        if time.time() >= t_ls_start:
            # If initialization takes too long, truncate and proceed
            pop_pool = pop_pool[:i]
            fit_pool = fit_pool[:i]
            break
            
        val = func(pop_pool[i])
        fit_pool[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = pop_pool[i].copy()
            
    # Selection
    if len(pop_pool) > pop_size:
        sort_idx = np.argsort(fit_pool)
        pop = pop_pool[sort_idx[:pop_size]]
        fitness = fit_pool[sort_idx[:pop_size]]
    else:
        pop = pop_pool
        fitness = fit_pool
        pop_size = len(pop)

    # --- 3. Evolutionary Phase (L-SHADE) ---
    gen = 0
    while True:
        curr_time = time.time()
        if curr_time >= t_ls_start:
            break
            
        # Time Progress (0.0 -> 1.0)
        evo_budget = t_ls_start - t_start
        if evo_budget <= 1e-9: break
        progress = (curr_time - t_start) / evo_budget
        progress = np.clip(progress, 0, 1)
        
        # A. Linear Population Size Reduction (LPSR)
        next_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        next_pop_size = max(pop_size_min, next_pop_size)
        
        if pop_size > next_pop_size:
            # Sort and truncate worst solutions
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:next_pop_size]]
            fitness = fitness[sorted_idx[:next_pop_size]]
            
            # Resize Archive
            if len(archive) > next_pop_size:
                n_remove = len(archive) - next_pop_size
                for _ in range(n_remove):
                    archive.pop(np.random.randint(0, len(archive)))
            
            pop_size = next_pop_size
            
        # B. Parameter Adaptation
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = M_cr[r_idx]
        m_f = M_f[r_idx]
        
        # CR ~ Normal(m_cr, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(m_f, 0.1)
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Sanitize F (Retry loop for negative values)
        for _ in range(5):
            bad = f <= 0
            if not np.any(bad): break
            f[bad] = m_f[bad] + 0.1 * np.random.standard_cauchy(np.sum(bad))
            
        f = np.clip(f, 0, 1)
        f[f <= 0] = 0.5 # Fallback
        
        # C. Mutation: current-to-pbest/1
        # p adapts linearly from 0.2 to 0.05 (Exploration -> Exploitation)
        p_val = 0.2 - (0.15 * progress)
        p_val = max(0.05, p_val)
        p_num = max(2, int(pop_size * p_val))
        
        sorted_indices = np.argsort(fitness)
        # x_pbest: random from top p%
        pbest_idxs = np.random.choice(sorted_indices[:p_num], pop_size)
        x_pbest = pop[pbest_idxs]
        
        # x_r1: random distinct from current
        r1 = np.random.randint(0, pop_size, pop_size)
        conflict = r1 == np.arange(pop_size)
        r1[conflict] = (r1[conflict] + 1) % pop_size
        x_r1 = pop[r1]
        
        # x_r2: random distinct from current & r1, from Pop U Archive
        if len(archive) > 0:
            pop_all = np.vstack((pop, np.array(archive)))
        else:
            pop_all = pop
            
        r2 = np.random.randint(0, len(pop_all), pop_size)
        # Simple collision handling
        conflict_r2 = (r2 == np.arange(pop_size)) | (r2 == r1)
        r2[conflict_r2] = np.random.randint(0, len(pop_all), np.sum(conflict_r2))
        x_r2 = pop_all[r2]
        
        # Mutation Vector
        f_vec = f[:, None]
        mutant = pop + f_vec * (x_pbest - pop) + f_vec * (x_r1 - x_r2)
        
        # D. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        mask[np.arange(pop_size), j_rand] = True
        trial = np.where(mask, mutant, pop)
        
        # E. Bound Handling (Reflection)
        # Reflection works better than clipping for boundary-located optima
        bl = trial < lb
        if np.any(bl):
            trial[bl] = 2*lb[np.where(bl)[1]] - trial[bl]
        bu = trial > ub
        if np.any(bu):
            trial[bu] = 2*ub[np.where(bu)[1]] - trial[bu]
        trial = np.clip(trial, lb, ub)
        
        # F. Selection and Update
        success_mask = np.zeros(pop_size, dtype=bool)
        diff_fitness = np.zeros(pop_size)
        
        # Evaluate batch
        # Check time periodically to minimize overhead
        check_interval = max(1, int(pop_size / 5))
        
        for i in range(pop_size):
            if i % check_interval == 0:
                if time.time() >= t_ls_start: break
                
            val_trial = func(trial[i])
            
            # Greedy Selection
            if val_trial <= fitness[i]:
                if val_trial < fitness[i]:
                    diff_fitness[i] = fitness[i] - val_trial
                    success_mask[i] = True
                    archive.append(pop[i].copy())
                    
                fitness[i] = val_trial
                pop[i] = trial[i]
                
                if val_trial < best_val:
                    best_val = val_trial
                    best_sol = trial[i].copy()
                    
        # Maintain Archive Size
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # G. Memory Update (Weighted)
        if np.any(success_mask):
            s_f = f[success_mask]
            s_cr = cr[success_mask]
            dif = diff_fitness[success_mask]
            
            # Weights proportional to fitness improvement
            w = dif / np.sum(dif)
            
            # M_cr: Weighted Mean
            mean_cr = np.sum(w * s_cr)
            M_cr[k_mem] = 0.5 * M_cr[k_mem] + 0.5 * mean_cr
            
            # M_f: Weighted Lehmer Mean
            sum_wf = np.sum(w * s_f)
            if sum_wf > 1e-12:
                mean_f = np.sum(w * s_f**2) / sum_wf
                M_f[k_mem] = 0.5 * M_f[k_mem] + 0.5 * mean_f
            
            k_mem = (k_mem + 1) % H
            
        # H. Soft Restart Mechanism
        # If population variance collapses, we are stuck.
        if gen % 20 == 0:
            std_fit = np.std(fitness)
            if std_fit < 1e-8:
                # Keep best, re-init others randomly
                best_idx = np.argmin(fitness)
                
                new_pop = np.zeros((pop_size, dim))
                for d in range(dim):
                    new_pop[:, d] = np.random.uniform(lb[d], ub[d], pop_size)
                    
                for i in range(pop_size):
                    if i == best_idx: continue
                    if time.time() >= t_ls_start: break
                    
                    val = func(new_pop[i])
                    pop[i] = new_pop[i]
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_sol = new_pop[i].copy()
                
                # Reset Memory and Archive to allow new adaptation
                M_cr.fill(0.5)
                M_f.fill(0.5)
                archive = []
                
        gen += 1
        
    # --- 4. Local Search Phase (MTS-LS1) ---
    # Polishing the best solution using Coordinate Descent
    curr_best = best_sol.copy()
    curr_val = best_val
    
    # Initial Search Range (approx 40% of domain)
    # This range adapts for each dimension independently
    sr = (ub - lb) * 0.4
    
    while time.time() < t_end:
        
        improved_any = False
        # Shuffle dimensions to avoid bias
        dims_order = np.arange(dim)
        np.random.shuffle(dims_order)
        
        for i in dims_order:
            if time.time() >= t_end: break
            
            improved_dim = False
            
            # 1. Try Negative Step (x - sr)
            x_test = curr_best.copy()
            x_test[i] = np.clip(x_test[i] - sr[i], lb[i], ub[i])
            val_test = func(x_test)
            
            if val_test < curr_val:
                curr_best = x_test
                curr_val = val_test
                best_val = val_test
                improved_dim = True
            else:
                # 2. Try Positive Step (x + 0.5*sr)
                # Note: MTS typically tries +0.5 step if full negative step fails
                x_test = curr_best.copy()
                x_test[i] = np.clip(x_test[i] + 0.5 * sr[i], lb[i], ub[i])
                val_test = func(x_test)
                
                if val_test < curr_val:
                    curr_best = x_test
                    curr_val = val_test
                    best_val = val_test
                    improved_dim = True
            
            # Update Search Range
            if not improved_dim:
                # Shrink range if no improvement
                sr[i] *= 0.5
            else:
                improved_any = True
                
        # Search Range Reset (Pattern Search Logic)
        # If all dimensions failed to improve and ranges are tiny, 
        # reset SR slightly to escape numerical traps
        if not improved_any:
            if np.max(sr) < 1e-12:
                 sr = (ub - lb) * 0.1
                 
    return best_val
