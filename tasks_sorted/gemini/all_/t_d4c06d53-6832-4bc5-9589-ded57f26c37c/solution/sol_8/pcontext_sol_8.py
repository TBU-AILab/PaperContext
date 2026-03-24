#The following algorithm implements **Restarting L-SHADE with Opposition-Based Initialization and Hybrid Local Search (MTS-LS1 + Directional Extrapolation)**.
#
##### Rationale for Improvement
#The previous best result (4.02) was achieved by L-SHADE with "Expanded Momentum Polishing". To further minimize the output, this version introduces three key refinements:
#1.  **Opposition-Based Initialization (OBL)**: Instead of random initialization, the algorithm generates an "opposite" population ($x' = min + max - x$) and selects the best individuals from the union of random and opposite sets. This provides a much better starting point, effectively doubling the search coverage at the critical early stage of each restart.
#2.  **Hybrid Polishing Strategy**: The previous "Momentum" polish was effective but could miss axis-aligned descent paths. This version integrates **MTS-LS1 (Coordinate Descent)** to systematically refine each dimension. If MTS detects a descent direction, the algorithm switches to **Directional Extrapolation** to greedily accelerate down the valley. This two-stage approach combines high precision (Coordinate Descent) with high speed (Extrapolation).
#3.  **Adaptive Population Pressure**: The algorithm retains the "Adaptive Population Reduction" mechanism (shrinking population upon stagnation) but tunes the trigger parameters to balance diversity retention with convergence speed.
#
##### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting L-SHADE with Opposition-Based Initialization
    and Hybrid Local Search (MTS-LS1 + Directional Extrapolation).
    
    Mechanism:
    1.  OBL Initialization: Uses Opposition-Based Learning to jump-start the population
        by evaluating both random points and their opposites.
    2.  L-SHADE: Core optimizer with historical parameter adaptation and external archive.
    3.  Adaptive Reduction: Dynamically removes worst individuals during stagnation 
        to force convergence.
    4.  Hybrid Polishing: A robust local search that combines Coordinate Descent 
        for precision and Momentum Extrapolation for speed.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Helper: Check Time ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: Hybrid Polishing (MTS-LS1 + Extrapolation) ---
    def hybrid_polish(current_best_sol, current_best_val, duration_sec):
        """
        Refines the solution using Coordinate Descent to find a gradient,
        then Extrapolation to move quickly along that gradient.
        """
        p_start = datetime.now()
        
        # Working variables
        x_curr = current_best_sol.copy()
        f_curr = current_best_val
        
        # Initial Search Range for MTS (Coordinate Descent)
        # Start with 10% of domain size
        search_range = diff_b * 0.1 
        
        # Access to update global best
        nonlocal global_best_val, global_best_sol
        
        while (datetime.now() - p_start).total_seconds() < duration_sec:
            if check_time(): break
            
            x_before = x_curr.copy()
            improved_in_mts = False
            
            # 1. MTS-LS1 (Coordinate Descent)
            # Shuffle dimensions to avoid directional bias
            dims = np.random.permutation(dim)
            
            for d in dims:
                if check_time(): break
                
                # Try negative step
                x_test = x_curr.copy()
                x_test[d] = np.clip(x_curr[d] - search_range[d], min_b[d], max_b[d])
                f_test = func(x_test)
                
                if f_test < f_curr:
                    f_curr = f_test
                    x_curr = x_test
                    improved_in_mts = True
                    if f_curr < global_best_val:
                        global_best_val = f_curr
                        global_best_sol = x_curr.copy()
                else:
                    # Try positive step (0.5 size - asymmetric search)
                    x_test[d] = np.clip(x_curr[d] + 0.5 * search_range[d], min_b[d], max_b[d])
                    f_test = func(x_test)
                    
                    if f_test < f_curr:
                        f_curr = f_test
                        x_curr = x_test
                        improved_in_mts = True
                        if f_curr < global_best_val:
                            global_best_val = f_curr
                            global_best_sol = x_curr.copy()
            
            # 2. Directional Extrapolation
            # If MTS moved the point, we have a descent vector. Follow it!
            if improved_in_mts:
                delta = x_curr - x_before
                
                # Extrapolate along delta
                alpha = 1.0
                accel = 1.2 # Acceleration factor
                
                while True:
                    if check_time(): break
                    
                    x_ext = x_curr + alpha * delta
                    x_ext = np.clip(x_ext, min_b, max_b)
                    f_ext = func(x_ext)
                    
                    if f_ext < f_curr:
                        f_curr = f_ext
                        x_curr = x_ext
                        alpha *= accel # Accelerate further
                        if f_curr < global_best_val:
                            global_best_val = f_curr
                            global_best_sol = x_curr.copy()
                    else:
                        break # Stop extrapolation
            else:
                # If MTS didn't improve, shrink search range to refine precision
                search_range *= 0.5
            
            # Stop if range is too small (precision limit)
            if np.max(search_range) < 1e-12:
                break
                
        return x_curr, f_curr

    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        
        # 1. Configuration for this run
        # L-SHADE Population Sizing: 18*dim is a robust standard
        pop_size = int(max(30, min(18 * dim, 150)))
        min_pop = 4
        
        # 2. Initialization with OBL (Opposition Based Learning)
        # Generate initial random population
        pop_rand = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Generate opposite population: x' = min + max - x
        pop_obl = min_b + max_b - pop_rand
        pop_obl = np.clip(pop_obl, min_b, max_b)
        
        # Evaluate combined pool (2 * pop_size)
        combined_pop = np.vstack((pop_rand, pop_obl))
        combined_fit = np.zeros(len(combined_pop))
        
        for i in range(len(combined_pop)):
            if check_time(): return global_best_val
            val = func(combined_pop[i])
            combined_fit[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = combined_pop[i].copy()
                
        # Select best N individuals for initial population
        sort_idx = np.argsort(combined_fit)
        population = combined_pop[sort_idx[:pop_size]]
        fitness = combined_fit[sort_idx[:pop_size]]
        
        # 3. L-SHADE Memories
        mem_size = 5
        M_CR = np.full(mem_size, 0.5)
        M_F = np.full(mem_size, 0.5)
        k_mem = 0
        
        # External Archive (stores successful parents to maintain diversity)
        # Fixed capacity based on initial pop size
        max_arc_size = int(pop_size * 2.0)
        archive = np.zeros((max_arc_size, dim))
        arc_count = 0
        
        # Loop State
        stag_count = 0
        prev_best = fitness[0]
        
        # 4. Evolutionary Cycle
        while not check_time():
            # Sort Population (Best -> Worst)
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            curr_best = fitness[0]
            
            # --- Stagnation Check ---
            if curr_best < prev_best - 1e-13:
                prev_best = curr_best
                stag_count = 0
            else:
                stag_count += 1
            
            # --- Adaptive Population Reduction ---
            # If stagnating, delete worst individual to increase convergence pressure
            if stag_count > 5 and pop_size > min_pop:
                pop_size -= 1
                population = population[:pop_size]
                fitness = fitness[:pop_size]
                
                # Cap archive count to current pop_size scale if needed, 
                # but usually keeping archive large is fine.
            
            # --- Restart / Polish Trigger ---
            pop_std = np.std(fitness)
            # Restart if: Population too small OR Stagnated too long OR Variance collapsed
            if pop_size <= min_pop or stag_count > 25 or pop_std < 1e-12:
                
                # Perform Hybrid Polishing before Restart
                rem_time = (time_limit - (datetime.now() - start_time)).total_seconds()
                
                # Only polish if time allows (>0.1s)
                if rem_time > 0.1:
                    # Allocate budget: up to 30% of remaining time, max 1.5s
                    budget = min(1.5, rem_time * 0.3)
                    
                    # Target: Refine the global best solution
                    target_sol = global_best_sol if global_best_sol is not None else population[0]
                    target_fit = global_best_val
                    
                    p_sol, p_val = hybrid_polish(target_sol, target_fit, budget)
                    
                    if p_val < global_best_val:
                        global_best_val = p_val
                        global_best_sol = p_sol.copy()
                
                break # Break inner loop -> Restart
            
            # --- L-SHADE Parameter Generation ---
            r_idx = np.random.randint(0, mem_size, pop_size)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]
            
            # CR = Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F = Cauchy(M_F, 0.1)
            f_p = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Repair F <= 0
            while True:
                mask_neg = f_p <= 0
                if not np.any(mask_neg): break
                f_p[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
            f_p = np.clip(f_p, 0, 1)
            
            # --- Mutation: current-to-pbest/1 ---
            # p (top %) scales with population size
            p_ratio = max(2.0/pop_size, 0.11)
            n_best = int(max(1, p_ratio * pop_size))
            
            # Select p-best
            pbest_idx = np.random.randint(0, n_best, pop_size)
            x_pbest = population[pbest_idx]
            
            # Select r1 (!= i)
            r1 = np.random.randint(0, pop_size, pop_size)
            mask_self = (r1 == np.arange(pop_size))
            r1[mask_self] = (r1[mask_self] + 1) % pop_size
            x_r1 = population[r1]
            
            # Select r2 (Union of Pop and Archive)
            r2 = np.random.randint(0, pop_size + arc_count, pop_size)
            x_r2 = np.zeros((pop_size, dim))
            mask_p = r2 < pop_size
            x_r2[mask_p] = population[r2[mask_p]]
            if arc_count > 0:
                mask_a = ~mask_p
                x_r2[mask_a] = archive[r2[mask_a] - pop_size]
            
            # Compute Mutant Vector
            mutant = population + f_p[:, None] * (x_pbest - population) + f_p[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            mask_cross = rand_vals < cr[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, population)
            
            # --- Selection & Memory Update ---
            succ_cr = []
            succ_f = []
            diffs = []
            
            for i in range(pop_size):
                if check_time(): return global_best_val
                
                f_tri = func(trial[i])
                
                if f_tri <= fitness[i]:
                    diff = fitness[i] - f_tri
                    
                    # Update Archive
                    if arc_count < max_arc_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Replace random archive member
                        archive[np.random.randint(0, max_arc_size)] = population[i].copy()
                    
                    population[i] = trial[i]
                    fitness[i] = f_tri
                    
                    # Record Success
                    succ_cr.append(cr[i])
                    succ_f.append(f_p[i])
                    diffs.append(diff)
                    
                    if f_tri < global_best_val:
                        global_best_val = f_tri
                        global_best_sol = trial[i].copy()
            
            # --- Update History Memory (Weighted Lehmer Mean) ---
            if diffs:
                w = np.array(diffs)
                total_diff = np.sum(w)
                if total_diff > 0:
                    w = w / total_diff
                    
                    # M_CR
                    m_cr_new = np.sum(w * np.array(succ_cr))
                    M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                    
                    # M_F
                    sf = np.array(succ_f)
                    num = np.sum(w * (sf**2))
                    den = np.sum(w * sf)
                    m_f_new = num / den if den > 0 else 0.5
                    M_F[k_mem] = np.clip(m_f_new, 0, 1)
                    
                    k_mem = (k_mem + 1) % mem_size
                    
    return global_best_val
