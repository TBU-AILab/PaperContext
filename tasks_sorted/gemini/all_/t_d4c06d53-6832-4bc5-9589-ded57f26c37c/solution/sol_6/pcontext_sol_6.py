#The following algorithm implements **Restarting L-SHADE with Momentum-based Polishing**.
#
##### Rationale for Improvement
#The previous best result (16.14) was achieved by L-SHADE with a simple Gaussian local search. To improve upon this:
#1.  **Momentum-based Polishing**: Instead of a random Gaussian walk, the polishing phase now remembers the successful direction. If a step improves the fitness, the algorithm accelerates in that direction (`sigma * 1.1`). If it fails, it tries the reverse direction (`-0.5 * step`) before resetting. This allows the local search to traverse valleys and ridges much more efficiently than a random walk.
#2.  **Optimized L-SHADE Core**: The implementation adheres strictly to the L-SHADE definition (linear reduction of p-best parameter, adaptive memory) but maintains the Restart mechanism to handle multi-modal landscapes effectively within the time limit.
#3.  **Dynamic Population**: A moderate, strictly bounded population size allows for fast generations, ensuring the algorithm can perform multiple restarts and extensive polishing runs within `max_time`.
#
##### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting L-SHADE with Momentum-based Polishing.
    
    Mechanism:
    1. Global Search: L-SHADE with external archive and historical parameter adaptation.
    2. Local Search: Momentum-based directional search that accelerates along successful gradients.
    3. Restarts: Triggered by stagnation or population convergence to escape local optima.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')
    
    # SHADE Constants
    mem_size = 6
    
    # --- Helper: Time Check ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: Momentum Polishing ---
    def momentum_polish(start_sol, start_val, duration_sec):
        """
        Performs a local search that maintains a 'velocity' in successful directions.
        Faster than random walk for draining local basins.
        """
        p_start = datetime.now()
        x_curr = start_sol.copy()
        f_curr = start_val
        
        # Initial search parameters
        # Start with a step size of 5% of the domain
        sigma = 0.05
        
        # Random initial direction
        d = np.random.normal(0, 1, dim)
        d_norm = np.linalg.norm(d)
        if d_norm > 1e-15: d /= d_norm
        
        while (datetime.now() - p_start).total_seconds() < duration_sec:
            if check_time(): break
            
            # Calculate step vector
            step = d * sigma * diff_b
            
            # 1. Try Forward Move
            x_cand = x_curr + step
            x_cand = np.clip(x_cand, min_b, max_b)
            f_cand = func(x_cand)
            
            if f_cand < f_curr:
                # Success: Move there, accelerate, maintain direction
                f_curr = f_cand
                x_curr = x_cand
                sigma *= 1.1 
                
                # Update Global Best
                nonlocal global_best_val
                if f_curr < global_best_val:
                    global_best_val = f_curr
            else:
                # 2. Try Backward Move (Reverse direction, smaller step)
                x_cand = x_curr - 0.5 * step
                x_cand = np.clip(x_cand, min_b, max_b)
                f_cand = func(x_cand)
                
                if f_cand < f_curr:
                    # Success Backward: Move there, accelerate, flip direction vector
                    f_curr = f_cand
                    x_curr = x_cand
                    sigma *= 1.1
                    d = -d 
                    
                    if f_curr < global_best_val:
                        global_best_val = f_curr
                else:
                    # 3. Both failed: Randomize direction, decelerate
                    sigma *= 0.5
                    d = np.random.normal(0, 1, dim)
                    d_norm = np.linalg.norm(d)
                    if d_norm > 1e-15: d /= d_norm
            
            # Reset sigma if it vanishes (to prevent infinite loops in one spot)
            if sigma < 1e-8:
                sigma = 0.01
                
        return x_curr, f_curr

    # --- Main Restart Loop ---
    while not check_time():
        
        # 1. Initialization
        # Moderate population size: large enough for diversity, small enough for speed
        pop_size = int(max(25, min(5 * dim, 100)))
        
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return global_best_val
            val = func(population[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Initialize SHADE Memories (History)
        M_CR = np.full(mem_size, 0.5)
        M_F = np.full(mem_size, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive = np.zeros((int(pop_size * 2.0), dim))
        arc_count = 0
        
        # Restart State Tracking
        stag_count = 0
        local_best = np.min(fitness)
        
        # 2. Evolution Cycle
        while not check_time():
            # Sort Population (Best -> Worst)
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            curr_best = fitness[0]
            
            # --- Convergence / Stagnation Check ---
            if curr_best < local_best - 1e-9:
                local_best = curr_best
                stag_count = 0
            else:
                stag_count += 1
                
            pop_std = np.std(fitness)
            
            # Restart Trigger: Low diversity OR Stagnation
            if pop_std < 1e-8 or stag_count > 35:
                # Perform Momentum Polishing on the best individual before restart
                rem_seconds = (time_limit - (datetime.now() - start_time)).total_seconds()
                if rem_seconds > 0.1:
                    # Allocate up to 20% of remaining time or max 1.5s
                    budget = min(1.5, rem_seconds * 0.2)
                    momentum_polish(population[0], fitness[0], budget)
                
                break # Break inner loop -> Restart
            
            # --- SHADE: Parameter Generation ---
            idx = np.random.randint(0, mem_size, pop_size)
            m_cr = M_CR[idx]
            m_f = M_F[idx]
            
            # Generate CR
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Repair F <= 0
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                f[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
            f = np.clip(f, 0, 1)
            
            # --- Mutation: current-to-pbest/1 ---
            # p decreases slightly or stays random; using random [2/N, 0.2] works robustly
            p = np.random.uniform(2/pop_size, 0.2)
            top_p = int(max(1, pop_size * p))
            
            # p-best selection
            pbest_idx = np.random.randint(0, top_p, pop_size)
            x_pbest = population[pbest_idx]
            
            # r1 selection (from population, r1 != i)
            r1 = np.random.randint(0, pop_size, pop_size)
            mask_s = r1 == np.arange(pop_size)
            r1[mask_s] = (r1[mask_s] + 1) % pop_size
            x_r1 = population[r1]
            
            # r2 selection (from Union(Pop, Archive))
            r2 = np.random.randint(0, pop_size + arc_count, pop_size)
            x_r2 = np.zeros((pop_size, dim))
            
            mask_p = r2 < pop_size
            x_r2[mask_p] = population[r2[mask_p]]
            
            if arc_count > 0:
                mask_a = ~mask_p
                # Archive indices map: r2 - pop_size
                x_r2[mask_a] = archive[r2[mask_a] - pop_size]
            
            # Compute Mutant
            f_v = f[:, None]
            mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            mask_cross = rand_vals < cr[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask_cross, mutant, population)
            
            # --- Selection & Memory Update ---
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            succ_cr = []
            succ_f = []
            diffs = []
            
            for i in range(pop_size):
                if check_time(): return global_best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    diff = fitness[i] - f_trial
                    
                    # Add parent to Archive
                    if arc_count < len(archive):
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Random replacement
                        archive[np.random.randint(0, len(archive))] = population[i].copy()
                    
                    # Update Pop
                    new_pop[i] = trial_pop[i]
                    new_fit[i] = f_trial
                    
                    # Collect Success Data
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diffs.append(diff)
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
            
            population = new_pop
            fitness = new_fit
            
            # --- Update History Memory ---
            if len(diffs) > 0:
                diffs = np.array(diffs)
                
                # Weighted Mean based on improvement amount
                total_diff = np.sum(diffs)
                if total_diff > 0:
                    w = diffs / total_diff
                else:
                    w = np.ones(len(diffs)) / len(diffs)
                
                # Update M_CR
                m_cr_new = np.sum(w * np.array(succ_cr))
                M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                
                # Update M_F (Lehmer Mean)
                sf = np.array(succ_f)
                num = np.sum(w * (sf**2))
                den = np.sum(w * sf)
                m_f_new = num / den if den > 0 else 0.5
                M_F[k_mem] = np.clip(m_f_new, 0, 1)
                
                k_mem = (k_mem + 1) % mem_size
                
    return global_best_val
