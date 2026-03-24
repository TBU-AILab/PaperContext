#The following algorithm implements **Restarting L-SHADE with Adaptive Population Reduction and Expanded Momentum Polishing**.
#
##### Rationale for Improvement
#The previous best result (6.76) was achieved by **Restarting L-SHADE with Momentum Polishing**. This improved version refines that approach based on the analysis of the trajectory:
#1.  **Adaptive Population Reduction**: Instead of waiting for a restart or using a global linear reduction, this version actively removes the worst individuals from the population during the evolutionary cycle if stagnation is detected (`stag_count > 5`). This forces the population to concentrate faster on the promising basin, mimicking the pressure of Linear Population Size Reduction (LPSR) but dynamically adapted to the convergence state.
#2.  **Expanded Momentum Polishing**: The previous momentum polishing only accelerated step size. This version introduces an **immediate line search expansion**. When a search direction yields an improvement, the algorithm greedily expands the step size (`*1.2`) in a loop within the same iteration. This allows the local search to traverse down a long valley or ridge in a single "move" rather than requiring multiple iterations of the polishing loop.
#3.  **Dynamic Archive Management**: The external archive size now scales dynamically with the reducing population size, ensuring that the diversity preservation mechanism remains proportional to the current search capability.
#
##### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting L-SHADE with Adaptive Population Reduction
    and Expanded Momentum Polishing.
    
    Mechanism:
    1. Global Search: L-SHADE with history-based parameter adaptation.
    2. Convergence Pressure: Dynamically reduces population size upon stagnation.
    3. Local Search: Momentum-based Polishing with Greedy Line Expansion.
    4. Restart: Re-initializes population when diversity vanishes or size hits minimum.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')
    
    # --- Helper: Check Time ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: Momentum Polishing with Line Expansion ---
    def momentum_polish(start_x, start_f, duration):
        """
        Local search that builds momentum in successful directions 
        and greedily expands steps to traverse valleys efficiently.
        """
        p_start = datetime.now()
        
        # Current state
        x_curr = start_x.copy()
        f_curr = start_f
        
        # Search Parameters
        sigma = 0.05 # Initial step size relative to domain
        
        # Initial random direction
        d = np.random.normal(0, 1, dim)
        norm = np.linalg.norm(d)
        if norm > 1e-15: d /= norm
        
        while (datetime.now() - p_start).total_seconds() < duration:
            if check_time(): break
            
            # Calculate step vector
            step = d * sigma * diff_b
            
            # 1. Try Forward Move
            x_cand = np.clip(x_curr + step, min_b, max_b)
            f_cand = func(x_cand)
            
            if f_cand < f_curr:
                # Success: Move and Expand
                x_curr = x_cand
                f_curr = f_cand
                
                # Update Global Best
                nonlocal global_best_val
                if f_curr < global_best_val:
                    global_best_val = f_curr
                
                # Greedy Line Expansion
                # If direction is good, try to go further immediately
                expand_mul = 1.2
                for _ in range(5): # Cap expansion to avoid over-commitment
                    if check_time(): break
                    step *= expand_mul
                    x_exp = np.clip(x_curr + step, min_b, max_b)
                    f_exp = func(x_exp)
                    
                    if f_exp < f_curr:
                        f_curr = f_exp
                        x_curr = x_exp
                        if f_curr < global_best_val:
                            global_best_val = f_curr
                    else:
                        break # Stop expansion if no improvement
                
                # Accelerate base sigma and keep direction
                sigma *= 1.1
                
            else:
                # 2. Try Backward Move (Reversal)
                step = -0.5 * step # Reverse and shrink
                x_cand = np.clip(x_curr + step, min_b, max_b)
                f_cand = func(x_cand)
                
                if f_cand < f_curr:
                    # Success Backward
                    x_curr = x_cand
                    f_curr = f_cand
                    if f_curr < global_best_val:
                        global_best_val = f_curr
                    
                    # Flip direction vector and accelerate
                    d = -d
                    sigma *= 1.1
                else:
                    # 3. Fail Both: Randomize Direction
                    d = np.random.normal(0, 1, dim)
                    norm = np.linalg.norm(d)
                    if norm > 1e-15: d /= norm
                    
                    # Decelerate sigma to refine search
                    sigma *= 0.5
            
            # Reset sigma if it becomes too small (avoid stagnation)
            if sigma < 1e-8:
                sigma = 0.02
        
        return x_curr, f_curr

    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        
        # 1. Initialization
        # Start with moderate population size (12*dim), capped at 120
        pop_size = int(max(25, min(12 * dim, 120)))
        min_pop = 4
        
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return global_best_val
            val = func(population[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Initialize L-SHADE Memories
        mem_size = 5
        M_CR = np.full(mem_size, 0.5)
        M_F = np.full(mem_size, 0.5)
        k_mem = 0
        
        # Initialize Archive
        # Stores decent solutions to maintain diversity
        archive = np.zeros((pop_size * 2, dim))
        arc_count = 0
        
        # Loop State
        stag_count = 0
        best_gen_val = np.min(fitness)
        
        # 2. Evolutionary Cycle
        while not check_time():
            # Sort Population (Best -> Worst)
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # --- Stagnation Check ---
            curr_best = fitness[0]
            if curr_best < best_gen_val - 1e-12:
                best_gen_val = curr_best
                stag_count = 0
            else:
                stag_count += 1
                
            # --- Adaptive Population Reduction ---
            # If stagnating, remove worst individual to force convergence
            if pop_size > min_pop and stag_count > 5:
                pop_size -= 1
                population = population[:pop_size]
                fitness = fitness[:pop_size]
                
                # Cap Archive size proportionally
                max_arc = pop_size * 2
                if arc_count > max_arc:
                    arc_count = max_arc
            
            # --- Restart Trigger ---
            pop_std = np.std(fitness)
            # Restart if: Population too small OR Stagnation too long OR Variance vanished
            if (pop_size <= min_pop) or (stag_count > 25) or (pop_std < 1e-12):
                # Perform Polishing before restart
                rem_time = (time_limit - (datetime.now() - start_time)).total_seconds()
                if rem_time > 0.1:
                    # Allocate up to 30% of remaining time, capped at 2s
                    pol_budget = min(2.0, rem_time * 0.3)
                    momentum_polish(population[0], fitness[0], pol_budget)
                break
            
            # --- L-SHADE: Parameter Generation ---
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
            # p varies [2/N, 0.2]
            p = np.random.uniform(2.0/pop_size, 0.2)
            n_best = int(max(1, p * pop_size))
            
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
            
            # Compute Mutant
            f_v = f_p[:, None]
            mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            mask_cross = rand_vals < cr[:, None]
            mask_cross[np.arange(pop_size), np.random.randint(0, dim, pop_size)] = True
            
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
                    if arc_count < pop_size * 2:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Replace random archive member
                        rep_idx = np.random.randint(0, arc_count)
                        archive[rep_idx] = population[i].copy()
                    
                    # Update Population
                    population[i] = trial[i]
                    fitness[i] = f_tri
                    
                    # Record Success
                    succ_cr.append(cr[i])
                    succ_f.append(f_p[i])
                    diffs.append(diff)
                    
                    if f_tri < global_best_val:
                        global_best_val = f_tri
            
            # --- Update History Memory (Weighted Lehmer Mean) ---
            if diffs:
                w = np.array(diffs)
                total_diff = np.sum(w)
                if total_diff > 0:
                    w = w / total_diff
                    
                    # M_CR Update
                    m_cr_new = np.sum(w * np.array(succ_cr))
                    M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                    
                    # M_F Update
                    sf = np.array(succ_f)
                    num = np.sum(w * (sf**2))
                    den = np.sum(w * sf)
                    m_f_new = num / den if den > 0 else 0.5
                    M_F[k_mem] = np.clip(m_f_new, 0, 1)
                    
                    k_mem = (k_mem + 1) % mem_size
                    
    return global_best_val
