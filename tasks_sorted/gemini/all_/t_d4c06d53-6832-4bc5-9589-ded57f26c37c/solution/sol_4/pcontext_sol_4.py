#The following algorithm implements **Restarting L-SHADE (Success-History Adaptive Differential Evolution) with an External Archive and Gaussian Polishing**.
#
##### Rationale for Improvement
#The previous best results were achieved by **jDE** (35.85) and **L-SHADE** (38.95).
#1.  **Hybrid Approach**: This version combines the superior parameter adaptation of **L-SHADE** with the **Restart** mechanism of the jDE implementation. L-SHADE is powerful but can get stuck in local basins in complex landscapes. Restarts allow it to escape and explore new basins using the remaining time budget.
#2.  **External Archive**: Unlike the jDE version, this algorithm uses an external archive. This maintains population diversity by allowing mutation strategies to pull difference vectors from historically "decent" solutions, mitigating premature convergence.
#3.  **Gaussian Polishing**: Before restarting, the algorithm performs a focused "hill-climbing" local search on the best solution found in the current run. This exploits the local basin to its absolute minimum (high precision) before abandoning the population.
#4.  **Robustness**: The population size is dynamically tuned to the dimension but capped to ensure enough generations run within `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting L-SHADE with External Archive and Local Polishing.
    
    Key Components:
    - L-SHADE: Historical memory-based adaptation of F and CR parameters.
    - External Archive: Stores good solutions replaced during selection to preserve diversity.
    - Restarts: Automatic population reset upon stagnation or convergence.
    - Gaussian Polishing: Local search to refine best solutions before restart.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution across all restarts
    global_best_val = float('inf')
    
    # SHADE Memory Parameters
    H = 6  # Size of the historical memory
    
    # --- Helper: Check Time ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    def get_remaining_seconds():
        return max_time - (datetime.now() - start_time).total_seconds()

    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        
        # 1. Initialization (New Restart)
        # Population size: Adaptive to dim but capped to allow fast iterations
        pop_size = int(max(20, min(4 * dim, 80)))
        
        # Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return global_best_val
            val = func(population[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                
        # Initialize SHADE Memories (M_CR, M_F) to 0.5
        mem_M_CR = np.full(H, 0.5)
        mem_M_F = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize External Archive
        # Stores parent vectors that were better than their offspring but replaced
        archive = np.zeros((pop_size, dim)) 
        arc_count = 0
        
        # State Tracking for Restart Conditions
        stagnation_count = 0
        prev_best_fit = np.min(fitness)
        
        # 2. Evolutionary Cycle
        while not check_time():
            # Sort population by fitness (Best to Worst)
            # This is required for 'current-to-pbest' mutation
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # --- Check Convergence & Stagnation ---
            current_best = fitness[0]
            
            # Check if we improved the local best
            if current_best < prev_best_fit - 1e-9:
                prev_best_fit = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Check Population Variance (Convergence)
            pop_std = np.std(fitness)
            
            # RESTART TRIGGER:
            # If population has collapsed (low std) OR hasn't improved for many generations
            if pop_std < 1e-6 or stagnation_count > 45:
                # -> Execute Polishing Phase before abandoning this population
                remaining = get_remaining_seconds()
                if remaining > 0.1: 
                    # Allocate a small budget (max 1s or 20% of remaining time)
                    budget = min(1.0, remaining * 0.2)
                    polish_start = datetime.now()
                    
                    # Hill Climb / Gaussian Walk
                    pc = population[0].copy() # Start from current best
                    pf = fitness[0]
                    sigma = np.max(diff_b) * 0.05 # Initial step size
                    
                    while (datetime.now() - polish_start).total_seconds() < budget:
                        if check_time(): return global_best_val
                        
                        # Sample neighbor
                        cand = pc + np.random.normal(0, 1, dim) * sigma
                        cand = np.clip(cand, min_b, max_b)
                        val = func(cand)
                        
                        if val < pf:
                            pf = val
                            pc = cand
                            sigma *= 1.2 # Accelerate on success
                            if val < global_best_val:
                                global_best_val = val
                        else:
                            sigma *= 0.5 # Decelerate on failure
                            
                        # Stop if step size is too small
                        if sigma < 1e-10: break
                
                break # Break inner loop -> triggers outer loop (Restart)

            # --- SHADE: Parameter Adaptation ---
            # Pick random memory slot for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_M_CR[r_idx]
            m_f = mem_M_F[r_idx]
            
            # Generate CR: Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F: Cauchy(M_F, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Handle F constraints (F > 0)
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                f[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
            f = np.clip(f, 0, 1) # Cap F at 1.0
            
            # --- Mutation: current-to-pbest/1 ---
            # Mutation Vector: V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # 1. Select X_pbest: Randomly from top p%
            p_val = max(2.0/pop_size, 0.15) # p = 15%
            top_p = int(pop_size * p_val)
            pbest_idxs = np.random.randint(0, top_p, pop_size)
            x_pbest = population[pbest_idxs]
            
            # 2. Select X_r1: Randomly from population, r1 != i
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            # Adjust indices where r1 == i
            mask_self = (r1_idxs == np.arange(pop_size))
            r1_idxs[mask_self] = (r1_idxs[mask_self] + 1) % pop_size
            x_r1 = population[r1_idxs]
            
            # 3. Select X_r2: Randomly from Union(Population, Archive)
            union_size = pop_size + arc_count
            r2_idxs = np.random.randint(0, union_size, pop_size)
            
            # Build x_r2 array
            x_r2 = np.zeros((pop_size, dim))
            mask_in_pop = r2_idxs < pop_size
            x_r2[mask_in_pop] = population[r2_idxs[mask_in_pop]]
            
            if arc_count > 0:
                mask_in_arc = ~mask_in_pop
                # Calculate archive indices: r2_idx - pop_size
                arc_indices = r2_idxs[mask_in_arc] - pop_size
                x_r2[mask_in_arc] = archive[arc_indices]
            
            # Compute Mutant
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover (Binomial) ---
            rand_cr = np.random.rand(pop_size, dim)
            mask_cross = rand_cr < cr[:, np.newaxis]
            
            # Ensure at least one parameter comes from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask_cross, mutant, population)
            
            # --- Selection ---
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            # Tracking for Memory Update
            succ_cr = []
            succ_f = []
            diffs = []
            
            for i in range(pop_size):
                if check_time(): return global_best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    # Solution improved
                    diff = fitness[i] - f_trial
                    
                    # 1. Add parent to Archive
                    if arc_count < pop_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Archive full: Replace random member
                        ridx = np.random.randint(0, pop_size)
                        archive[ridx] = population[i].copy()
                        
                    # 2. Store success data
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diffs.append(diff)
                    
                    # 3. Update Population
                    new_pop[i] = trial_pop[i]
                    new_fit[i] = f_trial
                    
                    # 4. Update Global Best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
            
            population = new_pop
            fitness = new_fit
            
            # --- Memory Update (SHADE Logic) ---
            # Weighted Lehmer Mean based on fitness improvement
            if len(succ_cr) > 0:
                succ_cr = np.array(succ_cr)
                succ_f = np.array(succ_f)
                diffs = np.array(diffs)
                
                total_diff = np.sum(diffs)
                if total_diff > 0:
                    weights = diffs / total_diff
                else:
                    weights = np.ones(len(diffs)) / len(diffs)
                
                # Update M_CR (Weighted Mean)
                m_cr_new = np.sum(weights * succ_cr)
                mem_M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                
                # Update M_F (Weighted Lehmer Mean)
                num = np.sum(weights * (succ_f ** 2))
                den = np.sum(weights * succ_f)
                m_f_new = num / den if den > 0 else 0.5
                mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                
                k_mem = (k_mem + 1) % H

    return global_best_val
