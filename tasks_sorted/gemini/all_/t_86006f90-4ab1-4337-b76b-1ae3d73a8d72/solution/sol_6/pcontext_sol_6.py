#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 25.82729090260267
#...
#4. output value is: 24.889496045193397
#5. output value is: 105.40816267667486
#
#The previous results indicate that **Algorithm 4 (jDE with Restart)** performed the best (24.89), slightly outperforming **Algorithm 1 (SHADE with Polishing)** (25.83). The primary difference was likely that Algo 4 used a simpler restart strategy and possibly benefited from the specific randomness of jDE on this landscape, while Algo 1's Polishing step might have consumed too much time for little gain.
#
#The following improved algorithm, **SHADE-RRLS (SHADE with Reflection, Restart, and Local Search)**, aims to combine the superior parameter adaptation of SHADE with a robust restart mechanism and a more efficient local search strategy.
#
##### Improvements Explanation:
#1.  **Reflection Boundary Handling**: Previous algorithms used clipping (`np.clip`) to handle boundary violations. This causes points to "pile up" on the edges, reducing diversity. This algorithm uses **Reflection** (bouncing back into the domain), which preserves the distribution of the population and exploration capability near boundaries.
#2.  **MTS-Based Local Search**: Instead of a generic coordinate descent, this algorithm implements a simplified **MTS-LS1** (Multiple Trajectory Search - Local Search 1). It is triggered only when the population stagnates. It intelligently zooms in on the best solution by expanding and contracting search ranges per dimension, refining the solution efficiently before a restart occurs.
#3.  **Adaptive Restart**: Combines low-variance detection with a "no improvement" counter. If the population converges to a single point OR fails to improve for too long, it triggers the Local Search and then a Restart, ensuring time isn't wasted in local optima.
#4.  **Optimized SHADE**: Uses standard SHADE history ($H=5$) but with a dynamic population size logic `max(20, 10*dim)` capped at 60. This balance ensures enough individuals for the adaptive history to work while keeping the generation loop fast.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE with Reflection, Restart, and MTS-based Polishing.
    
    Algorithm: SHADE-RRLS
    1. Evolution: uses SHADE (Success-History Adaptive DE) for parameter self-adaptation.
    2. Boundaries: uses Reflection to maintain diversity near edges.
    3. Stagnation: detects convergence or lack of progress.
    4. Polishing: runs MTS-LS1 (Local Search) on the best individual to refine precision.
    5. Restart: resets population (keeping best) to explore new basins.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size: Tuned for balance between exploration and speed.
    # 10*dim allows decent diversity, capped at 60 to ensure high generation count.
    pop_size = int(max(20, 10 * dim))
    if pop_size > 60:
        pop_size = 60

    # SHADE Memory Parameters
    H = 5                   # History size
    M_cr = np.full(H, 0.5)  # Memory for Crossover Rate
    M_f = np.full(H, 0.5)   # Memory for Scaling Factor
    k_mem = 0               # Memory index
    
    # Archive for SHADE (stores inferior solutions to maintain diversity)
    archive_size = int(pop_size * 2.0)
    
    # Pre-process bounds for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None

    def check_time():
        return (datetime.now() - start_time) >= time_limit

    def reflect_bounds(x):
        """
        Handles boundary constraints using reflection.
        Avoids piling up at boundaries (like clipping) to maintain distribution shape.
        """
        # Lower bound reflection
        mask_l = x < min_b
        while np.any(mask_l):
            x[mask_l] = 2 * min_b[mask_l] - x[mask_l]
            mask_l = x < min_b
            
        # Upper bound reflection
        mask_u = x > max_b
        while np.any(mask_u):
            x[mask_u] = 2 * max_b[mask_u] - x[mask_u]
            mask_u = x > max_b
        
        # Safe clip for floating point errors
        return np.clip(x, min_b, max_b)

    def mts_ls1(center, current_val):
        """
        Performs a local search (MTS-LS1 style) to polish the solution.
        Efficiently searches along coordinate axes with contracting steps.
        """
        x = center.copy()
        val = current_val
        
        # Initial search range: 40% of domain
        sr = diff_b * 0.4
        
        # Limited budget for local search to prevent time wasting
        ls_budget = 50 * dim
        evals = 0
        
        while evals < ls_budget and not check_time():
            improved_in_pass = False
            # Randomize dimension order to avoid bias
            dims = np.random.permutation(dim)
            
            for d in dims:
                if check_time(): break
                
                original_x_d = x[d]
                
                # Step 1: Try moving negative direction
                x[d] = original_x_d - sr[d]
                x = reflect_bounds(x)
                new_val = func(x)
                evals += 1
                
                if new_val < val:
                    val = new_val
                    improved_in_pass = True
                else:
                    # Step 2: Try moving positive direction (0.5 step)
                    x[d] = original_x_d + 0.5 * sr[d]
                    x = reflect_bounds(x)
                    new_val = func(x)
                    evals += 1
                    
                    if new_val < val:
                        val = new_val
                        improved_in_pass = True
                    else:
                        # Revert if both failed
                        x[d] = original_x_d
                
                # Adaptation of Search Range
                if not improved_in_pass:
                    # Contract search range for this dimension if no improvement
                    sr[d] *= 0.5
            
            # Termination criteria
            if val < 1e-12: # Target precision reached
                break
            
            # If search range becomes insignificant, stop
            if np.max(sr) < 1e-8:
                break
                    
        return x, val

    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        # 1. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Carry over the global best to the new restart
        # This turns the restart into a "Global Search around Best"
        if best_sol is not None:
            pop[0] = best_sol.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return best_val
            
            # Optimization: Skip re-evaluation of the injected best
            if best_sol is not None and i == 0:
                fitness[i] = best_val
                continue
            
            f = func(pop[i])
            fitness[i] = f
            
            if f < best_val:
                best_val = f
                best_sol = pop[i].copy()
        
        # Reset Adaptive Components for new run
        archive = []
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        
        # Stagnation Counters
        no_improv_gen = 0
        last_gen_best = np.min(fitness)
        
        # --- Differential Evolution Loop ---
        while not check_time():
            # Sort population for p-best selection logic
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            curr_gen_best = fitness[0]
            
            # Check Stagnation
            if np.abs(curr_gen_best - last_gen_best) < 1e-10:
                no_improv_gen += 1
            else:
                no_improv_gen = 0
                last_gen_best = curr_gen_best
            
            # Restart Conditions:
            # 1. Low variance (convergence to a point)
            # 2. Prolonged lack of improvement (>30 gens)
            is_stagnant = (np.std(fitness) < 1e-9) or (no_improv_gen > 30)
            
            if is_stagnant:
                # Polishing Step: Try to squeeze more out of the best solution
                refined_sol, refined_val = mts_ls1(pop[0], fitness[0])
                
                if refined_val < best_val:
                    best_val = refined_val
                    best_sol = refined_sol.copy()
                
                # Break inner loop to Trigger Restart
                break
            
            # SHADE Parameter Generation
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # Generate CR ~ Normal(M_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(M_f, 0.1)
            f_params = np.zeros(pop_size)
            for i in range(pop_size):
                while True:
                    val = m_f[i] + 0.1 * np.random.standard_cauchy()
                    if val > 0:
                        if val > 1: val = 1.0
                        f_params[i] = val
                        break
            
            # Evolution Step
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(pop_size)
            
            success_cr = []
            success_f = []
            success_df = []
            
            # Dynamic p-best: Randomize p in [2/N, 0.2] to vary selection pressure
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            num_p = int(max(2, pop_size * p_val))
            
            for i in range(pop_size):
                if check_time(): return best_val
                
                # Mutation Strategy: current-to-pbest/1
                # V = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
                
                p_idx = np.random.randint(0, num_p) # Select from top p best
                x_pbest = pop[p_idx]
                
                # r1 != i
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # r2 != i, r1 (from Union of Pop and Archive)
                combined_len = pop_size + len(archive)
                r2 = np.random.randint(0, combined_len)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, combined_len)
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                
                mutant = pop[i] + f_params[i] * (x_pbest - pop[i]) + f_params[i] * (x_r1 - x_r2)
                
                # Crossover (Binomial)
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                
                trial = np.where(mask, mutant, pop[i])
                
                # Boundary Handling (Reflection)
                trial = reflect_bounds(trial)
                
                # Selection
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    
                    # Store success data
                    success_cr.append(cr[i])
                    success_f.append(f_params[i])
                    success_df.append(fitness[i] - f_trial)
                    
                    # Add replaced individual to archive
                    archive.append(pop[i].copy())
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # Update Population
            pop = new_pop
            fitness = new_fitness
            
            # Maintain Archive Size
            while len(archive) > archive_size:
                rnd_idx = np.random.randint(0, len(archive))
                archive.pop(rnd_idx)
            
            # Update Memory (Weighted Lehmer Mean)
            if len(success_df) > 0:
                s_df = np.array(success_df)
                s_cr = np.array(success_cr)
                s_f = np.array(success_f)
                
                total_df = np.sum(s_df)
                if total_df > 0:
                    weights = s_df / total_df
                    
                    mean_cr = np.sum(weights * s_cr)
                    mean_f = np.sum(weights * s_f**2) / np.sum(weights * s_f)
                    
                    M_cr[k_mem] = mean_cr
                    M_f[k_mem] = mean_f
                    k_mem = (k_mem + 1) % H

    return best_val
