#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 23.576400416496995
#2. output value is: 32.12476466108296
#3. output value is: 21.050545782471445
#...
#8. output value is: 24.889496045193397
#9. output value is: 105.40816267667486
#
#The best performance so far was achieved by **Algorithm 3 (SHADE-RRLS)** with a value of **21.05**. It utilized SHADE parameter adaptation, Reflection boundary handling, and a coordinate-wise MTS-based Local Search.
#
#The following improved algorithm, **iSHADE-RRLS (Improved SHADE with Reflection, Restart, and Independent-Step Local Search)**, aims to lower the value further.
#
###### Improvements Explanation:
#1.  **Independent-Step MTS Local Search**: The previous Local Search (MTS-LS1) reduced the search range (`sr`) globally for all dimensions if a pass failed. This version maintains **independent search ranges for each dimension**. This allows the algorithm to "zoom in" on variables that are close to the optimal while taking larger steps in dimensions that are still far away, effectively handling ill-conditioned basins (valleys that are narrow in some directions and flat in others).
#2.  **Refined Stagnation Logic**: Stagnation is detected if the population variance becomes negligible (`1e-9`) OR if there is no improvement for a duration proportional to the dimension (capped at 40 generations).
#3.  **Reflection Boundary Handling**: Preserves the population distribution better than clipping by bouncing out-of-bound particles back into the search space.
#4.  **Optimized SHADE**: Retains the successful configuration of Algorithm 3 (Population ~ `10*dim`, `H=6`) but includes a fully vectorized parameter generation and safer time-check mechanism.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Improved SHADE with Reflection, Restart, and 
    Independent-Step-Size Local Search (iSHADE-RRLS).
    """
    # --- Timing Setup ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Configuration ---
    # Population Size: Based on previous success, moderate size allows for
    # sufficient diversity without slowing down generations significantly.
    pop_size = int(max(20, 10 * dim))
    if pop_size > 60:
        pop_size = 60
        
    # SHADE Parameters
    H = 6                   # History memory size
    M_cr = np.full(H, 0.5)  # Memory for Crossover Rate
    M_f = np.full(H, 0.5)   # Memory for Scaling Factor
    k_mem = 0               # Memory index
    
    # Archive size
    archive_size = int(pop_size * 2.0)
    
    # Pre-calc bounds for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Helper: Reflection Boundary Handling ---
    def reflect_bounds(x):
        """
        Reflects out-of-bound coordinates back into the domain.
        Preserves distribution better than clipping.
        """
        # Lower bound check
        mask_l = x < min_b
        while np.any(mask_l):
            x[mask_l] = 2 * min_b[mask_l] - x[mask_l]
            mask_l = x < min_b
        
        # Upper bound check
        mask_u = x > max_b
        while np.any(mask_u):
            x[mask_u] = 2 * max_b[mask_u] - x[mask_u]
            mask_u = x > max_b
            
        return np.clip(x, min_b, max_b)

    # --- Helper: Independent-Step MTS Local Search ---
    def mts_ls_independent(center, current_val):
        """
        MTS-LS1 with independent step sizes for each dimension.
        Allows adapting to the shape of the basin (conditioning).
        """
        nonlocal best_val, best_sol
        
        x = center.copy()
        val = current_val
        
        # Search Range vector (one per dimension)
        # Start with 40% of domain
        sr = diff_b * 0.4 
        
        # Budget for LS to ensure we don't get stuck
        ls_budget = 50 * dim
        used_evals = 0
        
        while used_evals < ls_budget:
            # Check time periodically (every complete pass is too slow, check inside loop)
            if check_time(): break
                
            improved_any = False
            
            # Search dimensions in random order to avoid bias
            dims = np.random.permutation(dim)
            
            for d in dims:
                if check_time(): break
                    
                # Optimization: skip if step size is too small to affect change
                if sr[d] < 1e-13:
                    continue
                
                original_val = x[d]
                
                # 1. Try Negative Move
                x[d] = original_val - sr[d]
                x = reflect_bounds(x)
                new_v = func(x)
                used_evals += 1
                
                if new_v < val:
                    val = new_v
                    improved_any = True
                    # Update global immediately if better
                    if val < best_val:
                        best_val = val
                        best_sol = x.copy()
                else:
                    # 2. Try Positive Move (0.5 step)
                    x[d] = original_val + 0.5 * sr[d]
                    x = reflect_bounds(x)
                    new_v = func(x)
                    used_evals += 1
                    
                    if new_v < val:
                        val = new_v
                        improved_any = True
                        if val < best_val:
                            best_val = val
                            best_sol = x.copy()
                    else:
                        # Revert and Shrink ONLY this dimension
                        x[d] = original_val
                        sr[d] *= 0.5
            
            # If all step sizes are tiny, stop early
            if np.max(sr) < 1e-12:
                break
                
        return x, val

    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        # 1. Initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject best solution found so far into new population
        if best_sol is not None:
            pop[0] = best_sol.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return best_val
            
            # Skip re-evaluation of injected best
            if best_sol is not None and i == 0:
                fitness[i] = best_val
                continue
                
            f = func(pop[i])
            fitness[i] = f
            
            if f < best_val:
                best_val = f
                best_sol = pop[i].copy()
                
        # Reset SHADE Adaptive Memory
        archive = []
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        
        # Stagnation Counters
        no_improv_gens = 0
        last_best_fit = np.min(fitness)
        
        # --- Generation Loop ---
        while not check_time():
            # Sort population (needed for p-best selection)
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            current_best = fitness[0]
            
            # Check Stagnation
            if np.abs(current_best - last_best_fit) < 1e-12:
                no_improv_gens += 1
            else:
                no_improv_gens = 0
                last_best_fit = current_best
            
            # Restart Conditions:
            # 1. Converged variance (population bunched up)
            # 2. No improvement for significant time
            is_converged = np.std(fitness) < 1e-9
            is_stagnant = no_improv_gens > 40
            
            if is_converged or is_stagnant:
                # Deep Polishing: Try to squeeze final precision from best
                polished_sol, polished_val = mts_ls_independent(pop[0], fitness[0])
                if polished_val < best_val:
                    best_val = polished_val
                    best_sol = polished_sol.copy()
                
                # Break inner loop to trigger Restart
                break
            
            # SHADE Parameter Generation
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # Generate CR ~ Normal(M_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(M_f, 0.1) (Vectorized with retry)
            f_params = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            while np.any(f_params <= 0):
                mask = f_params <= 0
                f_params[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
            f_params = np.minimum(f_params, 1.0)
            
            # Evolution Step
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(pop_size)
            
            succ_cr = []
            succ_f = []
            succ_df = []
            
            # Dynamic p-best parameter
            p_val = np.random.uniform(2/pop_size, 0.2)
            top_p = int(max(2, pop_size * p_val))
            
            for i in range(pop_size):
                if check_time(): return best_val
                
                # Mutation: current-to-pbest/1
                # Select pbest from top p%
                p_idx = np.random.randint(0, top_p)
                x_pbest = pop[p_idx]
                
                # Select r1 != i
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # Select r2 != i, r1 from Union(Population, Archive)
                len_arc = len(archive)
                r2 = np.random.randint(0, pop_size + len_arc)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, pop_size + len_arc)
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                    
                mutant = pop[i] + f_params[i] * (x_pbest - pop[i]) + f_params[i] * (x_r1 - x_r2)
                
                # Crossover: Binomial
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                
                trial = np.where(mask, mutant, pop[i])
                
                # Boundary Handling
                trial = reflect_bounds(trial)
                
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    
                    # Store success info
                    succ_cr.append(cr[i])
                    succ_f.append(f_params[i])
                    succ_df.append(fitness[i] - f_trial)
                    
                    # Archive the parent
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
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update SHADE Memory (Weighted Lehmer Mean)
            if len(succ_df) > 0:
                s_df = np.array(succ_df)
                weights = s_df / np.sum(s_df)
                
                mean_cr = np.sum(weights * np.array(succ_cr))
                s_f_arr = np.array(succ_f)
                mean_f = np.sum(weights * s_f_arr**2) / np.sum(weights * s_f_arr)
                
                M_cr[k_mem] = mean_cr
                M_f[k_mem] = mean_f
                k_mem = (k_mem + 1) % H

    return best_val
