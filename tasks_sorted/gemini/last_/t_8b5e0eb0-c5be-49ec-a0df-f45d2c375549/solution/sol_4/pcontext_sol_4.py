#The following algorithm implements **L-SHADE-MTS**, a hybrid optimization strategy combining **Linear Population Size Reduction (LPSR) L-SHADE** with a **Multiple Trajectory Search (MTS-LS1)** local search mechanism.
#
#**Key Improvements over the previous version:**
#1.  **Local Search Refinement (MTS-LS1)**: While L-SHADE is excellent at exploring the global basin, it can struggle to refine the final decimals. MTS-LS1 (a coordinate descent method) is applied to the best solution found so far to strictly exploit the local area.
#2.  **Optimized Restart Criteria**: The restart mechanism is preserved but tuned to be less aggressive, only triggering when the population diversity (variance) effectively collapses.
#3.  **Weighted Lehmer Mean**: Uses the precise L-SHADE memory update logic (weighted by fitness improvement) to adapt crossover (`CR`) and scaling (`F`) factors more accurately.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Linear Population Size Reduction and MTS-LS1 Local Search.
    
    Glossary:
    - func: Objective function
    - dim: Dimensions
    - bounds: [low, high] for each dim
    - max_time: Time budget in seconds
    """
    
    # --- Timer & Safety ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def check_remaining_time():
        return (datetime.now() - start_time) < time_limit

    # --- Pre-computation ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    bounds_range = max_b - min_b
    
    # Global Best tracking
    best_global_val = float('inf')
    best_global_vec = None

    # --- Helper Functions ---
    
    def apply_bounds(x):
        """Reflective boundary handling."""
        # Reflect lower
        bad_low = x < min_b
        while np.any(bad_low):
            x[bad_low] = 2 * min_b[bad_low] - x[bad_low]
            bad_low = x < min_b
        
        # Reflect upper
        bad_high = x > max_b
        while np.any(bad_high):
            x[bad_high] = 2 * max_b[bad_high] - x[bad_high]
            bad_high = x > max_b
            
        # Hard clip safety
        return np.clip(x, min_b, max_b)

    def lhs_init(n_samples):
        """Latin Hypercube Sampling."""
        grid = np.arange(n_samples).reshape(-1, 1)
        perm_grid = np.zeros((n_samples, dim))
        for d in range(dim):
            perm_grid[:, d] = np.random.permutation(grid[:, 0])
        jitter = np.random.rand(n_samples, dim)
        samples = (perm_grid + jitter) / n_samples
        return min_b + samples * bounds_range

    # --- Local Search (MTS-LS1) ---
    # Maintains a search range (sr) for each dimension
    mts_sr = (max_b - min_b) * 0.4
    
    def local_search_mts(curr_best_vec, curr_best_val, search_range):
        """
        Coordinate descent local search (MTS-LS1).
        Modifies best_vec in place if improved, updates global best.
        """
        nonlocal best_global_val, best_global_vec
        
        improved = False
        # Randomize dimension order
        dims_to_search = np.random.permutation(dim)
        
        # We perform a limited number of steps to conserve time
        # Check time before starting
        if not check_remaining_time(): return curr_best_vec, curr_best_val, search_range

        x = curr_best_vec.copy()
        val = curr_best_val

        for d in dims_to_search:
            # Check time periodically
            if not check_remaining_time(): break
                
            sr = search_range[d]
            
            # Try Negative Move
            x[d] -= sr
            x = apply_bounds(x)
            new_val = func(x)
            
            if new_val < val:
                val = new_val
                curr_best_vec[d] = x[d] # Update original vector dim
                if val < best_global_val:
                    best_global_val = val
                    best_global_vec = x.copy()
                improved = True
            else:
                # Restore and Try Positive Move (0.5 step)
                x[d] += sr # Restore
                x[d] += 0.5 * sr
                x = apply_bounds(x)
                new_val = func(x)
                
                if new_val < val:
                    val = new_val
                    curr_best_vec[d] = x[d]
                    if val < best_global_val:
                        best_global_val = val
                        best_global_vec = x.copy()
                    improved = True
                else:
                    # Restore and Shrink Range
                    x[d] -= 0.5 * sr
                    search_range[d] *= 0.5 # Shrink search radius for this dim
        
        return curr_best_vec, val, search_range

    # --- Main Loop (Restarts) ---
    
    restart_count = 0
    
    while check_remaining_time():
        
        # 1. Parameter Initialization for L-SHADE
        initial_pop_size = 18 * dim
        initial_pop_size = np.clip(initial_pop_size, 30, 300) # Reasonable limits
        min_pop_size = 4
        
        pop_size = int(initial_pop_size)
        if restart_count > 0:
            # Reduce restart size slightly to save evaluations
            pop_size = int(initial_pop_size * 0.8)
            mts_sr = (max_b - min_b) * 0.4 # Reset search ranges on restart
            
        pop = lhs_init(pop_size)
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if not check_remaining_time(): return best_global_val
            f = func(pop[i])
            fitness[i] = f
            if f < best_global_val:
                best_global_val = f
                best_global_vec = pop[i].copy()
                
        # Inject global best if restarting (Elitism)
        if restart_count > 0 and best_global_vec is not None:
            pop[0] = best_global_vec
            fitness[0] = best_global_val
            
        # L-SHADE Memory
        H = 5
        mem_M_sf = np.full(H, 0.5)
        mem_M_cr = np.full(H, 0.5)
        mem_k = 0
        archive = []
        
        # Inner Loop: Evolution
        while check_remaining_time():
            
            # -- Linear Population Size Reduction (LPSR) --
            # Calculate allowed evals approx (time based)
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = elapsed / max_time
            if progress > 1.0: break
            
            # Target population size
            plan_pop_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
            plan_pop_size = max(min_pop_size, plan_pop_size)
            
            if pop_size > plan_pop_size:
                # Reduce
                n_remove = pop_size - plan_pop_size
                sorted_idx = np.argsort(fitness)
                # Keep best
                pop = pop[sorted_idx[:-n_remove]]
                fitness = fitness[sorted_idx[:-n_remove]]
                pop_size = plan_pop_size
                # Resize archive
                if len(archive) > pop_size:
                    import random
                    random.shuffle(archive)
                    archive = archive[:pop_size]

            # -- Convergence Check (Restart Trigger) --
            if pop_size >= min_pop_size:
                if np.std(fitness) < 1e-10 and (np.max(fitness) - np.min(fitness)) < 1e-10:
                    break # Restart
            
            # -- Parameter Generation --
            # Sort for p-best
            sorted_idx = np.argsort(fitness)
            
            mem_rand_idx = np.random.randint(0, H, pop_size)
            mu_sf = mem_M_sf[mem_rand_idx]
            mu_cr = mem_M_cr[mem_rand_idx]
            
            # Generate CR (Normal)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            # Fix CR ~ -1 (terminal case handling in some papers, here we just clip)
            
            # Generate F (Cauchy)
            f = mu_sf + 0.1 * np.random.standard_cauchy(pop_size)
            f[f > 1.0] = 1.0
            f[f <= 0.0] = 0.4 # Check constraints
            
            # -- Mutation & Crossover --
            # current-to-pbest/1
            # Dynamic p
            p_min = 2.0 / pop_size
            p = 0.2 * (1 - progress) + p_min # reduces from 0.2 to p_min
            p = max(p, p_min)
            
            pbest_num = max(1, int(p * pop_size))
            
            # Union Population (Pop + Archive)
            if len(archive) > 0:
                pop_all = np.vstack((pop, np.array(archive)))
            else:
                pop_all = pop
                
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros_like(fitness)
            
            success_sf = []
            success_cr = []
            diff_fitness = []
            
            for i in range(pop_size):
                if not check_remaining_time(): return best_global_val
                
                # Select p-best
                pbest_idx = sorted_idx[np.random.randint(0, pbest_num)]
                x_pbest = pop[pbest_idx]
                
                # Select r1
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # Select r2
                r2 = np.random.randint(0, len(pop_all))
                while r2 == i or r2 == r1: r2 = np.random.randint(0, len(pop_all))
                x_r2 = pop_all[r2]
                
                # Mutate
                mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (x_r1 - x_r2)
                
                # Binomial Crossover
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                
                trial = apply_bounds(trial)
                
                trial_fit = func(trial)
                
                # Selection
                if trial_fit <= fitness[i]:
                    # Better or Equal
                    if trial_fit < fitness[i]:
                        archive.append(pop[i].copy())
                        success_sf.append(f[i])
                        success_cr.append(cr[i])
                        diff_fitness.append(fitness[i] - trial_fit)
                        
                    new_pop[i] = trial
                    new_fitness[i] = trial_fit
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
                    
                if trial_fit < best_global_val:
                    best_global_val = trial_fit
                    best_global_vec = trial.copy()

            pop = new_pop
            fitness = new_fitness
            
            # Trim Archive
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
            
            # Update Memory (Weighted Lehmer Mean)
            if len(success_sf) > 0:
                s_sf = np.array(success_sf)
                s_cr = np.array(success_cr)
                df = np.array(diff_fitness)
                
                # Weights
                total_df = np.sum(df)
                if total_df == 0:
                    w = np.ones_like(df) / len(df)
                else:
                    w = df / total_df
                
                # Mean SF
                mean_sf = np.sum(w * (s_sf ** 2)) / np.sum(w * s_sf)
                # Mean CR (ignore -1s if they existed, but we clipped to 0)
                mean_cr = np.sum(w * s_cr)
                
                mem_M_sf[mem_k] = 0.5 * mem_M_sf[mem_k] + 0.5 * mean_sf
                mem_M_cr[mem_k] = 0.5 * mem_M_cr[mem_k] + 0.5 * mean_cr
                mem_k = (mem_k + 1) % H

            # --- Local Search Trigger ---
            # Run LS on the global best periodically to exploit promising basins
            # Run if we had some success in the population, or every 50 gens
            if len(success_sf) > 0 and best_global_vec is not None:
                # Run LS on the best vector
                # We trust MTS-LS1 to return the modified vector and new value
                best_vec_copy = best_global_vec.copy()
                best_vec_new, best_val_new, mts_sr = local_search_mts(best_vec_copy, best_global_val, mts_sr)
                
                if best_val_new < best_global_val:
                    best_global_val = best_val_new
                    best_global_vec = best_vec_new
                    # Inject back into population (replace worst)
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_global_vec
                    fitness[worst_idx] = best_global_val

        restart_count += 1
        
    return best_global_val
