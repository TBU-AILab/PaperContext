#The following is an improved algorithm called **L-SHADE-RPS (Linear Population Size Reduction - Success-History Adaptive Differential Evolution with Restarts and Pattern Search)**.
#
#### Algorithm Explanation
#This solution integrates three powerful meta-heuristic strategies to minimize the output value efficiently:
#1.  **L-SHADE with Linear Population Reduction**: This is the core optimizer. It adapts the mutation ($F$) and crossover ($CR$) parameters based on historical success, learning the landscape's features. Crucially, it linearly reduces the population size from a large initial set (exploration) to a small set (exploitation) over the duration of each "restart episode."
#2.  **Adaptive Restarts**: Instead of a single run, the algorithm performs multiple "episodes". If the population converges or the allocated time for the episode expires, it restarts with a fresh, large population (initialized using **Opposition-Based Learning** to cover diverse areas). This prevents getting stuck in local optima.
#3.  **Pattern Search (Local Refinement)**: At the end of each restart cycle, a lightweight, derivative-free local search (Pattern Search) is applied to the best solution found. This "polishing" step significantly improves precision, often finding the exact minimum that Differential Evolution might hover around but miss.
#
#### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-RPS:
    - L-SHADE: Adaptive parameters and linear population reduction.
    - Restarts: To escape local optima.
    - Pattern Search: Local refinement for high precision.
    """
    start_time = datetime.now()
    # Reserve a small buffer to ensure safe return
    time_limit = timedelta(seconds=max_time * 0.98)

    # --- Pre-processing Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Global best tracking
    global_best_fit = float('inf')
    global_best_sol = None

    # Helper: Safe evaluation
    def safe_func(x):
        try:
            return func(x)
        except:
            return float('inf')

    # Helper: Local Search (Pattern Search / Coordinate Descent)
    def local_search(current_sol, current_fit, end_time):
        # Initial step size relative to bounds
        step_size = diff_b * 0.05
        ls_sol = current_sol.copy()
        ls_fit = current_fit
        min_step = 1e-9
        
        # Iterate while step size is significant and time permits
        while np.max(step_size) > min_step:
            if datetime.now() >= end_time: break
            
            improved = False
            for i in range(dim):
                if datetime.now() >= end_time: break
                
                # Forward move
                temp_sol = ls_sol.copy()
                temp_sol[i] += step_size[i]
                if temp_sol[i] > max_b[i]: temp_sol[i] = max_b[i]
                
                val = safe_func(temp_sol)
                if val < ls_fit:
                    ls_fit = val
                    ls_sol[i] = temp_sol[i]
                    improved = True
                    continue # Greedy move
                
                # Backward move
                temp_sol[i] = ls_sol[i] - step_size[i]
                if temp_sol[i] < min_b[i]: temp_sol[i] = min_b[i]
                
                val = safe_func(temp_sol)
                if val < ls_fit:
                    ls_fit = val
                    ls_sol[i] = temp_sol[i]
                    improved = True
            
            # If no improvement in any dimension, reduce step size
            if not improved:
                step_size *= 0.5
                
        return ls_sol, ls_fit

    # --- Main Restart Loop ---
    first_run = True
    
    while True:
        now = datetime.now()
        if now - start_time >= time_limit:
            break
            
        # Determine budget for this restart episode
        # Strategy: Allocate ~50% of remaining time to the current run
        # This ensures we effectively manage time for both exploration and refinement
        remaining = (start_time + time_limit - now).total_seconds()
        if remaining < 0.5: break # Too little time for a meaningful run
        
        if first_run:
            run_budget = max(2.0, remaining * 0.5)
            first_run = False
        else:
            run_budget = max(1.0, remaining * 0.5)
            
        episode_end_time = now + timedelta(seconds=run_budget)

        # --- L-SHADE Initialization ---
        # Initial population size (Exploration)
        N_init = int(round(18 * dim))
        N_init = max(20, N_init) # Lower bound constraint
        # Final population size (Exploitation)
        N_min = 4
        
        current_pop_size = N_init
        
        # OBL Initialization (Population + Opposition)
        rand_pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
        opp_pop = min_b + max_b - rand_pop
        
        # Combine and Select Best half
        combined_pop = np.vstack((rand_pop, opp_pop))
        combined_pop = np.clip(combined_pop, min_b, max_b)
        combined_fit = np.full(len(combined_pop), float('inf'))
        
        for i in range(len(combined_pop)):
            if datetime.now() >= episode_end_time: break
            combined_fit[i] = safe_func(combined_pop[i])
            
        # Select best N_init
        sort_indices = np.argsort(combined_fit)
        pop = combined_pop[sort_indices[:current_pop_size]]
        fitness = combined_fit[sort_indices[:current_pop_size]]
        
        # Elitism: Inject global best if it exists
        if global_best_sol is not None:
            if fitness[0] > global_best_fit:
                pop[0] = global_best_sol.copy()
                fitness[0] = global_best_fit
            elif fitness[0] < global_best_fit:
                global_best_fit = fitness[0]
                global_best_sol = pop[0].copy()
        else:
            # Initialize global best
            if len(fitness) > 0:
                global_best_fit = fitness[0]
                global_best_sol = pop[0].copy()

        # SHADE Memory
        mem_size = 5
        mem_M_F = np.full(mem_size, 0.5)
        mem_M_CR = np.full(mem_size, 0.5)
        k_mem = 0
        archive = []
        arc_rate = 2.0
        
        # --- Evolution Loop ---
        while datetime.now() < episode_end_time:
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate target size based on time elapsed in this episode
            elapsed_run = (datetime.now() - now).total_seconds()
            ratio = min(1.0, elapsed_run / run_budget)
            
            target_size = int(round((N_min - N_init) * ratio + N_init))
            target_size = max(N_min, target_size)
            
            # Reduce population
            if current_pop_size > target_size:
                # Keep best
                idx_sort = np.argsort(fitness)
                keep = idx_sort[:target_size]
                pop = pop[keep]
                fitness = fitness[keep]
                current_pop_size = target_size
                
                # Resize archive
                max_arc = int(current_pop_size * arc_rate)
                if len(archive) > max_arc:
                    # Randomly remove excess
                    import random # Safe to import here or top
                    archive = random.sample(archive, max_arc)

            # Convergence Check (Std Dev)
            if np.std(fitness) < 1e-9:
                break

            # 2. Parameter Adaptation
            r_idx = np.random.randint(0, mem_size, current_pop_size)
            m_f = mem_M_F[r_idx]
            m_cr = mem_M_CR[r_idx]
            
            # Generate CR (Normal)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy)
            f = m_f + 0.1 * np.random.standard_cauchy(current_pop_size)
            # Repair F
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
            f = np.minimum(f, 1.0)
            
            # 3. Mutation: current-to-pbest/1
            p_val = max(2, int(current_pop_size * 0.11))
            sorted_idx = np.argsort(fitness)
            top_p = sorted_idx[:p_val]
            
            pbest_idx = np.random.choice(top_p, current_pop_size)
            x_pbest = pop[pbest_idx]
            
            # r1 selection
            all_idx = np.arange(current_pop_size)
            r1 = np.random.randint(0, current_pop_size, current_pop_size)
            r1 = np.where(r1 == all_idx, (r1 + 1) % current_pop_size, r1)
            x_r1 = pop[r1]
            
            # r2 selection (Union of pop and archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
                
            r2 = np.random.randint(0, len(union_pop), current_pop_size)
            # Basic collision avoidance for r2 (vs r1 or current)
            conflict = (r2 == all_idx) | (r2 == r1)
            if np.any(conflict):
                r2[conflict] = (r2[conflict] + 1) % len(union_pop)
            x_r2 = union_pop[r2]
            
            # Compute Mutant
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, current_pop_size)
            mask_j = np.zeros((current_pop_size, dim), dtype=bool)
            mask_j[np.arange(current_pop_size), j_rand] = True
            mask_cr = np.random.rand(current_pop_size, dim) < cr[:, np.newaxis]
            
            trial_pop = np.where(mask_cr | mask_j, mutant, pop)
            
            # 5. Selection
            succ_f, succ_cr, diff_f = [], [], []
            
            for i in range(current_pop_size):
                if datetime.now() >= episode_end_time: break
                
                f_trial = safe_func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diff_f.append(fitness[i] - f_trial)
                        
                        archive.append(pop[i].copy())
                        max_arc = int(current_pop_size * arc_rate)
                        if len(archive) > max_arc:
                            archive.pop(np.random.randint(0, len(archive)))
                            
                    pop[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial
                        global_best_sol = trial_pop[i].copy()
                        
            # 6. Memory Update (Weighted Lehmer Mean)
            if len(succ_f) > 0:
                succ_f = np.array(succ_f)
                succ_cr = np.array(succ_cr)
                diff_f = np.array(diff_f)
                weights = diff_f / np.sum(diff_f)
                
                mean_f = np.sum(weights * (succ_f**2)) / np.sum(weights * succ_f)
                mean_cr = np.sum(weights * succ_cr)
                
                mem_M_F[k_mem] = 0.5 * mem_M_F[k_mem] + 0.5 * mean_f
                mem_M_CR[k_mem] = 0.5 * mem_M_CR[k_mem] + 0.5 * mean_cr
                k_mem = (k_mem + 1) % mem_size
        
        # --- Local Refinement Phase (Pattern Search) ---
        # Polish the global best solution with whatever time is left in this episode slice
        # or a small fixed budget to ensure we don't consume all global time.
        if global_best_sol is not None:
            # Check if we still have absolute time
            if datetime.now() < start_time + time_limit:
                # Refine
                polish_end = datetime.now() + timedelta(seconds=0.5) # limit polish duration
                p_sol, p_fit = local_search(global_best_sol, global_best_fit, polish_end)
                if p_fit < global_best_fit:
                    global_best_fit = p_fit
                    global_best_sol = p_sol

    return global_best_fit
