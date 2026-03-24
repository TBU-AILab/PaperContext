#To improve upon the previous results, I propose an algorithm based on **L-SHADE-MTS (Linear Population Size Reduction - Success-History Adaptive Differential Evolution with Multiple Trajectory Search)**.
#
##### Algorithm Explanation
#
#1.  **L-SHADE with Linear Population Reduction (LPSR)**: This is a state-of-the-art global optimizer. It dynamically adapts mutation ($F$) and crossover ($CR$) rates based on a memory of successful updates. The "Linear Population Size Reduction" strategy linearly decreases the population size from a large value (exploration) to a minimal value (exploitation) over the course of a run, maximizing the efficiency of the allocated budget.
#2.  **MTS-LS1 (Multiple Trajectory Search - Local Search 1)**: While Differential Evolution is great at finding the basin of attraction, it can be slow to converge to the precise minimum. MTS-LS1 is a powerful local search method that refines the best solution by probing dimensions with asymmetric steps and adjusting search ranges dynamically.
#3.  **OBL (Opposition-Based Learning) Initialization**: Populations are initialized by generating random solutions and their "opposites" within the bounds, selecting the fittest half. This improves initial coverage.
#4.  **Strategic Restarts**: The algorithm runs in "episodes". If the population converges (low variance) or the generation limit for the episode is reached, it restarts. The global best solution is preserved and injected into the new population (Elitism) to prevent regression.
#
##### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-MTS:
    - L-SHADE: Adaptive DE with Linear Population Size Reduction.
    - MTS-LS1: Multiple Trajectory Local Search for precision.
    - Strategic Restarts: To escape local optima.
    """
    start_time = datetime.now()
    # Reserve a small buffer to ensure we return before the strict timeout
    time_limit = timedelta(seconds=max_time * 0.99)
    end_time = start_time + time_limit

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

    # --- MTS-LS1 Local Search ---
    # A robust local search method to refine the best solution
    def mts_ls1(solution, fitness, search_range, budget_end):
        best_x = solution.copy()
        best_f = fitness
        improved_global = False
        
        # Optimize dimensions in random order
        dims = np.arange(dim)
        np.random.shuffle(dims)
        
        for i in dims:
            if datetime.now() >= budget_end:
                break
            
            original_x = best_x[i]
            sr = search_range[i]
            
            # Trajectory 1: Negative direction
            best_x[i] = np.clip(original_x - sr, min_b[i], max_b[i])
            f_new = safe_func(best_x)
            
            if f_new < best_f:
                best_f = f_new
                improved_global = True
            else:
                # Trajectory 2: Positive direction (0.5 step size)
                best_x[i] = np.clip(original_x + 0.5 * sr, min_b[i], max_b[i])
                f_new = safe_func(best_x)
                
                if f_new < best_f:
                    best_f = f_new
                    improved_global = True
                else:
                    # No improvement: revert and shrink search range
                    best_x[i] = original_x
                    search_range[i] *= 0.5
                    
        return best_x, best_f, search_range, improved_global

    # --- Main Restart Loop ---
    # Restarts help explore different basins of attraction
    while datetime.now() < end_time:
        
        # --- Initialization for Episode ---
        # Initial Population: ~18*D is a heuristic sweet spot, bounded for speed
        N_init = int(np.clip(18 * dim, 30, 150))
        N_min = 4 # Minimum population size at end of reduction
        pop_size = N_init
        
        # OBL Initialization: Random + Opposite
        rand_pop = min_b + np.random.rand(pop_size, dim) * diff_b
        opp_pop = min_b + max_b - rand_pop
        opp_pop = np.clip(opp_pop, min_b, max_b)
        
        # Combine and Select Best
        combined_pop = np.vstack((rand_pop, opp_pop))
        combined_fit = np.full(len(combined_pop), float('inf'))
        
        for i in range(len(combined_pop)):
            if datetime.now() >= end_time: break
            combined_fit[i] = safe_func(combined_pop[i])
            
            if combined_fit[i] < global_best_fit:
                global_best_fit = combined_fit[i]
                global_best_sol = combined_pop[i].copy()
                
        # Select best N_init individuals
        sort_idx = np.argsort(combined_fit)
        pop = combined_pop[sort_idx[:pop_size]]
        fitness = combined_fit[sort_idx[:pop_size]]
        
        # Elitism Injection: Ensure we never forget the global best
        if global_best_sol is not None:
             if global_best_fit < fitness[0]:
                 pop[0] = global_best_sol.copy()
                 fitness[0] = global_best_fit
        
        # L-SHADE Memory Initialization
        H_mem = 5
        mem_F = np.full(H_mem, 0.5)
        mem_CR = np.full(H_mem, 0.5)
        k_mem = 0
        
        # Archive (for external diversity)
        max_arc_size = int(2.0 * N_init)
        archive = np.zeros((max_arc_size, dim))
        arc_count = 0
        
        # Local Search Range Initialization
        ls_search_range = (max_b - min_b) * 0.4
        
        # LPSR Schedule: Estimated Max Generations for this episode
        # We assume a fixed budget of generations for linear reduction logic
        max_gens = 300 + 10 * dim
        gen = 0
        
        # --- Generation Loop ---
        while datetime.now() < end_time:
            gen += 1
            
            # 1. Linear Population Size Reduction (LPSR)
            progress = min(1.0, gen / max_gens)
            target_size = int(round((N_min - N_init) * progress + N_init))
            target_size = max(N_min, target_size)
            
            if pop_size > target_size:
                # Reduce population (keep best)
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:target_size]]
                fitness = fitness[sort_idx[:target_size]]
                
                # Resize Archive (maintain ratio)
                curr_arc_cap = int(target_size * 2.0)
                if arc_count > curr_arc_cap:
                    # Randomly subset archive
                    valid_indices = np.random.choice(arc_count, curr_arc_cap, replace=False)
                    archive[:curr_arc_cap] = archive[valid_indices]
                    arc_count = curr_arc_cap
                
                pop_size = target_size
                
            # 2. Parameter Adaptation
            r_idxs = np.random.randint(0, H_mem, pop_size)
            m_f = mem_F[r_idxs]
            m_cr = mem_CR[r_idxs]
            
            # Generate CR (Normal dist)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F (Cauchy dist)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            f = np.where(f <= 0, 0.5, f) # Fallback for non-positive
            f = np.minimum(f, 1.0)       # Clip max
            
            # 3. Mutation (current-to-pbest/1)
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_p_count = int(max(2, p_val * pop_size))
            sorted_idx = np.argsort(fitness)
            
            pbest_indices = np.random.choice(sorted_idx[:top_p_count], pop_size)
            x_pbest = pop[pbest_indices]
            
            # r1 selection
            r1_indices = np.random.randint(0, pop_size, pop_size)
            # Shift collision r1 != i
            r1_indices = np.where(r1_indices == np.arange(pop_size), (r1_indices + 1) % pop_size, r1_indices)
            x_r1 = pop[r1_indices]
            
            # r2 selection (Union of Pop + Archive)
            n_union = pop_size + arc_count
            r2_indices = np.random.randint(0, n_union, pop_size)
            # Collision handling r2 != r1
            r2_indices = np.where(r2_indices == r1_indices, (r2_indices + 1) % n_union, r2_indices)
            
            # Build Union (lazy evaluation)
            if arc_count > 0:
                active_archive = archive[:arc_count]
                x_r2 = np.empty((pop_size, dim))
                mask_in_pop = r2_indices < pop_size
                mask_in_arc = ~mask_in_pop
                x_r2[mask_in_pop] = pop[r2_indices[mask_in_pop]]
                x_r2[mask_in_arc] = active_archive[r2_indices[mask_in_arc] - pop_size]
            else:
                x_r2 = pop[r2_indices % pop_size]
                
            # Compute Mutant
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_j = np.zeros((pop_size, dim), dtype=bool)
            mask_j[np.arange(pop_size), j_rand] = True
            mask_cr = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            trial_pop = np.where(mask_cr | mask_j, mutant, pop)
            
            # 5. Selection and Memory Update
            success_f, success_cr, diff_f = [], [], []
            
            for i in range(pop_size):
                if datetime.now() >= end_time: break
                
                f_trial = safe_func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    # Add to archive
                    if arc_count < max_arc_size:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Replace random individual in archive
                        archive[np.random.randint(0, arc_count)] = pop[i].copy()
                        
                    if f_trial < fitness[i]:
                        diff_f.append(fitness[i] - f_trial)
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        
                    pop[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial
                        global_best_sol = trial_pop[i].copy()
            
            # 6. Update History Memory
            if len(diff_f) > 0:
                diff_f_np = np.array(diff_f)
                succ_f_np = np.array(success_f)
                succ_cr_np = np.array(success_cr)
                
                weights = diff_f_np / np.sum(diff_f_np)
                
                # Weighted Lehmer Mean for F
                mean_f = np.sum(weights * (succ_f_np ** 2)) / np.sum(weights * succ_f_np)
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(weights * succ_cr_np)
                
                mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * mean_f
                mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * mean_cr
                k_mem = (k_mem + 1) % H_mem
                
            # 7. Local Search Integration
            # Periodically refine the global best solution
            if gen % 15 == 0 and global_best_sol is not None:
                # Budget: dynamic small slice of remaining time
                ls_budget_sec = max(0.1, (end_time - datetime.now()).total_seconds() * 0.1)
                ls_end = datetime.now() + timedelta(seconds=ls_budget_sec)
                
                ls_sol, ls_fit, ls_search_range, imp = mts_ls1(global_best_sol, global_best_fit, ls_search_range, ls_end)
                
                if ls_fit < global_best_fit:
                    global_best_fit = ls_fit
                    global_best_sol = ls_sol.copy()
                    # Feedback into population: replace worst
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = ls_sol.copy()
                    fitness[worst_idx] = ls_fit

            # 8. Convergence/Restart Trigger
            # Restart if variance is low (convergence) or max generations reached
            if np.std(fitness) < 1e-8 or gen >= max_gens:
                break
                
    return global_best_fit
