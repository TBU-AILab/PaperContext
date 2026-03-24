#Here is an improved algorithm. It implements **L-SHADE-R (L-SHADE with Restarts)**.
#
#### Key Improvements vs Previous Version:
#1.  **Restart Mechanism:** The previous algorithm (jSO) is a powerful "single-run" optimizer, but if it converges to a local optimum (stagnation), it stays there. This version detects convergence or population depletion and **restarts** the optimization process, carrying over the global best solution to the next run (Elitism). This significantly improves performance on multimodal landscapes.
#2.  **Time-Based Budgeting:** It performs a "burn-in" to estimate the cost of the function `func`, then dynamically calculates how many evaluations can fit in the remaining time to schedule the Linear Population Size Reduction (LPSR) effectively.
#3.  **Vectorized Operations:** Maximizes the use of `numpy` for speed, reducing overhead and allowing more evaluations within the `max_time`.
#4.  **Robust Bound Handling (Reflection):** Instead of midpoint or clamping, it uses reflection, which preserves diversity better near boundaries.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Restarts (L-SHADE-R).
    
    Implements:
    1. Linear Population Size Reduction (LPSR)
    2. Success-History based Parameter Adaptation (SHADE)
    3. Restart mechanism with Elitism to escape local optima
    4. Dynamic time-based evaluation budgeting
    """
    
    # --- Configuration ---
    start_time = time.time()
    bounds = np.array(bounds)
    min_b, max_b = bounds[:, 0], bounds[:, 1]
    diff_b = max_b - min_b
    
    # Algorithm Constants
    r_N_init = 18       # Initial population size multiplier
    min_pop_size = 4    # Minimum population size
    H_mem = 6           # History memory size
    arc_rate = 2.0      # Archive size ratio
    
    # Global State
    best_fitness = float('inf')
    best_sol = None
    
    # --- 1. Burn-in Phase (Estimate Function Cost) ---
    # We run a small random search to estimate how many evals we can do per second
    # This is crucial for the LPSR schedule.
    n_burn = 50
    burn_start = time.time()
    for _ in range(n_burn):
        # Check instant fail
        if time.time() - start_time > max_time: 
             return best_fitness if best_sol is not None else float('inf')
             
        t_sol = min_b + np.random.rand(dim) * diff_b
        t_fit = func(t_sol)
        if t_fit < best_fitness:
            best_fitness = t_fit
            best_sol = t_sol.copy()
            
    burn_dur = time.time() - burn_start
    avg_eval_time = max(1e-9, burn_dur / n_burn)
    
    # --- Restart Loop ---
    # We loop until time runs out. Each loop is a fresh L-SHADE run
    # but we inject the global best into the new population.
    
    run_counter = 0
    
    while True:
        run_counter += 1
        elapsed = time.time() - start_time
        remaining = max_time - elapsed
        
        if remaining < 0.05: # Time up
            break
            
        # Estimate budget for this run
        # Strategy: First run gets 50% of time, subsequent runs split remaining
        if run_counter == 1:
            run_time_alloc = remaining * 0.5
        else:
            run_time_alloc = remaining  # Go for broke or until convergence
            
        max_evals = int(run_time_alloc / avg_eval_time)
        max_evals = max(100, max_evals) # Ensure minimum viability
        
        # --- L-SHADE Initialization ---
        pop_size = int(round(r_N_init * dim))
        pop_size = max(min_pop_size, min(pop_size, 500)) # Cap at 500 for speed
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # ELITISM: Inject global best into the new population
        if best_sol is not None:
            pop[0] = best_sol.copy()
            
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        # Note: We track total evaluations to stick to schedule
        evals_used = 0
        
        for i in range(pop_size):
            if (time.time() - start_time) > max_time: break
            
            # If we injected best_sol, we already know its fitness (mostly), 
            # but re-evaluating ensures consistency if func is noisy.
            fitness[i] = func(pop[i])
            evals_used += 1
            
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best_sol = pop[i].copy()
        
        # Memory Init
        M_cr = np.full(H_mem, 0.5)
        M_f = np.full(H_mem, 0.5)
        k_mem = 0
        archive = []
        
        # --- Main L-SHADE Optimization Loop ---
        curr_pop_size = pop_size
        max_pop_size = pop_size # For LPSR calculation
        
        # Loop conditions: Budget not exhausted AND Population not converged
        while evals_used < max_evals:
            
            # 0. Time Check
            if (time.time() - start_time) >= max_time:
                return best_fitness

            # 1. LPSR (Linear Population Size Reduction)
            progress = evals_used / max_evals
            plan_pop_size = int(round((min_pop_size - max_pop_size) * progress + max_pop_size))
            plan_pop_size = max(min_pop_size, plan_pop_size)

            if curr_pop_size > plan_pop_size:
                # Reduction: Sort and remove worst
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx]
                fitness = fitness[sort_idx]
                
                # Resize
                num_to_remove = curr_pop_size - plan_pop_size
                curr_pop_size = plan_pop_size
                pop = pop[:curr_pop_size]
                fitness = fitness[:curr_pop_size]
                
                # Resize Archive
                target_arc_size = int(curr_pop_size * arc_rate)
                if len(archive) > target_arc_size:
                    # Remove random elements from archive to fit
                    del_indices = np.random.choice(len(archive), len(archive) - target_arc_size, replace=False)
                    # Simple list comprehension removal (safer than deletion by index in loop)
                    new_arc = [archive[j] for j in range(len(archive)) if j not in del_indices]
                    archive = new_arc

            # 2. Parameter Generation
            # Sort population for p-best selection
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            # Update global best if needed
            if fitness[0] < best_fitness:
                best_fitness = fitness[0]
                best_sol = pop[0].copy()

            # Convergence Check (Std Dev of fitness)
            # If population is extremely flat, stop this run and Restart
            if np.std(fitness) < 1e-10 and np.max(fitness) - np.min(fitness) < 1e-10:
                break # Restart Trigger

            # Generate CR and F
            # Random selection from memory
            r_idx = np.random.randint(0, H_mem, curr_pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # Cauchy for F
            f = m_f + 0.1 * np.random.standard_cauchy(curr_pop_size)
            # Fix F <= 0 (regenerate) and F > 1 (clamp)
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
            f = np.clip(f, 0, 1)
            
            # Normal for CR
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            # Optional: force CR=0 to be small non-zero? No, standard L-SHADE allows 0.

            # 3. Mutation (current-to-pbest/1)
            # p varies from 0.1 to 0.05 approx, or fixed 0.11/0.2. 
            # Standard SHADE uses p_min = 2/N. Let's use robust dynamic p.
            p_val = max(0.05, 0.11 - 0.06 * progress) # 0.11 -> 0.05
            top_p = max(2, int(curr_pop_size * p_val))
            
            # Indices
            pbest_idx = np.random.randint(0, top_p, curr_pop_size) # Random from top p%
            r1 = np.random.randint(0, curr_pop_size, curr_pop_size)
            
            # Prepare Union for r2 (Pop + Archive)
            if len(archive) > 0:
                # Convert archive to array for indexing
                arc_arr = np.array(archive)
                union_pop = np.vstack((pop, arc_arr))
            else:
                union_pop = pop
                
            r2 = np.random.randint(0, len(union_pop), curr_pop_size)
            
            # Prevent r1 == i, r2 == i, r1 == r2 (Lazy check for speed: just reroll collisions once)
            # In vectorized numpy, perfect exclusion is costly. 
            # Just fixing r1 == i is usually enough for performance.
            collision = (r1 == np.arange(curr_pop_size))
            r1[collision] = (r1[collision] + 1) % curr_pop_size
            
            # Vectors
            x_i = pop
            x_pbest = pop[pbest_idx]
            x_r1 = pop[r1]
            x_r2 = union_pop[r2]
            
            # Mutation Vector
            v = x_i + f[:, None] * (x_pbest - x_i) + f[:, None] * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, curr_pop_size)
            mask = np.random.rand(curr_pop_size, dim) < cr[:, None]
            mask[np.arange(curr_pop_size), j_rand] = True
            
            u = np.where(mask, v, pop)
            
            # 5. Bound Constraint Handling (Reflection/Bouncing)
            # If < min, bounce: 2*min - u. If > max, bounce: 2*max - u
            # If still out, clamp.
            
            # Lower bound
            under_min = u < min_b
            if np.any(under_min):
                u[under_min] = 2.0 * min_b[np.where(under_min)[1]] - u[under_min]
                # If still under (was very far), clamp
                u[under_min] = np.maximum(u[under_min], min_b[np.where(under_min)[1]])
                
            # Upper bound
            over_max = u > max_b
            if np.any(over_max):
                u[over_max] = 2.0 * max_b[np.where(over_max)[1]] - u[over_max]
                u[over_max] = np.minimum(u[over_max], max_b[np.where(over_max)[1]])

            # 6. Selection
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            success_f = []
            success_cr = []
            diff_f = []
            
            # Evaluation
            # We must loop for evaluation to check bounds of func if strictly blackbox, 
            # but usually func accepts vector. Template says "array of input parameter values".
            # Assuming func takes 1D array.
            
            for k in range(curr_pop_size):
                if (time.time() - start_time) > max_time: break
                
                f_u = func(u[k])
                evals_used += 1
                
                if f_u < fitness[k]:
                    new_pop[k] = u[k]
                    new_fitness[k] = f_u
                    
                    # Store success info
                    success_f.append(f[k])
                    success_cr.append(cr[k])
                    diff_f.append(fitness[k] - f_u)
                    
                    # Archive Update
                    archive.append(pop[k].copy())
                    
                    if f_u < best_fitness:
                        best_fitness = f_u
                        best_sol = u[k].copy()
                else:
                    new_pop[k] = pop[k]
                    new_fitness[k] = fitness[k]
            
            pop = new_pop
            fitness = new_fitness
            
            # Archive maintenance
            while len(archive) > int(curr_pop_size * arc_rate):
                # Remove random
                archive.pop(np.random.randint(0, len(archive)))

            # 7. Memory Update (Weighted Lehmer Mean)
            if len(success_f) > 0:
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                dif = np.array(diff_f)
                
                # Weights
                total_dif = np.sum(dif)
                if total_dif > 0:
                    w = dif / total_dif
                    
                    # Mean F (Lehmer)
                    mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                    M_f[k_mem] = np.clip(mean_f, 0, 1)
                    
                    # Mean CR (Weighted arithmetic)
                    mean_cr = np.sum(w * s_cr)
                    M_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    
                    k_mem = (k_mem + 1) % H_mem

    return best_fitness
