#The following algorithm implements an improved **L-SHADE-jSO (Success-History based Adaptive Differential Evolution)** with **Eigen-Crossover** and **Linear Population Size Reduction (LPSR)**.
#
#**Key Improvements:**
#1.  **L-SHADE-jSO Architecture**: Incorporates the weighting and parameter adaptation rules from the jSO algorithm (a top-performing CEC competition winner). This includes a specific decay for the $p$-best parameter and weighted Lehmer mean updates.
#2.  **Eigen-Crossover**: To tackle rotated objective functions (where variables are dependent), the algorithm computes the covariance matrix of the top individuals and performs crossover in the eigen-coordinate system. This allows the search to follow diagonal valleys efficiently.
#3.  **Corrected Bound Handling**: Uses a "weighted center" reflection strategy ($(Limit + Old)/2$) instead of simple clipping. This preserves population diversity near the boundaries.
#4.  **MTS-LS1 Local Search**: A robust coordinate-descent local search is triggered at the end of restarts to refine the best solution found.
#5.  **Restart Strategy**: If the population converges or becomes too small, the algorithm restarts with a fresh population (while keeping the global best) to escape local optima within the time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using jSO (L-SHADE variant) with Linear Population Size Reduction,
    Eigen-Crossover for rotation invariance, and MTS-LS1 local search.
    """
    # --- Constants & Config ---
    start_time = time.time()
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    best_fitness = float('inf')
    best_sol = None
    
    # Configuration
    H = 5 # History memory size
    
    def get_remaining_time():
        return max_time - (time.time() - start_time)

    # --- MTS-LS1 Local Search (Polish) ---
    def mts_ls1(current_sol, current_fit, budget_time):
        """Coordinate descent local search to refine the solution."""
        ls_start = time.time()
        sol = current_sol.copy()
        fit = current_fit
        
        # Initial search range (40% of domain)
        sr = diff_b * 0.4
        
        improved = True
        while improved:
            improved = False
            # Search in random order of dimensions
            dims = np.random.permutation(dim)
            
            for i in dims:
                # Strict Time Check
                if (time.time() - ls_start) >= budget_time or get_remaining_time() <= 0:
                    return sol, fit
                
                # 1. Try negative step
                x_test = sol.copy()
                x_test[i] = np.clip(sol[i] - sr[i], min_b[i], max_b[i])
                
                val = func(x_test)
                
                # Update Global Best
                if val < best_fitness:
                    nonlocal best_sol, best_fitness
                    best_fitness = val
                    best_sol = x_test.copy()
                
                if val < fit:
                    fit = val
                    sol = x_test
                    sr[i] *= 1.2 # Expand step on success
                    improved = True
                else:
                    # 2. Try positive step
                    x_test = sol.copy()
                    x_test[i] = np.clip(sol[i] + sr[i], min_b[i], max_b[i])
                    
                    val = func(x_test)
                    
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = x_test.copy()
                        
                    if val < fit:
                        fit = val
                        sol = x_test
                        sr[i] *= 1.2
                        improved = True
                    else:
                        sr[i] *= 0.5 # Shrink step on failure
                        
            # Precision check
            if np.max(sr) < 1e-12:
                break
                
        return sol, fit

    # --- Main Optimization Loop (Restarts) ---
    while get_remaining_time() > 0.05:
        
        # 1. Initialize Population
        # jSO/L-SHADE typical initial size is around 18*dim to 25*dim
        pop_size_init = int(np.clip(25 * dim, 30, 200)) 
        pop_size_end = 4
        current_pop_size = pop_size_init
        
        pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
        fitness = np.full(current_pop_size, float('inf'))
        
        # Archive (stores inferior solutions to maintain diversity)
        archive = []
        arc_rate = 2.6 
        
        # Initialize Memory for Adaptive Parameters (F and CR)
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.8)
        k_mem = 0
        
        # Biased Restart: Inject global best if available
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            start_idx = 1
            
        # Evaluation of Initial Population
        eval_times = []
        evals_in_session = 0
        
        for i in range(start_idx, current_pop_size):
            if get_remaining_time() <= 0: return best_fitness
            
            t0 = time.time()
            val = func(pop[i])
            eval_times.append(time.time() - t0)
            
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
                best_sol = pop[i].copy()
                
        # Sort population
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Average evaluation time for pacing
        avg_eval_t = np.mean(eval_times) if eval_times else 1e-6
        if avg_eval_t < 1e-9: avg_eval_t = 1e-9
        
        # Eigen Adaptation Configuration
        # Only apply on low/medium dimensions to avoid SVD overhead
        eigen_enabled = (dim <= 60)
        
        # --- L-SHADE Evolution Loop ---
        while get_remaining_time() > 0.02:
            
            # 2. Time & Progress Management
            rem_time = get_remaining_time()
            max_evals_remain = rem_time / avg_eval_t
            
            # Heuristic progress (0.0 to 1.0) for this session
            total_est_evals = evals_in_session + max_evals_remain
            progress = evals_in_session / total_est_evals if total_est_evals > 0 else 1.0
            if progress > 1.0: progress = 1.0
            
            # 3. Linear Population Size Reduction (LPSR)
            target_pop = int(round(pop_size_init + (pop_size_end - pop_size_init) * progress))
            target_pop = max(pop_size_end, target_pop)
            
            if current_pop_size > target_pop:
                # Truncate population (worst individuals are at the end after sort)
                current_pop_size = target_pop
                pop = pop[:current_pop_size]
                fitness = fitness[:current_pop_size]
                
                # Resize archive
                target_arc = int(current_pop_size * arc_rate)
                if len(archive) > target_arc:
                    # Randomly remove
                    idxs = np.random.choice(len(archive), target_arc, replace=False)
                    archive = [archive[x] for x in idxs]

            # 4. Adaptive Parameter Generation
            # Dynamic p-best size (jSO strategy: 0.25 -> 0.05)
            p_val = 0.25 - (0.20 * progress)
            p_count = int(max(2, p_val * current_pop_size))
            
            # Select Memory
            r_idx = np.random.randint(0, H, current_pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F (Cauchy distribution)
            f = np.zeros(current_pop_size)
            for k in range(current_pop_size):
                while True:
                    val_f = m_f[k] + 0.1 * np.random.standard_cauchy()
                    if val_f > 0:
                        f[k] = min(1.0, val_f)
                        break
            
            # 5. Eigen Coordinate System (Rotation)
            do_eigen = False
            rot_matrix = None
            # Activate with 50% probability if enabled and population is large enough
            if eigen_enabled and np.random.rand() < 0.5 and current_pop_size > dim:
                # Compute covariance of top 50% individuals
                top_k = max(dim, int(current_pop_size * 0.5))
                cov = np.cov(pop[:top_k].T)
                try:
                    # SVD to get rotation matrix U
                    u, _, _ = np.linalg.svd(cov)
                    rot_matrix = u
                    do_eigen = True
                except:
                    pass
            
            # 6. Mutation: current-to-pbest/1
            pbest_idxs = np.random.randint(0, p_count, current_pop_size)
            x_pbest = pop[pbest_idxs]
            
            r1 = np.random.randint(0, current_pop_size, current_pop_size)
            # Ensure r1 != i
            mask_r1 = (r1 == np.arange(current_pop_size))
            r1[mask_r1] = (r1[mask_r1] + 1) % current_pop_size
            x_r1 = pop[r1]
            
            # r2 from Union(Population, Archive)
            if len(archive) > 0:
                pop_all = np.vstack((pop, np.array(archive)))
            else:
                pop_all = pop
            r2 = np.random.randint(0, len(pop_all), current_pop_size)
            x_r2 = pop_all[r2]
            
            # Compute Mutant V
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 7. Crossover (Binomial)
            mask_cr = np.random.rand(current_pop_size, dim) < cr[:, None]
            j_rand = np.random.randint(0, dim, current_pop_size)
            mask_cr[np.arange(current_pop_size), j_rand] = True
            
            if do_eigen:
                # Rotate Population and Mutant to Eigen-space
                pop_rot = np.dot(pop, rot_matrix)
                mut_rot = np.dot(mutant, rot_matrix)
                # Crossover in Eigen-space
                trial_rot = np.where(mask_cr, mut_rot, pop_rot)
                # Rotate back to Standard-space
                trial = np.dot(trial_rot, rot_matrix.T)
            else:
                trial = np.where(mask_cr, mutant, pop)
            
            # 8. Boundary Handling (Weighted Center)
            # Prevents stacking at bounds by placing violation halfway to bound
            below = trial < min_b
            if np.any(below):
                # Use pop (parent) as the reference for old position
                # trial = (min + old) / 2
                trial[below] = (min_b[np.where(below)[1]] + pop[below]) / 2.0
            
            above = trial > max_b
            if np.any(above):
                trial[above] = (max_b[np.where(above)[1]] + pop[above]) / 2.0
                
            trial = np.clip(trial, min_b, max_b)
            
            # 9. Selection & Evaluation
            success_f = []
            success_cr = []
            success_diff = []
            
            for i in range(current_pop_size):
                if get_remaining_time() <= 0: return best_fitness
                
                t0 = time.time()
                new_val = func(trial[i])
                eval_times.append(time.time() - t0)
                evals_in_session += 1
                
                # Update Global Best
                if new_val < best_fitness:
                    best_fitness = new_val
                    best_sol = trial[i].copy()
                
                # Selection
                if new_val < fitness[i]:
                    success_diff.append(fitness[i] - new_val)
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    
                    archive.append(pop[i].copy())
                    fitness[i] = new_val
                    pop[i] = trial[i]
                elif new_val == fitness[i]:
                    # Accept equal fitness but don't record success
                    pop[i] = trial[i]

            # 10. Update Memory (Weighted Lehmer Mean)
            if len(success_diff) > 0:
                weights = np.array(success_diff)
                weights /= np.sum(weights)
                
                sf = np.array(success_f)
                scr = np.array(success_cr)
                
                mean_cr = np.sum(weights * scr)
                mem_cr[k_mem] = mean_cr
                
                num_f = np.sum(weights * (sf**2))
                den_f = np.sum(weights * sf)
                mem_f[k_mem] = num_f / den_f if den_f > 0 else 0.5
                
                k_mem = (k_mem + 1) % H
                
            # 11. Sort Population
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Update average eval time
            if len(eval_times) > 50:
                avg_eval_t = np.mean(eval_times[-50:])
            else:
                avg_eval_t = np.mean(eval_times)
                
            # 12. Convergence Check (for restart)
            if np.std(fitness) < 1e-9:
                break
                
        # --- End of Session ---
        # If time permits, polish the best solution found
        rem = get_remaining_time()
        if rem > 0.1 and best_sol is not None:
            # Allocate a small budget to polish
            polish_budget = min(rem, 0.5)
            best_sol, best_fitness = mts_ls1(best_sol, best_fitness, polish_budget)

    return best_fitness
