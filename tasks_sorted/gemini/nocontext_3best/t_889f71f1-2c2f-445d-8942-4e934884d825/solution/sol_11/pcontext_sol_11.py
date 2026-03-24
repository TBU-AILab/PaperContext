#The following algorithm implements an optimized **L-SHADE-RSP (L-SHADE with Restart, Smart Population resizing, and Polish)**.
#
#**Key Improvements:**
#1.  **L-SHADE with LPSR:** Utilizes the state-of-the-art Success-History Adaptive Differential Evolution algorithm with Linear Population Size Reduction. This allows broad exploration initially and focused exploitation as the population shrinks.
#2.  **Adaptive $p$-value (jSO strategy):** The parameter $p$ (controlling the greediness of the mutation strategy) adapts linearly from exploration ($p=0.2$) to exploitation ($p=0.05$) over the estimated progress, improving convergence speed.
#3.  **Coordinate Descent Polish (MTS-LS1):** Instead of a random Gaussian walk, the final polish phase uses a strict coordinate descent search (based on MTS-LS1). This is significantly more efficient for local refinement, especially in high-precision scenarios.
#4.  **Robust Time Estimation:** The algorithm dynamically estimates the cost of function evaluations to pace the population reduction perfectly within `max_time`, ensuring strict adherence to the time limit while maximizing iteration count.
#
import numpy as np
from datetime import datetime

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-RSP:
    L-SHADE with Linear Population Size Reduction, Restart strategy, 
    and MTS-LS1 Coordinate Descent Polish.
    """
    start_time = datetime.now()
    
    # --- Helper Functions ---
    def get_remaining_seconds():
        return max_time - (datetime.now() - start_time).total_seconds()

    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global state
    best_fitness = float('inf')
    best_sol = None
    
    # Evaluation statistics
    eval_count = 0
    eval_times = [] # Keep a window of recent evaluation times
    
    # Wrapper for function evaluation to handle timing and stats
    def evaluate(x):
        nonlocal eval_count, best_fitness, best_sol
        
        t0 = datetime.now()
        val = func(x)
        dt = (datetime.now() - t0).total_seconds()
        
        # Update stats
        if len(eval_times) > 100: 
            eval_times.pop(0) # Keep rolling window small
        eval_times.append(dt)
        eval_count += 1
        
        # Update global best
        if val < best_fitness:
            best_fitness = val
            best_sol = x.copy()
            
        return val

    # --- Algorithm Parameters ---
    # Population size: Start large for exploration, capped for efficiency
    pop_size_init = int(np.clip(20 * dim, 40, 200))
    pop_size_min = 4
    
    # Memory for SHADE
    H = 5
    
    # --- Main Loop (Restarts) ---
    while get_remaining_seconds() > 0.1:
        
        # Reset population size and memory for new restart
        current_pop_size = pop_size_init
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Population
        pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
        fitness = np.full(current_pop_size, float('inf'))
        
        # Elitism: Inject best solution found so far into the new population
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol
            fitness[0] = best_fitness
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, current_pop_size):
            if get_remaining_seconds() <= 0: return best_fitness
            fitness[i] = evaluate(pop[i])
            
        # Archive for diversity maintenance
        archive = []
        
        # Sort population
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # --- L-SHADE Evolution Loop ---
        while True:
            # 1. Check Termination & Polish Condition
            remaining = get_remaining_seconds()
            if remaining <= 0: return best_fitness
            
            # Switch to Polish if time is running low (last 5% or < 2 seconds)
            if remaining < min(2.0, max_time * 0.05):
                break
                
            # 2. Time Management & LPSR
            avg_eval = np.mean(eval_times) if eval_times else 1e-6
            if avg_eval == 0: avg_eval = 1e-6
            
            # Estimate progress based on time
            max_evals_total = int(max_time / avg_eval)
            current_evals_est = eval_count # Approximation
            progress = min(1.0, eval_count / max_evals_total) if max_evals_total > 0 else 1.0
            
            # Linear Population Size Reduction
            target_pop = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_pop = max(pop_size_min, target_pop)
            
            if current_pop_size > target_pop:
                current_pop_size = target_pop
                # Truncate worst (pop is sorted at end of loop)
                pop = pop[:current_pop_size]
                fitness = fitness[:current_pop_size]
                
                # Resize Archive
                target_arc = int(current_pop_size * 2.0)
                if len(archive) > target_arc:
                    # Randomly remove to fit
                    del_cnt = len(archive) - target_arc
                    idxs = np.random.choice(len(archive), del_cnt, replace=False)
                    # Rebuild archive excluding deleted
                    archive = [archive[i] for i in range(len(archive)) if i not in idxs]

            # 3. Parameter Generation
            r_idx = np.random.randint(0, H, current_pop_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]
            
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            f_params = mu_f + 0.1 * np.random.standard_cauchy(current_pop_size)
            f_params = np.clip(f_params, 0, 1)
            f_params[f_params < 0.05] = 0.05 # Clamp lower bound
            
            # 4. Mutation: current-to-pbest/1
            # Dynamic p value (jSO strategy): Linear reduction from 0.2 to 0.05
            p_val = 0.15 * (1.0 - progress) + 0.05
            p_count = int(max(2, p_val * current_pop_size))
            
            pbest_indices = np.random.randint(0, p_count, current_pop_size)
            x_pbest = pop[pbest_indices]
            
            # r1 != i
            r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
            mask_same = (r1_indices == np.arange(current_pop_size))
            r1_indices[mask_same] = (r1_indices[mask_same] + 1) % current_pop_size
            x_r1 = pop[r1_indices]
            
            # r2 from Union(Pop, Archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            
            r2_indices = np.random.randint(0, len(union_pop), current_pop_size)
            x_r2 = union_pop[r2_indices]
            
            f_col = f_params[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 5. Crossover (Binomial)
            rand_cr = np.random.rand(current_pop_size, dim)
            cross_mask = rand_cr < cr[:, None]
            j_rand = np.random.randint(0, dim, current_pop_size)
            cross_mask[np.arange(current_pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 6. Selection
            success_f = []
            success_cr = []
            success_diff = []
            
            for i in range(current_pop_size):
                if get_remaining_seconds() <= 0: return best_fitness
                
                val = evaluate(trial[i])
                
                if val < fitness[i]:
                    success_diff.append(fitness[i] - val)
                    success_f.append(f_params[i])
                    success_cr.append(cr[i])
                    
                    archive.append(pop[i].copy())
                    fitness[i] = val
                    pop[i] = trial[i]
            
            # 7. Update Memory
            if success_diff:
                w = np.array(success_diff)
                w /= np.sum(w)
                
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                
                mean_f = np.sum(w * (s_f**2)) / np.sum(w * s_f)
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                
                mean_cr = np.sum(w * s_cr)
                mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                
                k_mem = (k_mem + 1) % H
            
            # 8. Sort and Check Convergence
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Restart if converged
            if np.std(fitness) < 1e-9:
                break
        
        # End of L-SHADE loop.
        # If we broke due to time for polish, exit outer loop
        if get_remaining_seconds() < min(2.0, max_time * 0.05):
            break

    # --- Final Polish: MTS-LS1 (Coordinate Descent) ---
    # Use remaining time to refine the global best
    if best_sol is not None:
        current_sol = best_sol.copy()
        current_fit = best_fitness
        
        # Initial Search Range (0.5 of domain)
        sr = (max_b - min_b) * 0.5
        
        while get_remaining_seconds() > 0:
            improved = False
            # Search dimensions in random order
            dims = np.random.permutation(dim)
            
            for i in dims:
                if get_remaining_seconds() <= 0: return best_fitness
                
                original_val = current_sol[i]
                
                # 1. Try decreasing
                current_sol[i] = np.clip(original_val - sr[i], min_b[i], max_b[i])
                val = evaluate(current_sol)
                
                if val < current_fit:
                    current_fit = val
                    improved = True
                else:
                    # 2. Try increasing (MTS strategy: +0.5 of sr relative to original)
                    # Effectively checking the other side with half step
                    current_sol[i] = np.clip(original_val + 0.5 * sr[i], min_b[i], max_b[i])
                    val = evaluate(current_sol)
                    
                    if val < current_fit:
                        current_fit = val
                        improved = True
                    else:
                        # 3. Restore
                        current_sol[i] = original_val
            
            if not improved:
                sr *= 0.5
                # Terminate polish if precision is extremely high
                if np.max(sr) < 1e-15:
                    break
                    
    return best_fitness
