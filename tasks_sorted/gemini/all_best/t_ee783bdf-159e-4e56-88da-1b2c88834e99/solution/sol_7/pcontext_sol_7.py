#To improve upon the best result (4.818), I propose an **Adaptive Multi-Start L-SHADE with Local Search**.
#
#**Rationale:**
#The previous best result was achieved using a restarted SHADE algorithm. Standard L-SHADE (Linear Population Size Reduction) excels at converging to a solution but can commit too early to a specific basin of attraction. By wrapping L-SHADE in a time-aware restart loop, we get the best of both worlds:
#1.  **L-SHADE Cycle**: Each cycle uses Linear Population Size Reduction (LPSR) driven by the *remaining time for that cycle* to force convergence.
#2.  **Basin Hopping**: If a cycle converges (stagnation) or completes its reduction schedule while time remains, a new cycle begins. This allows the algorithm to explore different basins.
#3.  **Elitism**: The global best solution is carried over to the next cycle (Soft Restart) to ensure we never lose ground.
#4.  **Local Polish**: A lightweight Coordinate Descent is applied at the end of cycles to refine the best solution found, exploiting the lack of gradient noise.
#5.  **Midpoint Bound Handling**: Preserves search direction near boundaries, superior to clipping.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Adaptive Multi-Start L-SHADE with Local Search.
    
    Splits the runtime into dynamic 'cycles'. Each cycle runs an L-SHADE optimization
    that scales its population reduction schedule to the remaining time.
    If stagnation occurs or the population shrinks to minimum, a restart is triggered.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Global Best Tracking ---
    global_best_fit = float('inf')
    global_best_sol = None

    # --- Precompute Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # --- Helper: Midpoint Target Bound Handling ---
    def apply_bounds(trial, old):
        # Lower bound violations
        mask_l = trial < min_b
        if np.any(mask_l):
            trial[mask_l] = (old[mask_l] + min_b[mask_l]) / 2.0
            
        # Upper bound violations
        mask_h = trial > max_b
        if np.any(mask_h):
            trial[mask_h] = (old[mask_h] + max_b[mask_h]) / 2.0
        return trial

    # --- Helper: Coordinate Descent Local Search ---
    def local_search(solution, current_fit, end_time):
        x = solution.copy()
        fit = current_fit
        d = len(x)
        
        # Step size based on domain scale (1% initially)
        steps = (max_b - min_b) * 0.01
        
        # Perform limited passes
        for _ in range(2): 
            if datetime.now() >= end_time: break
            improved = False
            
            for i in range(d):
                if datetime.now() >= end_time: break
                
                old_val = x[i]
                
                # Try positive step
                x[i] = np.clip(old_val + steps[i], min_b[i], max_b[i])
                f_new = func(x)
                if f_new < fit:
                    fit = f_new
                    improved = True
                    continue
                
                # Try negative step
                x[i] = np.clip(old_val - steps[i], min_b[i], max_b[i])
                f_new = func(x)
                if f_new < fit:
                    fit = f_new
                    improved = True
                    continue
                
                # Revert
                x[i] = old_val
                
            if not improved:
                steps *= 0.5  # Refine step size
                
        return x, fit

    # --- Main Optimization Loop (Restarts) ---
    while (datetime.now() - start_time) < time_limit:
        
        # Calculate budget for this cycle
        current_now = datetime.now()
        elapsed = (current_now - start_time).total_seconds()
        remaining = max_time - elapsed
        
        # If very little time remains, polish best and exit
        if remaining < 0.2:
            if global_best_sol is not None:
                # Use whatever tiny time is left to polish
                end_t = start_time + time_limit
                sol, fit = local_search(global_best_sol, global_best_fit, end_t)
                if fit < global_best_fit:
                    global_best_fit = fit
            return global_best_fit

        # Cycle setup
        cycle_start = current_now
        cycle_duration = max(0.5, remaining - 0.1) # Reserve 0.1s buffer
        
        # Parameters
        # Population: Adaptive to Dim, clamped
        initial_pop_size = int(np.clip(18 * dim, 30, 150))
        min_pop_size = 4
        
        pop_size = initial_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
        fitness = np.full(pop_size, float('inf'))
        
        # Soft Restart: Inject global best into new population
        if global_best_sol is not None:
            pop[0] = global_best_sol.copy()
            fitness[0] = global_best_fit
        
        # Evaluate Initial Population (skip index 0 if injected)
        start_idx = 1 if global_best_sol is not None else 0
        for i in range(start_idx, pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return global_best_fit
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = pop[i].copy()

        # SHADE Memory
        H = 6
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # --- Inner L-SHADE Cycle ---
        while True:
            t_now = datetime.now()
            # Check global timeout
            if (t_now - start_time) >= time_limit:
                return global_best_fit
            
            # Calculate Cycle Progress
            c_elapsed = (t_now - cycle_start).total_seconds()
            progress = c_elapsed / cycle_duration
            
            if progress >= 1.0:
                break # Cycle complete
            
            # 1. Linear Population Size Reduction (LPSR)
            plan_pop = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * progress))
            plan_pop = max(min_pop_size, plan_pop)
            
            if pop_size > plan_pop:
                # Sort and Reduce
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:plan_pop]]
                fitness = fitness[sort_idx[:plan_pop]]
                pop_size = plan_pop
                
                # Resize Archive
                arc_limit = int(pop_size * 2.0)
                while len(archive) > arc_limit:
                    archive.pop(np.random.randint(0, len(archive)))
            
            # Stagnation Check: If converged, break to restart
            if pop_size < 20:
                if np.std(fitness) < 1e-8:
                    break
            
            # 2. Parameter Generation
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # CR: Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F: Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                f[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
            f = np.clip(f, 0, 1)
            
            # 3. Mutation: current-to-pbest/1
            # Dynamic p (jSO strategy): Linear decrease from 0.2 to 2/N
            p_max = 0.2
            p_min = 2.0 / pop_size
            p_val = p_max - (p_max - p_min) * progress
            p_val = max(p_min, p_val)
            
            # Identify p-best
            sorted_idx = np.argsort(fitness)
            cnt_pbest = int(max(2, p_val * pop_size))
            pbest_pool = sorted_idx[:cnt_pbest]
            
            pbest_idx = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Identify r1 != i
            r1_idx = np.random.randint(0, pop_size, pop_size)
            hit_self = r1_idx == np.arange(pop_size)
            r1_idx[hit_self] = (r1_idx[hit_self] + 1) % pop_size
            x_r1 = pop[r1_idx]
            
            # Identify r2 != i, != r1 (from Union)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            
            r2_idx = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant
            f_col = f.reshape(-1, 1)
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask_c = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
            mask_c[np.arange(pop_size), j_rand] = True
            trial = np.where(mask_c, mutant, pop)
            
            # 5. Bound Handling (Midpoint)
            for i in range(pop_size):
                trial[i] = apply_bounds(trial[i], pop[i])
            
            # 6. Evaluation & Update
            succ_cr = []
            succ_f = []
            diff_fit = []
            
            new_pop = pop.copy()
            new_fit = fitness.copy()
            
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return global_best_fit
                
                val = func(trial[i])
                
                if val <= fitness[i]:
                    if val < fitness[i]:
                        archive.append(pop[i].copy())
                        succ_cr.append(cr[i])
                        succ_f.append(f[i])
                        diff_fit.append(fitness[i] - val)
                    
                    new_pop[i] = trial[i]
                    new_fit[i] = val
                    
                    if val < global_best_fit:
                        global_best_fit = val
                        global_best_sol = trial[i].copy()
                        
            pop = new_pop
            fitness = new_fit
            
            # Archive Limit
            while len(archive) > int(pop_size * 2.0):
                archive.pop(np.random.randint(0, len(archive)))
                
            # 7. Update Memory (Weighted Lehmer Mean)
            if len(diff_fit) > 0:
                w = np.array(diff_fit)
                w = w / np.sum(w)
                
                scr = np.array(succ_cr)
                sf = np.array(succ_f)
                
                m_cr_new = np.sum(w * scr)
                
                num = np.sum(w * (sf**2))
                den = np.sum(w * sf)
                m_f_new = num / den if den > 0 else 0.5
                
                mem_cr[k_mem] = m_cr_new
                mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                k_mem = (k_mem + 1) % H
        
        # --- End of Cycle Polish ---
        # Before restarting, polish the best found
        if global_best_sol is not None:
            # Budget: 5% of max_time or 0.5s max
            budget = min(0.5, max_time * 0.05)
            ls_end = datetime.now() + timedelta(seconds=budget)
            if ls_end > (start_time + time_limit):
                ls_end = start_time + time_limit
                
            sol, fit = local_search(global_best_sol, global_best_fit, ls_end)
            if fit < global_best_fit:
                global_best_fit = fit
                global_best_sol = sol
                
    return global_best_fit
