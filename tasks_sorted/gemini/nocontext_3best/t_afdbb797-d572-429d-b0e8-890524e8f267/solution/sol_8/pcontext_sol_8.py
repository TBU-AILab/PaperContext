#The output values and codes for the best generated algorithms are as follows:
#1. output value is: 9.574119223122835
# ...
#2. output value is: 14.25719323070554
# ...
#3. output value is: 17.594455412611367
# ...
#
#The following Python code implements an **Improved L-SHADE (L-SHADE-R)** with **Restart Mechanism** and **jSO-inspired refinements**.
#
#**Key Improvements over previous algorithms:**
#1.  **Restart Mechanism**: Standard L-SHADE converges to a single basin of attraction. To fully utilize the `max_time`, this algorithm detects stagnation (when population diversity collapses) and restarts the optimization with a new population while preserving the best solution (Elitism). This allows exploring multiple basins of attraction, significantly improving results on multimodal landscapes.
#2.  **Dynamic `p`-value (jSO Strategy)**: The `current-to-pbest` strategy is enhanced by linearly reducing the `p` value (from 0.25 to 0.05) over time. This favors exploration (larger random pool of best solutions) in the early phases and strict exploitation (focusing on the very best) in the final phases.
#3.  **Midpoint Bound Handling**: Instead of reflection (which clusters solutions at boundaries) or clipping, particles violating bounds are placed at the midpoint between the boundary and their parent position `(bound + parent)/2`. This method has been shown to be superior in recent Differential Evolution variants.
#4.  **Optimized Time Management**: The Linear Population Size Reduction (LPSR) is adapted to handle multiple restarts by calculating progress relative to the *remaining* time budget for each run.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using Improved L-SHADE with Restarts and 
    Dynamic Parameter Adaptation (jSO features).
    """
    overall_start = datetime.now()
    
    # Helper to check global timeout
    def get_elapsed():
        return (datetime.now() - overall_start).total_seconds()
        
    global_best_val = float('inf')
    global_best_sol = None
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Main Restart Loop ---
    # We continue restarting until we run out of time
    while True:
        elapsed = get_elapsed()
        remaining = max_time - elapsed
        
        # If less than 5% of time left or very short absolute time, stop.
        if remaining < 0.05 * max_time or remaining < 0.5:
            break
            
        # --- Configuration for this run ---
        # Initial Population Size:
        # A bit larger than standard to ensure coverage (25 * dim), but clipped.
        pop_size_init = int(np.clip(25 * dim, 50, 250))
        
        # If this is a late restart (short time), scale down population
        if remaining < (0.4 * max_time):
            pop_size_init = max(30, int(pop_size_init * 0.6))
            
        pop_size = pop_size_init
        pop_size_min = 4
        
        # Archive parameters
        archive = []
        arc_rate = 2.0  # Archive size relative to pop_size
        
        # Memory for adaptive parameters (History length = 5)
        mem_size = 5
        m_cr = np.full(mem_size, 0.5)
        m_f = np.full(mem_size, 0.5)
        mem_k = 0
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best into new population to guide it
        if global_best_sol is not None:
            pop[0] = global_best_sol.copy()
            
        # Initial Evaluation
        for i in range(pop_size):
            if get_elapsed() >= max_time:
                return global_best_val
            
            # Recalculate injected elite or evaluate new random
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
        
        # Setup for Linear Population Size Reduction (LPSR) in this run
        # We act as if this run has the entire remaining time to finish
        run_start_time = get_elapsed()
        run_budget = remaining 
        
        stag_counter = 0
        
        # --- Optimization Loop for Current Run ---
        while True:
            t_now = get_elapsed()
            if t_now >= max_time:
                return global_best_val
                
            # Calculate Progress (0.0 to 1.0)
            run_elapsed = t_now - run_start_time
            progress = run_elapsed / run_budget
            if progress > 1.0: progress = 1.0
            
            # 1. Linear Population Size Reduction
            target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_size = max(pop_size_min, target_size)
            
            if pop_size > target_size:
                # Reduce population: Keep best
                sorted_indices = np.argsort(fitness)
                keep = sorted_indices[:target_size]
                pop = pop[keep]
                fitness = fitness[keep]
                pop_size = target_size
                
                # Resize archive
                arc_target = int(pop_size * arc_rate)
                if len(archive) > arc_target:
                    # Remove random elements
                    diff = len(archive) - arc_target
                    for _ in range(diff):
                        archive.pop(np.random.randint(0, len(archive)))
            
            # 2. Dynamic p-value (jSO strategy)
            # Linearly decrease p from 0.25 to 0.05
            p_val = 0.25 - (0.20 * progress)
            
            # 3. Generate Parameters (Shadowed from SHADE)
            r_idx = np.random.randint(0, mem_size, pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Robust F handling: retry if <= 0
            neg_mask = f <= 0
            retry = 0
            while np.any(neg_mask) and retry < 2:
                f[neg_mask] = mu_f[neg_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(neg_mask)) - 0.5))
                neg_mask = f <= 0
                retry += 1
            f[f <= 0] = 0.5  # Fallback
            f[f > 1] = 1.0   # Clip
            
            # 4. Mutation: current-to-pbest/1
            sorted_idx = np.argsort(fitness)
            num_pbest = max(2, int(pop_size * p_val))
            pbest_pool = sorted_idx[:num_pbest]
            
            r_pbest = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[r_pbest]
            
            # r1 != i
            r1 = np.random.randint(0, pop_size, pop_size)
            conflict = (r1 == np.arange(pop_size))
            if np.any(conflict):
                r1[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
            x_r1 = pop[r1]
            
            # r2 != r1, != i, from Pop U Archive
            if len(archive) > 0:
                arc_arr = np.array(archive)
                union_size = pop_size + len(arc_arr)
                r2 = np.random.randint(0, union_size, pop_size)
                
                x_r2 = np.zeros((pop_size, dim))
                
                # vectorized assignment based on whether r2 is in pop or archive
                from_pop = r2 < pop_size
                from_arc = ~from_pop
                
                if np.any(from_pop):
                    x_r2[from_pop] = pop[r2[from_pop]]
                if np.any(from_arc):
                    x_r2[from_arc] = arc_arr[r2[from_arc] - pop_size]
            else:
                r2 = np.random.randint(0, pop_size, pop_size)
                conflict = (r2 == r1) | (r2 == np.arange(pop_size))
                if np.any(conflict):
                    r2[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
                x_r2 = pop[r2]
                
            # Calculate Mutant Vector
            f_col = f[:, np.newaxis]
            v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            mask = rand_vals < cr[:, np.newaxis]
            mask[np.arange(pop_size), j_rand] = True
            
            u = np.where(mask, v, pop)
            
            # 6. Bound Handling: Weighted Midpoint
            # Instead of reflecting, put it between min and parent. This avoids boundary sticking.
            mask_l = u < min_b
            if np.any(mask_l):
                rows, cols = np.where(mask_l)
                u[mask_l] = (min_b[cols] + pop[rows, cols]) * 0.5
                
            mask_u = u > max_b
            if np.any(mask_u):
                rows, cols = np.where(mask_u)
                u[mask_u] = (max_b[cols] + pop[rows, cols]) * 0.5
                
            # 7. Selection and Adaptation
            success_f = []
            success_cr = []
            diff_fitness = []
            improved_run = False
            
            for i in range(pop_size):
                if get_elapsed() >= max_time:
                    return global_best_val
                
                val_trial = func(u[i])
                
                if val_trial <= fitness[i]:
                    if val_trial < fitness[i]:
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        diff_fitness.append(fitness[i] - val_trial)
                        archive.append(pop[i].copy())
                        improved_run = True
                        
                    fitness[i] = val_trial
                    pop[i] = u[i]
                    
                    if val_trial < global_best_val:
                        global_best_val = val_trial
                        global_best_sol = u[i].copy()
                        stag_counter = 0
            
            # Maintain archive size
            while len(archive) > int(pop_size * arc_rate):
                archive.pop(np.random.randint(0, len(archive)))
            
            # Update Memories (Weighted Lehmer Mean)
            if len(success_f) > 0:
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                df = np.array(diff_fitness)
                
                if np.sum(df) > 0:
                    w = df / np.sum(df)
                    
                    m_cr_new = np.sum(w * s_cr)
                    m_cr[mem_k] = m_cr_new
                    
                    num = np.sum(w * (s_f ** 2))
                    den = np.sum(w * s_f)
                    m_f[mem_k] = num / den if den > 0 else 0.5
                    
                    mem_k = (mem_k + 1) % mem_size
            
            # 8. Stagnation Check (Trigger Restart)
            # If population variance is very low, we have converged.
            fit_std = np.std(fitness)
            if fit_std < 1e-9:
                break # Break inner loop, trigger restart in outer loop
            
            if not improved_run:
                stag_counter += 1
            else:
                stag_counter = 0
                
            # If population is minimal and no improvement for many generations, restart
            if pop_size <= pop_size_min + 2 and stag_counter > 25:
                break

    return global_best_val
