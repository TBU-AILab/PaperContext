#An algorithm for minimizing a function within a limited time frame, utilizing **L-SHADE with Restarts (Linear Population Size Reduction Success-History Adaptive Differential Evolution)**.
#
#### Algorithm Description
#1.  **L-SHADE Core**: The algorithm uses the Success-History Adaptive Differential Evolution (SHADE) method, which automatically adapts the scaling factor ($F$) and crossover rate ($CR$) based on the success of past mutations.
#2.  **Linear Population Size Reduction (LPSR)**: The population size is linearly reduced from an initial size ($N_{init}$) to a minimum size ($N_{min}$) over the course of the execution. This allows for exploration (high diversity) in the early stages and exploitation (fast convergence) in the later stages.
#3.  **Dynamic Budget Estimation**: Since the limit is defined by time (`max_time`) rather than function evaluations, the algorithm continuously estimates the remaining number of possible evaluations based on the average execution speed, adjusting the LPSR schedule dynamically.
#4.  **Restart Mechanism**: If the population converges (low variance) before the time runs out, the algorithm triggers a restart. It retains the best global solution but re-initializes the population to escape local optima, continuing the search with the remaining time budget.
#5.  **Bound Handling**: A reflection method is used (`if x < lb: x = lb + (lb - x)`) instead of simple clipping. This preserves the distributional properties of the mutation near boundaries.
#
#### Python Implementation
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Restarts (Linear Population Size Reduction
    Success-History Adaptive Differential Evolution).
    
    This algorithm adapts parameters (F, CR) and population size dynamically to 
    maximize performance within the given time limit.
    """
    start_time = datetime.now()
    # Reserve a small buffer to ensure safe return
    time_limit = timedelta(seconds=max_time - 0.1)

    def has_time():
        return (datetime.now() - start_time) < time_limit

    # --- Pre-processing ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    bound_diff = ub - lb

    # --- Configuration ---
    # Initial population size: 18 * dim (standard for L-SHADE)
    n_init = int(18 * dim)
    # Minimum population size
    n_min = 4
    # History memory size
    H = 6
    
    # Global best tracking
    best_fitness = float('inf')
    best_sol = None
    
    # Initial speed estimation
    # Perform one dummy evaluation to estimate function cost
    t0 = datetime.now()
    func(lb + bound_diff * 0.5)
    t1 = datetime.now()
    eval_time = (t1 - t0).total_seconds()
    if eval_time < 1e-6: eval_time = 1e-6
    
    # --- Main Loop (Restarts) ---
    while has_time():
        
        # 1. Initialize for this restart
        pop_size = n_init
        pop = lb + np.random.rand(pop_size, dim) * bound_diff
        fitness = np.zeros(pop_size)
        
        # Evaluate initial population
        for i in range(pop_size):
            if not has_time(): return best_fitness
            val = func(pop[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
                best_sol = pop[i].copy()
                
        # Initialize Memory (M_F, M_CR) to 0.5
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        # We allocate max space but track valid count with arc_cnt
        archive = np.zeros((n_init, dim))
        arc_cnt = 0
        
        # Restart specific timers
        restart_start = datetime.now()
        fe_count = pop_size 
        
        # --- Generation Loop ---
        while has_time():
            
            # 2. Time & Population Management (LPSR)
            now = datetime.now()
            elapsed_run = (now - restart_start).total_seconds()
            rem_global = (time_limit - (now - start_time)).total_seconds()
            
            if rem_global <= 0: return best_fitness
            
            # Estimate remaining FEs based on current speed
            speed = elapsed_run / fe_count if fe_count > 0 else eval_time
            est_rem_fes = rem_global / speed
            max_fes_run = fe_count + est_rem_fes
            
            # Calculate Progress (0.0 to 1.0)
            progress = fe_count / max_fes_run if max_fes_run > 0 else 1.0
            progress = min(1.0, progress)
            
            # Calculate Target Population Size
            n_target = int(round((n_min - n_init) * progress + n_init))
            n_target = max(n_min, n_target)
            
            # Reduce Population if needed
            if pop_size > n_target:
                # Sort by fitness (best first)
                sort_idx = np.argsort(fitness)
                keep_idx = sort_idx[:n_target]
                
                pop = pop[keep_idx]
                fitness = fitness[keep_idx]
                pop_size = n_target
                
                # Resize Archive (Capacity tracks pop_size)
                if arc_cnt > pop_size:
                    # Randomly select elements to keep
                    keep_mask = np.random.choice(arc_cnt, pop_size, replace=False)
                    new_arc = archive[keep_mask]
                    archive[:pop_size] = new_arc
                    arc_cnt = pop_size

            # 3. Convergence Check
            # If population is extremely flat, restart to explore new areas
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break
            
            # 4. Generate Parameters
            # Sort population for p-best selection
            sorted_idx = np.argsort(fitness)
            pop_sorted = pop[sorted_idx]
            
            # Select memory slots
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate F (Cauchy Distribution)
            F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Retry if F <= 0, Clip if F > 1
            while np.any(F <= 0):
                mask = F <= 0
                F[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
            F = np.minimum(F, 1.0)
            
            # Generate CR (Normal Distribution)
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # 5. Mutation (current-to-pbest/1)
            # p decreases linearly? Standard is constant or small range.
            # We use a robust dynamic p-value logic.
            p_val = max(2.0/pop_size, 0.11)
            n_pbest = max(2, int(pop_size * p_val))
            
            # x_pbest
            pbest_indices = np.random.randint(0, n_pbest, pop_size)
            x_pbest = pop_sorted[pbest_indices]
            
            # x_r1 (Random from pop, distinct from i)
            r1 = np.random.randint(0, pop_size, pop_size)
            r1 = (r1 + np.arange(pop_size) + 1) % pop_size
            x_r1 = pop[r1]
            
            # x_r2 (Random from Union(Pop, Archive))
            total_pool = pop_size + arc_cnt
            r2 = np.random.randint(0, total_pool, pop_size)
            
            x_r2 = np.zeros((pop_size, dim))
            mask_pop = r2 < pop_size
            mask_arc = ~mask_pop
            
            x_r2[mask_pop] = pop[r2[mask_pop]]
            if arc_cnt > 0 and np.any(mask_arc):
                idx_arc = r2[mask_arc] - pop_size
                x_r2[mask_arc] = archive[idx_arc]
            elif np.any(mask_arc):
                x_r2[mask_arc] = pop[0] # Fallback
                
            # Compute Mutant V
            mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
            
            # 6. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask = np.random.rand(pop_size, dim) < CR[:, None]
            mask[np.arange(pop_size), j_rand] = True
            trial = np.where(mask, mutant, pop)
            
            # 7. Bound Handling (Reflection)
            # If x < lb, x = lb + (lb - x)
            mask_l = trial < lb
            trial[mask_l] = lb[mask_l] + (lb[mask_l] - trial[mask_l])
            mask_u = trial > ub
            trial[mask_u] = ub[mask_u] - (trial[mask_u] - ub[mask_u])
            # Clip final result to be safe
            trial = np.clip(trial, lb, ub)
            
            # 8. Selection & Updates
            success_diff = []
            success_f = []
            success_cr = []
            
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            for k in range(pop_size):
                if not has_time(): return best_fitness
                
                f_trial = func(trial[k])
                fe_count += 1
                
                if f_trial < fitness[k]:
                    # Successful Update
                    new_pop[k] = trial[k]
                    new_fitness[k] = f_trial
                    
                    # Update Archive
                    if arc_cnt < pop_size:
                        archive[arc_cnt] = pop[k].copy()
                        arc_cnt += 1
                    else:
                        # Replace random element
                        rep_idx = np.random.randint(0, pop_size)
                        archive[rep_idx] = pop[k].copy()
                    
                    # Store success info for parameter adaptation
                    diff = fitness[k] - f_trial
                    success_diff.append(diff)
                    success_f.append(F[k])
                    success_cr.append(CR[k])
                    
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_sol = trial[k].copy()
            
            pop = new_pop
            fitness = new_fitness
            
            # 9. Update Historical Memory
            if len(success_diff) > 0:
                s_diff = np.array(success_diff)
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                
                # Weights based on improvement magnitude
                w = s_diff / np.sum(s_diff)
                
                # Update M_CR (Weighted Mean)
                m_cr_new = np.sum(w * s_cr)
                mem_cr[k_mem] = np.clip(m_cr_new, 0, 1)
                
                # Update M_F (Weighted Lehmer Mean)
                num = np.sum(w * (s_f**2))
                den = np.sum(w * s_f)
                m_f_new = num / den if den > 0 else 0.5
                mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                
                k_mem = (k_mem + 1) % H

    return best_fitness
