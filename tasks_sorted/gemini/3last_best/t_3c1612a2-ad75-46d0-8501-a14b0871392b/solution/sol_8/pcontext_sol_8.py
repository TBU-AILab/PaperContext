#The following algorithm is an implementation of **L-SHADE-R (Linear Population Size Reduction SHADE with Restarts)**.
#
#### Algorithm Description
#This algorithm improves upon the previous SHADE implementation by introducing **Linear Population Size Reduction (LPSR)**, a technique used in state-of-the-art evolutionary algorithms (like L-SHADE-EpSin, a frequent winner of CEC competitions).
#
#1.  **Time-Based Budget Estimation**: Before starting, the algorithm runs a brief calibration to estimate the execution time of `func`. It uses this to calculate a precise "Evaluation Budget" ($MaxEvals$) that fits within the allotted time, preventing the algorithm from running out of time during a critical convergence phase.
#2.  **Linear Population Reduction**: The population size ($N$) starts large (exploration) and linearly decreases to a minimum size of 4 (exploitation) as the number of evaluations progresses towards the calculated budget.
#    *   $N_{g+1} = \text{round} \left( (N_{min} - N_{init}) \cdot \frac{CurrentEvals}{MaxEvals} + N_{init} \right)$
#    *   This forces the mutation strategy (`current-to-pbest`) to become increasingly greedy and focused, refining the solution significantly faster in the final stages.
#3.  **Dynamic Archive Resizing**: As the population shrinks, the external archive size is also reduced to match the population size, ensuring the diversity buffer doesn't overpower the convergence pressure.
#4.  **Restart Mechanism**: If the L-SHADE process converges early (population variance drops below threshold) or completes its reduction cycle while time remains, the algorithm triggers a restart. It keeps the global best found so far but resets the population and adaptive memories to explore new basins of attraction.
#
#### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # -------------------------------------------------------------------------
    # 1. Initialization and Timing
    # -------------------------------------------------------------------------
    start_time = datetime.now()
    # Reserve 5% buffer to guarantee return
    end_time = start_time + timedelta(seconds=max_time * 0.95)
    
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    bound_diff = ub - lb
    
    best_fitness = float('inf')
    global_best_sol = None

    # -------------------------------------------------------------------------
    # 2. Calibration: Estimate Function Cost
    # -------------------------------------------------------------------------
    # We perform a few evaluations to estimate how many we can fit in max_time.
    # This is crucial for L-SHADE's linear reduction schedule.
    calibration_evals = 5
    cal_start = datetime.now()
    for _ in range(calibration_evals):
        # Evaluate random point
        temp_sol = lb + np.random.rand(dim) * bound_diff
        f_val = func(temp_sol)
        if f_val < best_fitness:
            best_fitness = f_val
            global_best_sol = temp_sol
            
    cal_duration = (datetime.now() - cal_start).total_seconds()
    avg_time_per_eval = cal_duration / calibration_evals
    
    # Calculate total budget (evaluations) available in the remaining time
    time_remaining = max_time * 0.95 - cal_duration
    if time_remaining <= 0:
        return best_fitness
        
    estimated_total_evals = int(time_remaining / (avg_time_per_eval + 1e-10))
    
    # Heuristic: Limit the budget per restart to ensure we can restart if stuck.
    # For high dimensions or fast functions, huge budgets dilute the reduction pressure.
    # We cap "evals per run" to approx 2500 * dim (common in benchmarks) or the time limit.
    max_evals_per_run = min(estimated_total_evals, 2500 * dim)
    
    # -------------------------------------------------------------------------
    # 3. Algorithm Parameters
    # -------------------------------------------------------------------------
    # L-SHADE Constants
    initial_pop_size = int(18 * dim)
    initial_pop_size = max(30, min(300, initial_pop_size)) # Clamp
    min_pop_size = 4
    
    H_memory_size = 5
    
    # -------------------------------------------------------------------------
    # 4. Main Optimization Loop (Restarts)
    # -------------------------------------------------------------------------
    while True:
        # Check time before starting a new run
        if datetime.now() >= end_time:
            return best_fitness

        # Reset State for Restart
        pop_size = initial_pop_size
        max_evals = max_evals_per_run
        curr_evals = 0
        
        # Initialize Memory
        mem_cr = np.full(H_memory_size, 0.5)
        mem_f = np.full(H_memory_size, 0.5)
        k_mem = 0
        
        # Initialize Population
        pop = lb + np.random.rand(pop_size, dim) * bound_diff
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluation
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_fitness
            val = func(pop[i])
            fitness[i] = val
            curr_evals += 1
            if val < best_fitness:
                best_fitness = val
                global_best_sol = pop[i].copy()
                
        # Initialize Archive (stores inferior solutions)
        archive = np.empty((initial_pop_size, dim)) # Max capacity matches max N
        arc_count = 0
        
        # ---------------------------------------------------------------------
        # Evolution Loop (L-SHADE)
        # ---------------------------------------------------------------------
        while curr_evals < max_evals:
            if datetime.now() >= end_time: return best_fitness
            
            # --- Linear Population Size Reduction (LPSR) ---
            # Calculate target population size based on progress ratio
            eval_ratio = curr_evals / max_evals
            next_pop_size = int(round(((min_pop_size - initial_pop_size) * eval_ratio) + initial_pop_size))
            next_pop_size = max(min_pop_size, next_pop_size)
            
            # If reduction is needed:
            if pop_size > next_pop_size:
                # Sort population by fitness (worst at the end)
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx]
                fitness = fitness[sort_idx]
                
                # Truncate to new size
                pop = pop[:next_pop_size]
                fitness = fitness[:next_pop_size]
                
                # Resize Archive: Archive size shouldn't exceed population size in L-SHADE
                curr_arc_capacity = next_pop_size
                if arc_count > curr_arc_capacity:
                    # Randomly remove excess from archive to fit
                    keep_indices = np.random.choice(arc_count, curr_arc_capacity, replace=False)
                    archive[:curr_arc_capacity] = archive[keep_indices]
                    arc_count = curr_arc_capacity
                
                pop_size = next_pop_size
                
            # --- Parameter Generation ---
            r_idx = np.random.randint(0, H_memory_size, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Cauchy for F, Normal for CR
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F with retry for negative values
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # p-best selection (greedy)
            # p is typically linear from 0.1 to 0.2, or fixed. We use fixed 0.11 (top 11%)
            p_val = 0.11
            p_num = max(2, int(p_val * pop_size))
            sorted_indices = np.argsort(fitness)
            pbest_indices = sorted_indices[:p_num]
            pbest_vectors = pop[np.random.choice(pbest_indices, pop_size)]
            
            # r1 selection (distinct from i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            col_mask = (r1_idx == np.arange(pop_size))
            r1_idx[col_mask] = (r1_idx[col_mask] + 1) % pop_size
            x_r1 = pop[r1_idx]
            
            # r2 selection (distinct from i, r1, from Pop U Archive)
            union_size = pop_size + arc_count
            r2_idx = np.random.randint(0, union_size, pop_size)
            
            # Logic to fetch x_r2 from Pop or Archive
            x_r2 = np.empty((pop_size, dim))
            mask_pop = r2_idx < pop_size
            mask_arc = ~mask_pop
            
            x_r2[mask_pop] = pop[r2_idx[mask_pop]]
            if np.any(mask_arc):
                x_r2[mask_arc] = archive[r2_idx[mask_arc] - pop_size]
            
            # Compute Mutant
            # v = x + F * (pbest - x) + F * (r1 - r2)
            f_col = f[:, None]
            mutant = pop + f_col * (pbest_vectors - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask = np.random.rand(pop_size, dim) < cr[:, None]
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            trial = np.clip(trial, lb, ub)
            
            # --- Selection ---
            fitness_old = fitness.copy()
            
            succ_scr = []
            succ_sf = []
            diff_fitness = []
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_fitness
                
                new_val = func(trial[i])
                curr_evals += 1
                
                if new_val <= fitness[i]:
                    # Improved or equal
                    # Add parent to archive
                    if arc_count < pop_size:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Replace random
                        del_idx = np.random.randint(0, arc_count)
                        archive[del_idx] = pop[i].copy()
                        
                    diff = fitness[i] - new_val
                    succ_scr.append(cr[i])
                    succ_sf.append(f[i])
                    diff_fitness.append(diff)
                    
                    fitness[i] = new_val
                    pop[i] = trial[i]
                    
                    if new_val < best_fitness:
                        best_fitness = new_val
                        global_best_sol = trial[i].copy()
                        
            # --- Memory Update ---
            if len(diff_fitness) > 0:
                # Weighted Lehmer Mean
                w = np.array(diff_fitness)
                w_sum = np.sum(w)
                if w_sum != 0:
                    w = w / w_sum
                    
                    # Update M_CR
                    m_cr_new = np.sum(w * np.array(succ_scr))
                    # Moving average (c=0.5 usually, or direct replacement)
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * m_cr_new
                    
                    # Update M_F
                    sf = np.array(succ_sf)
                    m_f_new = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-15)
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * m_f_new
                    
                    k_mem = (k_mem + 1) % H_memory_size
            
            # --- Convergence/Stagnation Check ---
            # If population has collapsed to a point, restart to save time
            if np.std(fitness) < 1e-9 and np.max(fitness) - np.min(fitness) < 1e-9:
                break

    return best_fitness
