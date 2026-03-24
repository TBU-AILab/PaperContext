#An improved algorithm implementation using **SHADE with Archive, Opposition-Based Learning (OBL), and Stagnation-based Restarts**.
#
#### Improvements Explained:
#
#1.  **SHADE with External Archive**: This implementation uses the Success-History Adaptive Differential Evolution (SHADE) algorithm. It maintains a history of successful control parameters ($F$ and $CR$) to adapt to the function landscape. Crucially, it includes an **external archive** of inferior solutions, which preserves population diversity and prevents premature convergence, allowing the `current-to-pbest/1` mutation strategy to explore more effectively.
#2.  **Opposition-Based Learning (OBL) Initialization**: At the start of every restart, the algorithm generates both random solutions and their "opposite" counterparts ($lb + ub - x$) within the search space. It selects the best fit individuals from the combined pool. This significantly accelerates the discovery of the basin of attraction.
#3.  **Midpoint Bound Handling**: Instead of simple clipping (which causes solutions to stick to the edges), particles violating bounds are reset to the midpoint between the parent position and the bound ($ (x + bound)/2 $). This preserves the search direction and momentum.
#4.  **Stagnation-based Restarts**: The algorithm monitors population fitness variance and improvement stagnation. If the population converges (variance $\to 0$) or fails to improve the best solution for a set number of generations, it triggers a restart. The global best solution is preserved (elitism) across restarts to ensure monotonic improvement.
#5.  **Vectorized Operations**: All mutation, crossover, and boundary handling operations are vectorized using NumPy to maximize the number of function evaluations within the `max_time` limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE with Opposition-Based Learning (OBL),
    External Archive, and Stagnation-based Restarts.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Use a small buffer to ensure we return before the strict cutoff
    time_limit = timedelta(seconds=max_time - 0.05)

    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global Best Tracking
    global_best_fit = float('inf')
    global_best_sol = None
    
    # SHADE Parameters
    H = 5  # History memory size
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        if check_time(): return global_best_fit
        
        # 1. Adaptive Population Sizing
        # Use a population size scaling with dimension, clipped to reasonable limits
        pop_size = int(np.clip(20 * dim, 40, 150))
        
        # 2. Initialization with OBL (Opposition-Based Learning)
        # Generate Random Population
        p_rand = lb + np.random.rand(pop_size, dim) * (ub - lb)
        f_rand = np.full(pop_size, float('inf'))
        
        # Evaluate Random
        for i in range(pop_size):
            if i % 10 == 0 and check_time(): return global_best_fit
            val = func(p_rand[i])
            f_rand[i] = val
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = p_rand[i].copy()
                
        # Generate Opposite Population
        p_opp = lb + ub - p_rand
        p_opp = np.clip(p_opp, lb, ub) # Clip opposite to bounds
        f_opp = np.full(pop_size, float('inf'))
        
        # Evaluate Opposite
        for i in range(pop_size):
            if i % 10 == 0 and check_time(): return global_best_fit
            val = func(p_opp[i])
            f_opp[i] = val
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = p_opp[i].copy()
                
        # Selection: Choose best N individuals from (Random + Opposite)
        combined_pop = np.vstack((p_rand, p_opp))
        combined_fit = np.concatenate((f_rand, f_opp))
        
        sort_idx = np.argsort(combined_fit)
        pop = combined_pop[sort_idx[:pop_size]]
        fitness = combined_fit[sort_idx[:pop_size]]
        
        # Elitism: Inject global best from previous restart if available
        if global_best_sol is not None:
            # Replace worst individual
            pop[-1] = global_best_sol
            fitness[-1] = global_best_fit
            # Re-sort to maintain order for p-best selection
            s_idx = np.argsort(fitness)
            pop = pop[s_idx]
            fitness = fitness[s_idx]

        # Initialize SHADE Memory
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive = []
        
        # Stagnation Counters
        last_gen_best = fitness[0]
        stag_count = 0
        
        # --- Evolution Loop ---
        while True:
            if check_time(): return global_best_fit
            
            # --- A. Parameter Generation ---
            # Select random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]
            
            # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F: Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # F handling: Retry if <= 0, Clip if > 1
            bad_f = f <= 0
            while np.any(bad_f):
                count = np.sum(bad_f)
                f[bad_f] = m_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(count) - 0.5))
                bad_f = f <= 0
            f = np.minimum(f, 1.0)
            
            # --- B. Mutation (current-to-pbest/1) ---
            # Population is sorted by fitness from start or end of loop
            
            # p-best selection: random top p% (p in [2/N, 0.2])
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_cut = int(max(2, pop_size * p_val))
            
            pbest_indices = np.random.randint(0, top_cut, pop_size)
            x_pbest = pop[pbest_indices]
            
            # r1 selection: random from pop, r1 != i
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Handle collision r1 == i
            col_r1 = (r1_idx == np.arange(pop_size))
            r1_idx[col_r1] = (r1_idx[col_r1] + 1) % pop_size
            x_r1 = pop[r1_idx]
            
            # r2 selection: random from Union(Pop, Archive), r2 != r1, r2 != i
            if len(archive) > 0:
                arc_np = np.array(archive)
                union_pop = np.vstack((pop, arc_np))
            else:
                union_pop = pop
            
            r2_idx = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # Bound Handling: Midpoint correction
            # If v < lb, set v = (x + lb) / 2
            low_mask = mutant < lb
            mutant[low_mask] = (pop[low_mask] + lb[np.where(low_mask)[1]]) / 2.0
            
            high_mask = mutant > ub
            mutant[high_mask] = (pop[high_mask] + ub[np.where(high_mask)[1]]) / 2.0
            
            # --- C. Crossover (Binomial) ---
            rand_cr = np.random.rand(pop_size, dim)
            mask_cr = rand_cr < cr[:, None]
            # Ensure at least one parameter comes from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cr[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cr, mutant, pop)
            
            # --- D. Evaluation & Selection ---
            trial_fit = np.zeros(pop_size)
            winners = np.zeros(pop_size, dtype=bool)
            diffs = np.zeros(pop_size)
            
            for i in range(pop_size):
                # Check time periodically to minimize overhead
                if i % 10 == 0 and check_time(): return global_best_fit
                
                val = func(trial[i])
                trial_fit[i] = val
                
                # Selection
                if val <= fitness[i]:
                    winners[i] = True
                    diffs[i] = fitness[i] - val
                    if val < global_best_fit:
                        global_best_fit = val
                        global_best_sol = trial[i].copy()
            
            # --- E. Archive Update ---
            # Add parent solutions that were replaced to archive
            replaced = pop[winners].copy()
            for vec in replaced:
                if len(archive) < pop_size:
                    archive.append(vec)
                else:
                    # Randomly replace an archive member
                    k = np.random.randint(0, len(archive))
                    archive[k] = vec
            
            # --- F. Population Update ---
            pop[winners] = trial[winners]
            fitness[winners] = trial_fit[winners]
            
            # --- G. Memory Update (Weighted Lehmer Mean) ---
            if np.any(winners):
                w_f = f[winners]
                w_cr = cr[winners]
                w_diff = diffs[winners]
                
                sum_diff = np.sum(w_diff)
                if sum_diff > 0:
                    weights = w_diff / sum_diff
                    
                    # Update M_CR (Weighted Mean)
                    m_cr_new = np.sum(weights * w_cr)
                    
                    # Update M_F (Weighted Lehmer Mean)
                    sum_f = np.sum(weights * w_f)
                    sum_f2 = np.sum(weights * (w_f ** 2))
                    if sum_f > 0:
                        m_f_new = sum_f2 / sum_f
                    else:
                        m_f_new = 0.5
                        
                    mem_cr[k_mem] = np.clip(m_cr_new, 0, 1)
                    mem_f[k_mem] = np.clip(m_f_new, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # --- H. Sorting & Restart Check ---
            s_idx = np.argsort(fitness)
            pop = pop[s_idx]
            fitness = fitness[s_idx]
            
            curr_best = fitness[0]
            
            # Stagnation Check
            if (last_gen_best - curr_best) < 1e-9:
                stag_count += 1
            else:
                stag_count = 0
                last_gen_best = curr_best
            
            std_dev = np.std(fitness)
            
            # Restart Triggers:
            # 1. Population converged (std dev ~ 0)
            # 2. No improvement for 30 generations
            if std_dev < 1e-9 or stag_count > 30:
                break

    return global_best_fit
