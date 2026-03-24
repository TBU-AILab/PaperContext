#The following Python code implements the **L-SHADE** algorithm (Linear Population Size Reduction Success-History based Adaptive Differential Evolution).
#
##### Improvements Explained:
#1.  **L-SHADE Strategy**: Unlike standard SHADE, this algorithm linearly reduces the population size over time. It starts with a larger population to encourage exploration and shrinks it to focus on exploitation as time runs out. This is highly effective for limited-time optimization.
#2.  **Time-Based Schedule**: The linear reduction is calculated based on the `max_time` budget rather than a fixed number of evaluations, allowing the algorithm to adapt dynamically to the provided time constraint.
#3.  **External Archive**: Maintains a diversity of successful past solutions to guide the mutation strategy (`current-to-pbest/1`), preventing premature convergence.
#4.  **Weighted Lehmer Mean**: Adaptive parameters (`F` and `CR`) are updated using a weighted mean based on fitness improvements, giving more weight to highly successful mutations.
#5.  **Robust Restarts**: If the population converges before time runs out, the algorithm triggers a restart with the remaining time budget, ensuring continuous searching.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History based Adaptive Differential Evolution) with Restarts.
    """
    
    # --- Configuration & Pre-processing ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Time Management
    start_time = datetime.now()
    # Buffer to ensure we return before the hard limit
    time_limit = timedelta(seconds=max_time - 0.05)
    
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # Global Best Tracking
    global_best_val = float('inf')
    
    # --- Main Restart Loop ---
    # We continue restarting until time runs out
    while not check_timeout():
        
        # 1. Initialization per Run
        # Population Size: Standard L-SHADE uses 18*dim. 
        # We clamp it to [30, 150] to balance exploration with Python's interpretation overhead.
        r_init = int(18 * dim)
        pop_size = int(np.clip(r_init, 30, 150))
        min_pop_size = 4
        
        # Initialize Population
        pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_timeout(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Initialize Memory (History)
        H = 5
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        mem_k = 0
        
        # Initialize Archive
        archive = []
        
        # Linear Reduction Setup
        # We calculate progress based on the remaining time for this run.
        run_start = datetime.now()
        elapsed_total = (run_start - start_time).total_seconds()
        remaining_budget = max_time - elapsed_total
        
        # If very little time remains, don't start a fresh heavy run
        if remaining_budget < 0.2:
            return global_best_val

        # --- Evolutionary Loop ---
        while True:
            if check_timeout(): return global_best_val
            
            # A. Linear Population Size Reduction (LPSR)
            # Calculate progress ratio based on time
            now = datetime.now()
            run_elapsed = (now - run_start).total_seconds()
            progress = run_elapsed / remaining_budget
            if progress > 1.0: progress = 1.0
            
            # Calculate new target population size
            next_pop_size = int(round(pop_size + (min_pop_size - pop_size) * progress))
            next_pop_size = max(min_pop_size, next_pop_size)
            
            # Resize Population if needed
            if next_pop_size < len(pop):
                # Sort and keep best
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:next_pop_size]]
                fitness = fitness[sort_idx[:next_pop_size]]
                
                # Resize Archive (Archive size tracks Population size)
                if len(archive) > next_pop_size:
                    # Randomly remove elements to shrink
                    # (Implementation: shuffle and slice)
                    np.random.shuffle(archive)
                    archive = archive[:next_pop_size]
            
            N = len(pop) # Current population size
            
            # B. Adaptive Parameter Generation
            # Pick random memory slot for each individual
            r_idx = np.random.randint(0, H, N)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F: Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(N) - 0.5))
            # Retry if F <= 0, Clip if F > 1
            while True:
                mask_neg = f <= 0
                if not np.any(mask_neg): break
                f[mask_neg] = m_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # C. Mutation: current-to-pbest/1 with Archive
            # Sort for p-best selection
            sorted_indices = np.argsort(fitness)
            sorted_pop = pop[sorted_indices]
            
            # Select p-best (top p%, where p is random in [2/N, 0.2])
            p_val = np.random.uniform(2/N, 0.2)
            top_cut = int(max(2, N * p_val))
            pbest_ind = np.random.randint(0, top_cut, N)
            x_pbest = sorted_pop[pbest_ind]
            
            # Select r1 (random from pop)
            r1 = np.random.randint(0, N, N)
            x_r1 = pop[r1]
            
            # Select r2 (random from Union(Pop, Archive))
            n_arc = len(archive)
            if n_arc > 0:
                limit = N + n_arc
                r2_idx = np.random.randint(0, limit, N)
                
                x_r2 = np.zeros((N, dim))
                mask_pop = r2_idx < N
                mask_arc = ~mask_pop
                
                # Take from pop
                x_r2[mask_pop] = pop[r2_idx[mask_pop]]
                
                # Take from archive
                if np.any(mask_arc):
                    arc_arr = np.array(archive)
                    arc_ptr = r2_idx[mask_arc] - N
                    x_r2[mask_arc] = arc_arr[arc_ptr]
            else:
                r2 = np.random.randint(0, N, N)
                x_r2 = pop[r2]
            
            # Calculate Mutation Vector
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # D. Bound Handling (Midpoint Correction)
            # If mutant violates bounds, place it halfway between parent and bound
            # Lower bounds
            mask_l = mutant < lb
            if np.any(mask_l):
                rows, cols = np.where(mask_l)
                mutant[rows, cols] = (pop[rows, cols] + lb[cols]) / 2.0
            
            # Upper bounds
            mask_u = mutant > ub
            if np.any(mask_u):
                rows, cols = np.where(mask_u)
                mutant[rows, cols] = (pop[rows, cols] + ub[cols]) / 2.0
            
            # E. Crossover (Binomial)
            mask_cross = np.random.rand(N, dim) < cr[:, None]
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(0, dim, N)
            mask_cross[np.arange(N), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # F. Selection & Updates
            trial_fitness = np.zeros(N)
            winners = np.zeros(N, dtype=bool)   # trial <= target
            improved = np.zeros(N, dtype=bool)  # trial < target
            
            old_fitness = fitness.copy()
            
            for i in range(N):
                if check_timeout(): return global_best_val
                
                f_tri = func(trial[i])
                trial_fitness[i] = f_tri
                
                if f_tri <= fitness[i]:
                    winners[i] = True
                    if f_tri < fitness[i]:
                        improved[i] = True
                    
                    if f_tri < global_best_val:
                        global_best_val = f_tri
            
            # Update Archive: Parent solutions replaced by better trials go to archive
            if np.any(winners):
                replaced_pop = pop[winners]
                for sol in replaced_pop:
                    if len(archive) < N:
                        archive.append(sol.copy())
                    else:
                        # Random replacement if full
                        ridx = np.random.randint(0, len(archive))
                        archive[ridx] = sol.copy()
            
            # Update Population
            pop[winners] = trial[winners]
            fitness[winners] = trial_fitness[winners]
            
            # G. Update Memory (Weighted Lehmer Mean)
            if np.any(improved):
                diffs = np.abs(old_fitness[improved] - fitness[improved])
                total_diff = np.sum(diffs)
                
                if total_diff > 0:
                    weights = diffs / total_diff
                    
                    # Update CR (Weighted Arithmetic Mean)
                    m_cr_new = np.sum(weights * cr[improved])
                    
                    # Update F (Weighted Lehmer Mean)
                    f_imp = f[improved]
                    denom = np.sum(weights * f_imp)
                    if denom > 0:
                        m_f_new = np.sum(weights * (f_imp ** 2)) / denom
                    else:
                        m_f_new = 0.5
                    
                    # Store in circular memory
                    mem_f[mem_k] = np.clip(m_f_new, 0, 1)
                    mem_cr[mem_k] = np.clip(m_cr_new, 0, 1)
                    mem_k = (mem_k + 1) % H
            
            # H. Check Convergence (Restart trigger)
            if np.std(fitness) < 1e-9:
                break
                
    return global_best_val
