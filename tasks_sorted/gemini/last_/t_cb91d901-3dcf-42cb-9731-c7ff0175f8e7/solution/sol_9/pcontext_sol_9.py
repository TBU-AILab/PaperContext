#Here is the fully operational, self-contained Python algorithm.
#
#It implements **Restart L-SHADE with Local Search (L-SHADE-R-LS)**. This algorithm combines the global search power of L-SHADE (with Linear Population Size Reduction and Adaptive Parameters) with a deterministic **Coordinate Descent Local Search** to refine the best solution found, and a **Biased Restart** strategy to efficiently explore the landscape within the time limit.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Restart L-SHADE with Local Search (L-SHADE-R-LS)
    
    Features:
    1. L-SHADE: State-of-the-art Differential Evolution with:
       - Linear Population Size Reduction (LPSR)
       - Success-History based Parameter Adaptation
       - External Archive
    2. Biased Restart: Re-initializes population with a mix of global random points 
       and a Gaussian cloud around the best-so-far solution to escape local optima 
       while exploiting the current basin.
    3. Coordinate Descent Local Search: Runs upon convergence to fine-tune the solution.
    4. Robust Time Management: Adapts budget dynamically.
    """
    
    # --- Configuration ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Bound processing
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    bound_diff = ub - lb
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_x = None
    
    # --- Helper: Time Check & Safe Evaluation ---
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    class TimeoutException(Exception):
        pass

    def safe_evaluate(x):
        nonlocal best_fitness, best_x
        if check_timeout():
            raise TimeoutException
        
        val = func(x)
        
        if val < best_fitness:
            best_fitness = val
            best_x = x.copy()
        return val

    # --- Initial Cost Estimation ---
    # Run a small batch to estimate function evaluation time
    n_est = 5
    t_start_est = datetime.now()
    try:
        for _ in range(n_est):
            rnd_x = lb + np.random.rand(dim) * bound_diff
            safe_evaluate(rnd_x)
    except TimeoutException:
        return best_fitness

    elapsed = (datetime.now() - t_start_est).total_seconds()
    avg_eval_time = elapsed / n_est if elapsed > 1e-6 else 0.0

    # --- L-SHADE Parameters ---
    # Initial Population Size: 18 * dim (standard for L-SHADE), clipped for safety
    N_init = int(round(18 * dim))
    N_init = max(30, min(N_init, 300))
    N_min = 4
    
    # History Memory Size
    H = 5
    
    # Loop Counters
    nfe_total = n_est
    restart_count = 0
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check remaining time
        now = datetime.now()
        remaining_seconds = (start_time + time_limit - now).total_seconds()
        if remaining_seconds <= 0:
            break
            
        # 1. Budgeting for this Epoch
        if avg_eval_time > 0:
            est_capacity = int(remaining_seconds / avg_eval_time)
        else:
            est_capacity = 1000000 # Large dummy value
            
        # Allocate budget: 
        # If it's the first run, assume we might need restarts -> don't use 100% capacity for LPSR scale.
        # Use a heuristic: max(500*dim, 60% of remaining) to allow for restarts.
        epoch_nfe_budget = max(int(500 * dim), int(est_capacity * 0.6))
        
        # 2. Initialization
        pop_size = N_init
        pop = np.zeros((pop_size, dim))
        
        # Biased Initialization Logic (Exploit + Explore)
        # If we have a good solution, seed 30% of new pop around it
        n_biased = 0
        if restart_count > 0 and best_x is not None:
            n_biased = int(0.3 * pop_size)
            # Sigma: 5% of domain width for local cloud
            sigma = 0.05 * bound_diff
            bias_cloud = np.random.normal(best_x, sigma, (n_biased, dim))
            bias_cloud = np.clip(bias_cloud, lb, ub)
            pop[:n_biased] = bias_cloud
            
        n_rand = pop_size - n_biased
        pop[n_biased:] = lb + np.random.rand(n_rand, dim) * bound_diff
        
        fitness = np.zeros(pop_size)
        try:
            for i in range(pop_size):
                fitness[i] = safe_evaluate(pop[i])
                nfe_total += 1
        except TimeoutException:
            return best_fitness
            
        # Sort population
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        
        # L-SHADE State
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        nfe_epoch_start = nfe_total
        
        # 3. Evolution Loop
        while (nfe_total - nfe_epoch_start) < epoch_nfe_budget:
            if check_timeout(): return best_fitness
            
            # --- Linear Population Size Reduction (LPSR) ---
            progress = (nfe_total - nfe_epoch_start) / epoch_nfe_budget
            target_size = int(round((N_min - N_init) * progress + N_init))
            target_size = max(N_min, target_size)
            
            if pop_size > target_size:
                # Truncate lowest fitness (pop is sorted at end of loop)
                pop = pop[:target_size]
                fitness = fitness[:target_size]
                pop_size = target_size
                # Reduce archive
                if len(archive) > pop_size:
                    random.shuffle(archive)
                    archive = archive[:pop_size]
            
            # --- Parameter Adaptation ---
            r_idxs = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idxs]
            m_f = M_f[r_idxs]
            
            # Generate Cr (Normal)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy)
            f = np.zeros(pop_size)
            # Vectorized rejection sampling for Cauchy > 0
            todo_mask = np.ones(pop_size, dtype=bool)
            while np.any(todo_mask):
                n_todo = np.sum(todo_mask)
                f_gen = m_f[r_idxs[todo_mask]] + 0.1 * np.tan(np.pi * (np.random.rand(n_todo) - 0.5))
                valid = f_gen > 0
                
                # Update valid entries
                indices = np.where(todo_mask)[0]
                valid_indices = indices[valid]
                f[valid_indices] = np.minimum(f_gen[valid], 1.0)
                todo_mask[valid_indices] = False
            
            # --- Mutation: current-to-pbest/1 ---
            # Sort for p-best selection
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # p-best selection (top 11%)
            p_val = 0.11
            p_num = max(1, int(round(p_val * pop_size)))
            pbest_idxs = np.random.randint(0, p_num, pop_size)
            x_pbest = pop[pbest_idxs]
            
            # r1 selection (r1 != i)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            conflict = r1_idxs == np.arange(pop_size)
            while np.any(conflict):
                r1_idxs[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
                conflict = r1_idxs == np.arange(pop_size)
            x_r1 = pop[r1_idxs]
            
            # r2 selection (r2 != r1, r2 != i, from Pop U Archive)
            if len(archive) > 0:
                pop_union = np.vstack((pop, np.array(archive)))
            else:
                pop_union = pop
            
            n_union = len(pop_union)
            r2_idxs = np.random.randint(0, n_union, pop_size)
            conflict = (r2_idxs == np.arange(pop_size)) | (r2_idxs == r1_idxs)
            while np.any(conflict):
                r2_idxs[conflict] = np.random.randint(0, n_union, np.sum(conflict))
                conflict = (r2_idxs == np.arange(pop_size)) | (r2_idxs == r1_idxs)
            x_r2 = pop_union[r2_idxs]
            
            # Compute Mutant V
            v = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            mask = np.random.rand(pop_size, dim) < cr[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            u = np.where(mask, v, pop)
            
            # --- Bound Handling (Midpoint Bounce-Back) ---
            # Better than clipping for diversity
            mask_l = u < lb
            if np.any(mask_l):
                r, c = np.where(mask_l)
                u[r, c] = (pop[r, c] + lb[c]) / 2.0
            mask_u = u > ub
            if np.any(mask_u):
                r, c = np.where(mask_u)
                u[r, c] = (pop[r, c] + ub[c]) / 2.0
            
            u = np.clip(u, lb, ub)
            
            # --- Evaluation ---
            new_fitness = np.zeros(pop_size)
            try:
                for i in range(pop_size):
                    new_fitness[i] = safe_evaluate(u[i])
                    nfe_total += 1
            except TimeoutException:
                return best_fitness
            
            # --- Selection & Update ---
            # Selection for next generation (u <= x)
            selection_mask = new_fitness <= fitness
            
            # Archive update: strictly better children push parent to archive
            archive_mask = new_fitness < fitness
            if np.any(archive_mask):
                parents_to_archive = pop[archive_mask].copy()
                for p_vec in parents_to_archive:
                    archive.append(p_vec)
                # Resize archive
                while len(archive) > pop_size:
                    archive.pop(random.randint(0, len(archive)-1))
                    
                # Update Memory (Weighted Lehmer Mean)
                diffs = np.abs(fitness[archive_mask] - new_fitness[archive_mask])
                total_diff = np.sum(diffs)
                if total_diff > 0:
                    weights = diffs / total_diff
                    
                    # F Update
                    succ_f = f[archive_mask]
                    mean_f_wl = np.sum(weights * (succ_f**2)) / np.sum(weights * succ_f)
                    M_f[k_mem] = mean_f_wl
                    
                    # Cr Update
                    succ_cr = cr[archive_mask]
                    if np.max(succ_cr) == 0:
                        M_cr[k_mem] = 0
                    else:
                        denom = np.sum(weights * succ_cr)
                        if denom > 0:
                            M_cr[k_mem] = np.sum(weights * (succ_cr**2)) / denom
                        else:
                            M_cr[k_mem] = 0.5
                            
                    k_mem = (k_mem + 1) % H
            
            # Apply Selection
            pop[selection_mask] = u[selection_mask]
            fitness[selection_mask] = new_fitness[selection_mask]
            
            # Check for Convergence
            # 1. Population too small
            if pop_size <= N_min:
                break
            # 2. Fitness converged
            if abs(np.max(fitness) - np.min(fitness)) < 1e-8:
                break
        
        # --- Local Search (Coordinate Descent) ---
        # Run on the best solution found to refine results before restart
        if best_x is not None:
            # Simple coordinate descent
            # Initial step size
            step_size = (ub - lb) * 0.01 
            # Max 3 passes to save time
            try:
                for _ in range(3):
                    improved_ls = False
                    for d in range(dim):
                        if check_timeout(): raise TimeoutException
                        
                        origin = best_x[d]
                        curr_best = best_fitness
                        
                        # Try negative direction
                        target = max(lb[d], origin - step_size[d])
                        temp_x = best_x.copy()
                        temp_x[d] = target
                        val = safe_evaluate(temp_x)
                        
                        if val < curr_best:
                            improved_ls = True
                            continue 
                        
                        # Try positive direction
                        target = min(ub[d], origin + step_size[d])
                        temp_x = best_x.copy() # Reset using latest best_x
                        temp_x[d] = target
                        val = safe_evaluate(temp_x)
                        
                        if val < curr_best:
                            improved_ls = True
                    
                    if not improved_ls:
                        step_size /= 2.0 # Refine step
                    if np.max(step_size) < 1e-9:
                        break
            except TimeoutException:
                return best_fitness

        restart_count += 1
        
    return best_fitness
