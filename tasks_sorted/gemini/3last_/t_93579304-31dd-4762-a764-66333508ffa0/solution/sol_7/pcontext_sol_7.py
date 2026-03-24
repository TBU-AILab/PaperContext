#The following algorithm implements **L-SHADE with MTS-LS1 Local Search Phase**.
#
#**Rationale for Improvements:**
#1.  **L-SHADE Backbone**: The base is the L-SHADE algorithm (Linear Population Size Reduction Success-History Adaptive Differential Evolution), which proved most effective in previous trials. It uses linear population reduction (LPSR) mapped to the time limit to enforce convergence.
#2.  **MTS-LS1 Local Search**: Instead of the simple Coordinate Descent used in the previous attempt (which scored 10.26), this version uses the **MTS-LS1** (Multiple Trajectory Search - Local Search 1) strategy. MTS-LS1 is dynamic; it adjusts the search range (step size) for each dimension independently based on success, allowing it to navigate complex valleys and "polish" the solution to a much higher precision.
#3.  **Hybrid Triggering**: The algorithm switches from global search (L-SHADE) to local search (MTS-LS1) under two conditions:
#    *   **Time-based**: When 90% of the allocated time has elapsed, ensuring the final moments are spent refining the best solution.
#    *   **Convergence-based**: If the population size reduces to the minimum ($N=4$) before time runs out, it immediately switches to local search to utilize the remaining time efficiently.
#4.  **Optimal Bound Handling**: Based on previous results (where Clipping scored 10.26 vs. Midpoint's 228), this version strictly uses **Clipping** (`np.clip`) to handle bound constraints.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE optimized for limited time, 
    followed by an MTS-LS1 local search phase.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Parameters ---
    # Initial Population: 25*dim is a standard effective heuristic for L-SHADE
    # Capped to maintain speed on high dimensions/low time
    pop_size_init = int(25 * dim)
    pop_size_init = max(30, min(pop_size_init, 500)) 
    min_pop_size = 4
    
    # Adaptive Memory (History length H=5)
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive for diversity maintenance
    archive = []
    arc_rate = 2.0 # Archive size is 2.0x population size
    
    # Bound parsing
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # --- Initialization ---
    pop_size = pop_size_init
    population = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if (datetime.now() - start_time) >= limit:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
            
    # Sort population (required for L-SHADE operations)
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Main Optimization Loop ---
    while True:
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        time_ratio = elapsed / max_time
        
        # Check termination
        if time_ratio >= 1.0:
            break
            
        # --- Local Search Phase Trigger ---
        # Trigger if:
        # 1. Population has collapsed to minimum (Converged)
        # 2. Or, we are in the final 10% of the time budget (Polish phase)
        is_converged = (pop_size <= min_pop_size)
        
        if is_converged or time_ratio > 0.90:
            
            # MTS-LS1 (Multiple Trajectory Search - Local Search 1)
            # Initialize search range (step sizes) for each dimension
            sr = (ub - lb) * 0.4
            
            # Run until time is up
            while (datetime.now() - start_time) < limit:
                
                # Search dimensions in random order to avoid bias
                dims_order = np.random.permutation(dim)
                improved = False
                
                for d in dims_order:
                    if (datetime.now() - start_time) >= limit:
                        return best_val
                    
                    x_curr = best_sol.copy()
                    
                    # Direction 1: x - SR
                    x_curr[d] = np.clip(best_sol[d] - sr[d], lb[d], ub[d])
                    val = func(x_curr)
                    
                    if val < best_val:
                        best_val = val
                        best_sol = x_curr.copy()
                        improved = True
                    else:
                        # Direction 2: x + 0.5 * SR (MTS Logic)
                        x_curr[d] = np.clip(best_sol[d] + 0.5 * sr[d], lb[d], ub[d])
                        val = func(x_curr)
                        
                        if val < best_val:
                            best_val = val
                            best_sol = x_curr.copy()
                            improved = True
                        else:
                            # If neither improved, reduce search range for this dimension
                            sr[d] *= 0.5
                            
                # Check search range validity
                if np.max(sr) < 1e-15:
                    # If step size is too small, effectively converged.
                    # We can break early or reset. Given template, we return.
                    return best_val
                    
            return best_val

        # --- L-SHADE Global Search Step ---
        
        # 1. Linear Population Size Reduction (LPSR) based on Time
        # Calculate target population size
        target_size = int(round(pop_size_init + (min_pop_size - pop_size_init) * time_ratio))
        target_size = max(min_pop_size, target_size)
        
        if pop_size > target_size:
            # Truncate population (worst individuals are at the end due to sorting)
            population = population[:target_size]
            fitness = fitness[:target_size]
            pop_size = target_size
            
            # Resize Archive
            max_arc_size = int(pop_size * arc_rate)
            while len(archive) > max_arc_size:
                archive.pop(np.random.randint(len(archive)))
        
        # 2. Adaptive Parameter Generation
        # p-best value decreases linearly from 0.11 to 0.02 to shift exploration -> exploitation
        p_val = 0.11 - 0.09 * time_ratio
        p_val = max(0.02, p_val)
        
        # Pick memory cells
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idxs]
        m_f = mem_f[r_idxs]
        
        # Generate CR (Normal, clipped)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy, repaired)
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        while np.any(f <= 0):
            mask = f <= 0
            f[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
        f = np.clip(f, 0, 1)
        
        # 3. Mutation: current-to-pbest/1
        # Select p-best
        p_num = max(2, int(pop_size * p_val))
        pbest_idxs = np.random.randint(0, p_num, pop_size)
        x_pbest = population[pbest_idxs]
        
        # Select r1 (!= i)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        # Simple fix for self-selection
        for i in range(pop_size):
            while r1_idxs[i] == i:
                r1_idxs[i] = np.random.randint(0, pop_size)
        x_r1 = population[r1_idxs]
        
        # Select r2 (!= i, != r1) from Union(Population, Archive)
        if len(archive) > 0:
            union_pop = np.vstack((population, np.array(archive)))
        else:
            union_pop = population
        n_union = len(union_pop)
        
        r2_idxs = np.random.randint(0, n_union, pop_size)
        for i in range(pop_size):
            while r2_idxs[i] == i or r2_idxs[i] == r1_idxs[i]:
                r2_idxs[i] = np.random.randint(0, n_union)
        x_r2 = union_pop[r2_idxs]
        
        # Compute Mutant
        f_v = f[:, np.newaxis]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals < cr[:, np.newaxis]
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask, mutant, population)
        
        # 5. Bound Constraints (Clipping)
        trial = np.clip(trial, lb, ub)
        
        # 6. Evaluation and Selection
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        success_f = []
        success_cr = []
        diff_f = []
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= limit:
                return best_val
            
            y = func(trial[i])
            
            if y <= fitness[i]:
                new_pop[i] = trial[i]
                new_fit[i] = y
                
                if y < fitness[i]:
                    # Successful update
                    archive.append(population[i].copy())
                    diff_f.append(fitness[i] - y)
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    
                if y < best_val:
                    best_val = y
                    best_sol = trial[i].copy()
                    
        population = new_pop
        fitness = new_fit
        
        # 7. Update Adaptive Memories (Weighted Lehmer Mean)
        if len(diff_f) > 0:
            w = np.array(diff_f)
            w_sum = np.sum(w)
            if w_sum > 0:
                w /= w_sum
                
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                
                mean_scr = np.sum(w * s_cr)
                
                sum_w_sf = np.sum(w * s_f)
                if abs(sum_w_sf) < 1e-9:
                    mean_sf = 0.5
                else:
                    mean_sf = np.sum(w * (s_f ** 2)) / sum_w_sf
                
                mem_f[k_mem] = np.clip(mean_sf, 0, 1)
                mem_cr[k_mem] = np.clip(mean_scr, 0, 1)
                k_mem = (k_mem + 1) % H
        
        # 8. Archive Maintenance (External check)
        max_arc_size = int(pop_size * arc_rate)
        while len(archive) > max_arc_size:
            archive.pop(np.random.randint(len(archive)))
            
        # 9. Sort for next generation
        sorted_idxs = np.argsort(fitness)
        population = population[sorted_idxs]
        fitness = fitness[sorted_idxs]

    return best_val
