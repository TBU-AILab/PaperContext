#The following algorithm implements a **Hybrid L-SHADE with Coordinate Descent**.
#
#**Rationale for Improvements:**
#1.  **L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution)**: This is widely considered the state-of-the-art for black-box continuous optimization. It adapts its mutation ($F$) and crossover ($CR$) parameters based on historical success, and it linearly reduces the population size to shift from exploration (global search) to exploitation (refining solutions).
#2.  **Hybrid Local Search (Coordinate Descent)**: Standard evolutionary algorithms (like DE) are excellent at finding the promising region (basin of attraction) but can be slow to refine the solution to high precision (polishing). By switching to a **Coordinate Descent** local search in the final phase of the allocated time (or if the population converges early), this algorithm can drastically improve the precision of the result, often by several orders of magnitude.
#3.  **Vectorized Implementation**: The code uses NumPy vectorization for population generation, mutation, and crossover, minimizing Python loop overhead.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Hybrid L-SHADE with Terminal Local Search.
    
    Phases:
    1. L-SHADE: Global search with adaptive parameters and reducing population.
    2. Local Search (Coordinate Descent): Triggered in the final 5% of time 
       or if L-SHADE converges, to polish the best solution found.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Reserve last 5% of time for local polishing (or at least 0.2s if time permits)
    ls_ratio = 0.05
    ls_start_time = start_time + limit * (1.0 - ls_ratio)
    
    # --- Configuration ---
    # Population Size: Start with roughly 18*dim but cap it to ensure speed
    # in high dimensions or short time limits.
    pop_size_init = int(min(200, max(30, 18 * dim)))
    pop_size_min = 4
    
    # L-SHADE Memory (History length H=5)
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive for diversity
    archive = []
    
    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Random initial population
    population = min_b + np.random.rand(pop_size_init, dim) * diff_b
    fitness = np.full(pop_size_init, float('inf'))
    
    best_val = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    # Perform strict time checks to handle very short durations
    for i in range(pop_size_init):
        if (datetime.now() - start_time) >= limit:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        now = datetime.now()
        
        # 1. Check for termination
        if now >= start_time + limit:
            return best_val
            
        # 2. Check for Local Search Trigger
        # Trigger if: Time is nearly up OR Population has converged (variance is low)
        converged = (np.std(fitness) < 1e-8) if len(fitness) > 0 else False
        
        if now >= ls_start_time or converged:
            # --- Phase 2: Coordinate Descent Local Search ---
            
            # Determine initial step size
            # If converged, use small step. If not, use population spread.
            if converged or len(population) < 2:
                step = np.mean(diff_b) * 0.01
            else:
                step = np.mean(np.std(population, axis=0))
                # Fallback if step is too small
                if step < 1e-9 * np.mean(diff_b):
                    step = np.mean(diff_b) * 0.005
            
            # Loop until time runs out
            while datetime.now() < start_time + limit:
                improved_iter = False
                
                # Iterate over dimensions in random order
                dims = np.random.permutation(dim)
                for d in dims:
                    if datetime.now() >= start_time + limit: return best_val
                    
                    original = best_sol[d]
                    
                    # Try decreasing value
                    best_sol[d] = np.clip(original - step, min_b[d], max_b[d])
                    val = func(best_sol)
                    if val < best_val:
                        best_val = val
                        improved_iter = True
                        continue # Keep change and move to next dim
                    
                    # Try increasing value
                    best_sol[d] = np.clip(original + step, min_b[d], max_b[d])
                    val = func(best_sol)
                    if val < best_val:
                        best_val = val
                        improved_iter = True
                        continue
                        
                    # Revert if no improvement
                    best_sol[d] = original
                
                # Adjust step size
                if not improved_iter:
                    step /= 2.0
                    # If step becomes insignificant, reset/restart or break?
                    if step < 1e-13:
                        # Small restart or just wait for timeout
                        step = np.mean(diff_b) * 0.001
            
            return best_val

        # --- Phase 1: L-SHADE ---
        
        # Calculate Linear Population Reduction
        elapsed = (now - start_time).total_seconds()
        # Use time up to LS start for scaling
        eff_dur = (ls_start_time - start_time).total_seconds()
        if eff_dur <= 0: eff_dur = 1e-9
        progress = min(1.0, elapsed / eff_dur)
        
        target_size = int(round(pop_size_init - progress * (pop_size_init - pop_size_min)))
        target_size = max(pop_size_min, target_size)
        
        # Resize Population
        curr_size = len(population)
        if curr_size > target_size:
            idxs = np.argsort(fitness)
            population = population[idxs[:target_size]]
            fitness = fitness[idxs[:target_size]]
            curr_size = target_size
            
            # Resize Archive (cap at 2x initial pop size or keep small)
            # Keeping it bounded ensures O(1) ops
            while len(archive) > pop_size_init:
                archive.pop(np.random.randint(len(archive)))

        # Adaptive Parameter Generation
        r_idxs = np.random.randint(0, H, curr_size)
        
        # Generate CR (Normal dist)
        cr = np.random.normal(mem_cr[r_idxs], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy dist)
        f = mem_f[r_idxs] + 0.1 * np.random.standard_cauchy(curr_size)
        # Repair F <= 0
        retry_mask = f <= 0
        while np.any(retry_mask):
            f[retry_mask] = mem_f[r_idxs[retry_mask]] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
            retry_mask = f <= 0
        f = np.clip(f, 0, 1)
        
        # Mutation Strategy: current-to-pbest/1
        # p-best selection (top 11%)
        p = 0.11
        p_num = max(2, int(p * curr_size))
        sorted_idxs = np.argsort(fitness)
        pbest_idxs = np.random.choice(sorted_idxs[:p_num], curr_size)
        x_pbest = population[pbest_idxs]
        
        # r1 selection (distinct from i)
        r1_idxs = np.random.randint(0, curr_size, curr_size)
        for i in range(curr_size):
            while r1_idxs[i] == i:
                r1_idxs[i] = np.random.randint(0, curr_size)
        x_r1 = population[r1_idxs]
        
        # r2 selection (from Union(Population, Archive), distinct from i and r1)
        if len(archive) > 0:
            arc_arr = np.array(archive)
            union = np.vstack((population, arc_arr))
        else:
            union = population
            
        r2_idxs = np.random.randint(0, len(union), curr_size)
        for i in range(curr_size):
            while r2_idxs[i] == i or r2_idxs[i] == r1_idxs[i]:
                r2_idxs[i] = np.random.randint(0, len(union))
        x_r2 = union[r2_idxs]
        
        # Compute Mutant Vectors (Vectorized)
        f_v = f[:, np.newaxis]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        mutant = np.clip(mutant, min_b, max_b)
        
        # Crossover (Binomial)
        j_rand = np.random.randint(0, dim, curr_size)
        rand_vals = np.random.rand(curr_size, dim)
        mask = rand_vals < cr[:, np.newaxis]
        mask[np.arange(curr_size), j_rand] = True
        
        trial = np.where(mask, mutant, population)
        
        # Evaluation
        diff_f = []
        success_f = []
        success_cr = []
        
        for i in range(curr_size):
            if datetime.now() >= ls_start_time: break
            
            t_val = func(trial[i])
            
            if t_val <= fitness[i]:
                # If better, store improvement for memory update
                if t_val < fitness[i]:
                    archive.append(population[i].copy())
                    diff_f.append(fitness[i] - t_val)
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    
                population[i] = trial[i]
                fitness[i] = t_val
                
                if t_val < best_val:
                    best_val = t_val
                    best_sol = trial[i].copy()
                    
        # Memory Update (Weighted Lehmer Mean)
        if len(diff_f) > 0:
            w = np.array(diff_f)
            w = w / np.sum(w)
            
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            mean_f = np.sum(w * (s_f**2)) / (np.sum(w * s_f) + 1e-9)
            mean_cr = np.sum(w * s_cr)
            
            mem_f[k_mem] = mean_f
            mem_cr[k_mem] = mean_cr
            k_mem = (k_mem + 1) % H

    return best_val
