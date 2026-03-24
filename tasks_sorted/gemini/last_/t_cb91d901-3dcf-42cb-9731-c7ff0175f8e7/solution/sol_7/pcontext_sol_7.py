#Here is a fully operational, self-contained Python algorithm implementing **jSO (L-SHADE variant)** with adaptive resource allocation.
#
#### Improvement Logic
#1.  **Adaptive Budget Estimation**: Instead of assuming a fixed number of evaluations or relying purely on time checks, the algorithm estimates the total affordable evaluations (NFE) based on the first generation's execution time. This allows the linear schedules (for population reduction and parameter adaptation) to be perfectly scaled to the available time window.
#2.  **jSO Refinements**: Implements specific mechanisms from the jSO algorithm (a top-performing CEC winner), including:
#    *   **Weighted Lehmer Mean** for both $F$ and $Cr$ memory updates (biasing towards high-performance parameters).
#    *   **Dynamic $p$-best**: The top percentage $p$ for mutation selection decays linearly from 0.25 to 0.05, shifting focus from exploration to exploitation.
#    *   **Midpoint Bounce-Back**: A robust boundary handling technique that preserves diversity better than clipping.
#3.  **Linear Population Size Reduction (LPSR)**: The population size linearly decreases based on the consumed budget, ensuring broad search initially and efficient local convergence towards the end.
#4.  **Vectorized Operations**: Maximizes performance using NumPy vectorization for mutation, crossover, and constraint handling.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    jSO (L-SHADE variant) with Adaptive Budget Estimation and LPSR.
    Optimized for minimizing black-box functions within a strict time limit.
    """
    # --- Configuration & Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    
    # Initial Population Size
    # Heuristic: 18 * dim is standard for L-SHADE, but we cap at 200 
    # to ensure sufficient generations can run within limited time.
    pop_size = int(round(18 * dim))
    pop_size = max(10, min(pop_size, 200))
    
    # Initialize Population
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    
    # --- Step 1: Initial Evaluation & Budget Estimation ---
    nfe = 0
    t_start_eval = datetime.now()
    
    for i in range(pop_size):
        # Safety check: if time runs out during initialization
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness
            
        val = func(pop[i])
        nfe += 1
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            
    # Estimate total available NFE (Number of Function Evaluations)
    # Based on time taken for the initial batch
    elapsed_init = (datetime.now() - t_start_eval).total_seconds()
    
    if elapsed_init > 1e-6:
        # Rate = evals / second. Projected NFE = Rate * max_time.
        # We use a safety factor (0.95) to avoid timeout during the last generation.
        est_total_nfe = int((nfe / elapsed_init) * max_time * 0.95)
    else:
        # Function is extremely fast; assume ample budget
        est_total_nfe = 10**7
        
    # Ensure budget is at least enough for a few generations
    max_nfe = max(est_total_nfe, nfe + 200)
    
    # --- jSO / L-SHADE Parameters ---
    # Memory for adaptive parameters
    H = 6 
    M_cr = np.full(H, 0.8) # Starts high (0.8) for exploration
    M_f = np.full(H, 0.5)  # Starts balanced (0.5)
    k_mem = 0
    archive = [] 
    
    # LPSR Configuration
    n_init = pop_size
    n_min = 4
    
    # --- Main Optimization Loop ---
    while True:
        # Time Check
        if (datetime.now() - start_time) >= time_limit:
            break
            
        # Calculate Progress (0.0 to 1.0)
        progress = nfe / max_nfe
        if progress > 1.0: progress = 1.0
        
        # 1. Linear Population Size Reduction (LPSR)
        target_size = int(round((n_min - n_init) * progress + n_init))
        target_size = max(n_min, target_size)
        
        if pop_size > target_size:
            # Sort population by fitness and truncate
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx[:target_size]]
            fitness = fitness[sort_idx[:target_size]]
            
            # Resize Archive (Archive size = Population size)
            if len(archive) > target_size:
                random.shuffle(archive)
                archive = archive[:target_size]
                
            pop_size = target_size
            
        # 2. Adaptive Parameter Generation
        # p-best rate decays linearly from 0.25 to 0.05
        p_curr = 0.25 - 0.20 * progress
        
        # Select memory slot
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = M_cr[r_idx]
        m_f = M_f[r_idx]
        
        # Generate Cr ~ Normal(M_cr, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(M_f, 0.1)
        # Must be > 0. If > 1, clip to 1.
        f = np.zeros(pop_size)
        todo_mask = np.ones(pop_size, dtype=bool)
        
        while np.any(todo_mask):
            n_todo = np.sum(todo_mask)
            f_gen = m_f[todo_mask] + 0.1 * np.tan(np.pi * (np.random.rand(n_todo) - 0.5))
            
            valid = f_gen > 0
            
            # Indices to update
            current_indices = np.where(todo_mask)[0]
            valid_indices = current_indices[valid]
            
            f[valid_indices] = np.minimum(f_gen[valid], 1.0)
            todo_mask[valid_indices] = False
            
        # 3. Mutation: current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Select p-best individuals
        sorted_indices = np.argsort(fitness)
        n_pbest = max(1, int(round(p_curr * pop_size)))
        pbest_pool = sorted_indices[:n_pbest]
        pbest_idx = np.random.choice(pbest_pool, pop_size)
        
        # Select r1 (distinct from i)
        r1_idx = np.random.randint(0, pop_size, pop_size)
        collision = r1_idx == np.arange(pop_size)
        while np.any(collision):
            r1_idx[collision] = np.random.randint(0, pop_size, np.sum(collision))
            collision = r1_idx == np.arange(pop_size)
            
        # Select r2 (distinct from i and r1, from Pop U Archive)
        if len(archive) > 0:
            arr_arc = np.array(archive)
            pop_all = np.vstack((pop, arr_arc))
        else:
            pop_all = pop
            
        n_all = len(pop_all)
        r2_idx = np.random.randint(0, n_all, pop_size)
        
        # Collision handling for r2
        c1 = r2_idx == np.arange(pop_size) # Collision with self (only if r2 points to pop)
        c2 = r2_idx == r1_idx              # Collision with r1
        collision = c1 | c2
        while np.any(collision):
            r2_idx[collision] = np.random.randint(0, n_all, np.sum(collision))
            c1 = r2_idx == np.arange(pop_size)
            c2 = r2_idx == r1_idx
            collision = c1 | c2
            
        # Calculate Mutant Vectors
        x = pop
        x_pbest = pop[pbest_idx]
        x_r1 = pop[r1_idx]
        x_r2 = pop_all[r2_idx]
        
        v = x + f[:, None] * (x_pbest - x) + f[:, None] * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True # Ensure at least one parameter comes from mutant
        
        u = np.where(mask, v, x)
        
        # 5. Bound Handling (Midpoint Bounce-back)
        # If out of bounds, place halfway between limit and old position
        # Lower bounds
        mask_l = u < lb
        if np.any(mask_l):
            rows, cols = np.where(mask_l)
            u[rows, cols] = (lb[cols] + x[rows, cols]) / 2.0
            
        # Upper bounds
        mask_u = u > ub
        if np.any(mask_u):
            rows, cols = np.where(mask_u)
            u[rows, cols] = (ub[cols] + x[rows, cols]) / 2.0
            
        u = np.clip(u, lb, ub)
        
        # 6. Evaluation and Selection
        new_fitness = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Periodically check time within generation to ensure responsiveness
            if i % 20 == 0 and (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            val = func(u[i])
            nfe += 1
            new_fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                
        # 7. Update State
        improved = new_fitness < fitness
        
        if np.any(improved):
            # Update Archive with parents that are about to be replaced
            parents_to_archive = pop[improved]
            for p_vec in parents_to_archive:
                archive.append(p_vec.copy())
            
            # Maintain Archive Size
            while len(archive) > pop_size:
                archive.pop(random.randint(0, len(archive)-1))
                
            # Update Memory (Weighted Lehmer Mean)
            diff = np.abs(fitness[improved] - new_fitness[improved])
            weights = diff / np.sum(diff)
            
            imp_f = f[improved]
            imp_cr = cr[improved]
            
            # Update M_f
            mean_f = np.sum(weights * (imp_f ** 2)) / np.sum(weights * imp_f)
            M_f[k_mem] = mean_f
            
            # Update M_cr (jSO uses Weighted Lehmer for Cr as well)
            if np.sum(weights * imp_cr) > 0:
                mean_cr = np.sum(weights * (imp_cr ** 2)) / np.sum(weights * imp_cr)
            else:
                mean_cr = 0.0
            M_cr[k_mem] = mean_cr
            
            k_mem = (k_mem + 1) % H
            
            # Update Population
            pop[improved] = u[improved]
            fitness[improved] = new_fitness[improved]

    return best_fitness
