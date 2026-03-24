#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution)** with a fully **Vectorized** implementation.
#
#**Key Improvements:**
#1.  **Vectorized Operations**: Unlike the previous JADE implementation which used nested loops for mutation and crossover, this code uses NumPy's array operations to generate the entire population's trial vectors simultaneously. This drastically reduces the overhead, allowing significantly more generations to be processed within the `max_time`, leading to better convergence.
#2.  **SHADE Parameter Adaptation**: It utilizes a historical memory ($M_{CR}, M_F$) to adapt the crossover rate $CR$ and mutation factor $F$. Parameters are sampled from this memory and updated based on a weighted Lehmer mean of successful individuals. This is more robust than simple adaptive schemes.
#3.  **current-to-pbest/1 Mutation with Archive**: This strategy (proven effective in the previous best attempt) is retained but optimized. It guides the search towards the top $p\%$ best solutions while maintaining diversity via an external archive of inferior solutions.
#4.  **Robust Evaluation Loop**: The function evaluation is the only part inside a Python loop (as required by the interface), but strict time checks ensure the algorithm returns the best result immediately upon timeout without wasting cycles on the rest of the population.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Vectorized SHADE (Success-History Adaptive Differential Evolution).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------
    # Configuration
    # -------------------------------
    # Population size: Fixed robust size. 
    # SHADE typically works well with N=100, but for time-constrained 
    # python execution, a slightly smaller population allows more generations.
    pop_size = max(30, min(10 * dim, 60))
    
    # Memory Size for Adaptive Parameters (H)
    H = pop_size 
    
    # -------------------------------
    # Initialization
    # -------------------------------
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Memory for SHADE (initialized to 0.5)
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer
    
    # External Archive
    # Stores inferior solutions to maintain diversity.
    archive = []
    
    best_val = float('inf')
    
    # -------------------------------
    # Initial Evaluation
    # -------------------------------
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # -------------------------------
    # Main Optimization Loop
    # -------------------------------
    while True:
        if (datetime.now() - start_time) >= time_limit:
            return best_val
            
        # --- 1. Parameter Generation (Vectorized) ---
        # Select random memory index for each individual
        r_idxs = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idxs]
        mu_f = mem_f[r_idxs]
        
        # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(mu_f, 0.1)
        # We need F > 0. If F <= 0, regenerate. If F > 1, clip to 1.
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle constraints on F vectorized
        f[f > 1] = 1.0
        
        # Regenerate non-positive F
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg):
                break
            n_neg = np.sum(mask_neg)
            # Resample only the negative ones
            f[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(n_neg)
            f[f > 1] = 1.0
            
        # --- 2. Mutation: current-to-pbest/1 (Vectorized) ---
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Identify p-best (top 10%)
        sorted_idx = np.argsort(fitness)
        p_pct = 0.1
        n_pbest = max(2, int(p_pct * pop_size))
        pbest_pool = sorted_idx[:n_pbest]
        
        # Randomly assign a pbest for each individual
        pbest_choices = np.random.choice(pbest_pool, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1: Random from pop, distinct from current i
        # We use a random shift to ensure r1 != i
        shift_r1 = np.random.randint(1, pop_size, pop_size)
        r1_indices = (np.arange(pop_size) + shift_r1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2: Random from Union(Pop, Archive), distinct from i and r1
        if len(archive) > 0:
            # Convert archive to array for indexing
            arr_archive = np.array(archive)
            pop_all = np.vstack((pop, arr_archive))
        else:
            pop_all = pop
            
        n_all = len(pop_all)
        r2_indices = np.random.randint(0, n_all, pop_size)
        
        # Fix collisions: r2 cannot be i (current) or r1
        # Note: indices 0..pop_size-1 in pop_all correspond to current pop
        current_indices = np.arange(pop_size)
        
        # Mask where r2 collides with i or r1
        bad_r2 = (r2_indices == current_indices) | (r2_indices == r1_indices)
        
        # Iteratively fix collisions (usually fast)
        while np.any(bad_r2):
            n_bad = np.sum(bad_r2)
            r2_indices[bad_r2] = np.random.randint(0, n_all, n_bad)
            bad_r2 = (r2_indices == current_indices) | (r2_indices == r1_indices)
            
        x_r2 = pop_all[r2_indices]
        
        # Calculate Mutant Vectors
        # Expand F for broadcasting: (N,) -> (N, 1)
        f_v = f[:, None]
        mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
        
        # --- 3. Crossover (Binomial) ---
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        
        # Ensure at least one parameter is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trials = np.where(cross_mask, mutant, pop)
        
        # --- 4. Bounds Constraint ---
        trials = np.clip(trials, min_b, max_b)
        
        # --- 5. Evaluation & Selection ---
        successful_f = []
        successful_cr = []
        improvement = []
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            f_tri = func(trials[i])
            
            # Greedy Selection
            if f_tri <= fitness[i]:
                # If strictly better, record success info for adaptation
                if f_tri < fitness[i]:
                    successful_f.append(f[i])
                    successful_cr.append(cr[i])
                    improvement.append(fitness[i] - f_tri)
                    # Archive the inferior parent
                    archive.append(pop[i].copy())
                
                pop[i] = trials[i]
                fitness[i] = f_tri
                
                if f_tri < best_val:
                    best_val = f_tri
                    
        # --- 6. Archive Maintenance ---
        # Keep archive size <= pop_size
        while len(archive) > pop_size:
            rm_idx = np.random.randint(0, len(archive))
            archive.pop(rm_idx)
            
        # --- 7. Update Adaptive Parameters (SHADE) ---
        if len(successful_f) > 0:
            succ_f = np.array(successful_f)
            succ_cr = np.array(successful_cr)
            imp = np.array(improvement)
            total_imp = np.sum(imp)
            
            if total_imp > 0:
                # Weights based on fitness improvement
                weights = imp / total_imp
                
                # Weighted Lehmer Mean for F
                denom = np.sum(weights * succ_f)
                if denom == 0:
                    m_f_new = 0.5
                else:
                    m_f_new = np.sum(weights * (succ_f**2)) / denom
                
                # Weighted Mean for CR
                m_cr_new = np.sum(weights * succ_cr)
                
                # Update Memory at pointer k
                mem_f[k_mem] = m_f_new
                mem_cr[k_mem] = m_cr_new
                
                # Increment memory pointer
                k_mem = (k_mem + 1) % H

    return best_val
