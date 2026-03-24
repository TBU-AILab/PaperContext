#The following algorithm implements **L-SHADE (Linear Population Size Reduction - Success-History Adaptive Differential Evolution)**, adapted for time-constrained execution.
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR)**: The algorithm starts with a larger population to effectively explore the search space and linearly reduces the population size over time (down to a minimum of 4). This strategy shifts the focus from exploration to exploitation as the time limit approaches and significantly increases the convergence speed in the final stages.
#2.  **Time-Based Progress**: Unlike standard L-SHADE which uses evaluation counts, this implementation drives the reduction and parameter adaptation based on the elapsed time relative to `max_time`, ensuring the algorithm optimally utilizes the available computational budget.
#3.  **Vectorized SHADE with Archive**: Retains the efficient vectorized operations, historical memory parameter adaptation ($M_{CR}, M_F$), and `current-to-pbest` mutation strategy with an external archive from the previous best attempt.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE (Linear Population Size Reduction - SHADE).
    Adapted for time-constrained execution in Python using Vectorization.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # ------------------------------------------------
    # L-SHADE Configuration
    # ------------------------------------------------
    # Initial Population Size (N_init)
    # We start larger to ensure good exploration, but cap it to avoid 
    # excessive overhead if the dimension is high or time is very short.
    # A cap of 120 is chosen to be robust for Python execution.
    pop_size_init = int(18 * dim)
    pop_size_init = max(30, min(pop_size_init, 120))
    
    # Minimum Population Size (N_min)
    pop_size_min = 4
    
    # Current Population Size
    pop_size = pop_size_init
    
    # External Archive Size Rate (Archive capacity = pop_size * arc_rate)
    arc_rate = 2.0
    
    # Memory Size for Adaptive Parameters (H)
    H = 6 
    
    # ------------------------------------------------
    # Initialization
    # ------------------------------------------------
    # Historical Memory for CR and F, initialized to 0.5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer
    
    # Bounds processing
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    
    # External Archive (list of arrays)
    archive = []
    
    # ------------------------------------------------
    # Initial Evaluation
    # ------------------------------------------------
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # ------------------------------------------------
    # Main Optimization Loop
    # ------------------------------------------------
    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        if elapsed >= time_limit:
            return best_val
            
        # ------------------------------------------------
        # 1. Linear Population Size Reduction (LPSR)
        # ------------------------------------------------
        # Calculate progress ratio based on Time
        prog = elapsed.total_seconds() / max_time
        if prog > 1.0: prog = 1.0
        
        # Calculate target population size
        target_size = int(round(((pop_size_min - pop_size_init) * prog) + pop_size_init))
        target_size = max(pop_size_min, target_size)
        
        # Resize if target is smaller than current
        if pop_size > target_size:
            # Sort population by fitness (ascending)
            sort_indices = np.argsort(fitness)
            
            # Keep only the top 'target_size' individuals
            pop = pop[sort_indices[:target_size]]
            fitness = fitness[sort_indices[:target_size]]
            pop_size = target_size
            
            # Resize Archive to maintain arc_rate proportionality
            curr_arc_capacity = int(pop_size * arc_rate)
            if len(archive) > curr_arc_capacity:
                # Reduce archive by removing random elements
                # (Random removal preserves diversity distribution better than oldest)
                keep_idxs = np.random.choice(len(archive), curr_arc_capacity, replace=False)
                new_archive = [archive[idx] for idx in keep_idxs]
                archive = new_archive
                
        # ------------------------------------------------
        # 2. Parameter Generation (Vectorized)
        # ------------------------------------------------
        # Select random memory index for each individual
        r_idxs = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idxs]
        mu_f = mem_f[r_idxs]
        
        # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints
        # 1. If F > 1, clip to 1.0
        f[f > 1] = 1.0
        
        # 2. If F <= 0, regenerate until positive
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg):
                break
            n_neg = np.sum(mask_neg)
            # Regenerate using the corresponding mu_f
            f[mask_neg] = mu_f[mask_neg] + 0.1 * np.random.standard_cauchy(n_neg)
            f[f > 1] = 1.0
            
        # ------------------------------------------------
        # 3. Mutation: current-to-pbest/1 (Vectorized)
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        # ------------------------------------------------
        
        # Identify p-best (top 11%)
        sorted_idx = np.argsort(fitness)
        n_pbest = max(2, int(0.11 * pop_size))
        pbest_indices = sorted_idx[:n_pbest]
        
        # Assign pbest for each individual randomly from top p%
        pbest_choices = np.random.choice(pbest_indices, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1: Random from pop, distinct from current i
        # We shift indices where r1 == i
        r1_choices = np.random.randint(0, pop_size, pop_size)
        collisions = (r1_choices == np.arange(pop_size))
        r1_choices[collisions] = (r1_choices[collisions] + 1) % pop_size
        x_r1 = pop[r1_choices]
        
        # Select r2: Random from Union(Pop, Archive), distinct from i and r1
        if len(archive) > 0:
            arr_archive = np.array(archive)
            pop_all = np.vstack((pop, arr_archive))
        else:
            pop_all = pop
            
        n_all = len(pop_all)
        r2_choices = np.random.randint(0, n_all, pop_size)
        
        # Fix collisions: r2 cannot be i (current) or r1
        # Indices in pop_all: 0..pop_size-1 correspond to current pop
        curr_indices = np.arange(pop_size)
        bad_r2 = (r2_choices == curr_indices) | (r2_choices == r1_choices)
        
        # Iteratively fix r2 collisions
        while np.any(bad_r2):
            n_bad = np.sum(bad_r2)
            r2_choices[bad_r2] = np.random.randint(0, n_all, n_bad)
            bad_r2 = (r2_choices == curr_indices) | (r2_choices == r1_choices)
            
        x_r2 = pop_all[r2_choices]
        
        # Calculate Mutant Vectors
        # Expand F for broadcasting
        f_v = f[:, None]
        mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
        
        # ------------------------------------------------
        # 4. Crossover (Binomial)
        # ------------------------------------------------
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        
        # Ensure at least one parameter is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trials = np.where(cross_mask, mutant, pop)
        
        # ------------------------------------------------
        # 5. Bounds Constraint
        # ------------------------------------------------
        trials = np.clip(trials, min_b, max_b)
        
        # ------------------------------------------------
        # 6. Evaluation & Selection
        # ------------------------------------------------
        # Storage for successful parameters
        succ_f = []
        succ_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            # Strict time check inside the loop
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            f_tri = func(trials[i])
            
            # Greedy Selection
            if f_tri <= fitness[i]:
                # If strictly better, record success for SHADE adaptation
                if f_tri < fitness[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fitness.append(fitness[i] - f_tri)
                    # Archive the inferior parent
                    archive.append(pop[i].copy())
                
                pop[i] = trials[i]
                fitness[i] = f_tri
                
                if f_tri < best_val:
                    best_val = f_tri
                    
        # ------------------------------------------------
        # 7. Archive Maintenance & Memory Update
        # ------------------------------------------------
        # Enforce archive size limit (capacity might have changed due to reduction)
        curr_arc_capacity = int(pop_size * arc_rate)
        while len(archive) > curr_arc_capacity:
            idx_rem = np.random.randint(0, len(archive))
            archive.pop(idx_rem)
            
        # Update Memory (SHADE)
        if len(succ_f) > 0:
            arr_succ_f = np.array(succ_f)
            arr_succ_cr = np.array(succ_cr)
            arr_diff = np.array(diff_fitness)
            
            # Calculate weights based on fitness improvement
            total_diff = np.sum(arr_diff)
            if total_diff > 0:
                weights = arr_diff / total_diff
                
                # Weighted Lehmer Mean for F
                num_f = np.sum(weights * (arr_succ_f**2))
                den_f = np.sum(weights * arr_succ_f)
                new_mu_f = num_f / den_f if den_f != 0 else 0.5
                
                # Weighted Mean for CR
                new_mu_cr = np.sum(weights * arr_succ_cr)
                
                # Update Memory at pointer k_mem
                mem_f[k_mem] = new_mu_f
                mem_cr[k_mem] = new_mu_cr
                
                # Increment memory pointer
                k_mem = (k_mem + 1) % H

    return best_val
