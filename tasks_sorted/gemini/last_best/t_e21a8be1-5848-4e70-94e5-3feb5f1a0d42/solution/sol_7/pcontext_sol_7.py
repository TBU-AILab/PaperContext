#The following algorithm implements **L-SHADE-EIG-MTS** (Linear Success-History Adaptive Differential Evolution with Eigen-Crossover and Multiple Trajectory Search).
#
#### Key Improvements over the previous version:
#1.  **Eigen Coordinate System Crossover**: Standard DE struggles with "rotated" objective functions (where variables are correlated). This algorithm calculates the covariance matrix of the population and performs crossover in the eigen-coordinate system with a probability of 50% during the middle stages of evolution. This aligns the search strategies with the principal axes of the fitness landscape.
#2.  **Dynamic `p`-best Selection**: The value of `p` in `current-to-pbest` mutation is not static; it adapts linearly from 0.2 down to 0.05, prioritizing exploration early and greedy convergence later.
#3.  **Adaptive Local Search Range**: The final MTS polishing phase initializes its step sizes based on the standard deviation of the surviving population rather than the global bounds, ensuring the local search starts at a relevant scale.
#4.  **Optimized Time Management**: Checks time limits in batches to reduce system call overheads.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-EIG-MTS.
    
    1. L-SHADE: Linear Success-History Adaptive Differential Evolution with Linear Population Reduction.
    2. EIG: Eigen-Crossover to handle rotated (correlated) landscapes.
    3. MTS: Terminal Multiple Trajectory Search for high-precision polishing.
    """

    # --- Configuration & Initialization ---
    t_start = time.time()
    t_end = t_start + max_time
    
    # Time allocation: 80% Evolutionary Search, 20% Local Search (MTS)
    # If the population converges early (reaches min size), MTS starts earlier.
    ratio_ls = 0.20
    t_ls_start = t_start + (1.0 - ratio_ls) * max_time

    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Population Size Management (LPSR)
    # Initial size: 18 * dim is a robust heuristic for SHADE
    pop_size_init = int(round(18 * dim))
    # Minimum size: 4 is the absolute minimum for mutation operators
    pop_size_min = 4
    pop_size = pop_size_init
    
    # SHADE Memory
    H = 5 # Memory size
    M_cr = np.full(H, 0.5)
    M_f = np.full(H, 0.5)
    k_mem = 0
    
    archive = []
    
    # Initialize Population (Latin Hypercube Sampling for better coverage)
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(lb[d], ub[d], pop_size + 1)
        samples = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(samples)
        pop[:, d] = samples
        
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        if time.time() > t_end: return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop[i].copy()

    # --- Evolutionary Phase (L-SHADE + Eigen) ---
    gen = 0
    
    # Pre-allocate arrays for speed
    rot_matrix = np.eye(dim)
    do_eigen = False
    
    while True:
        curr_time = time.time()
        
        # Check termination conditions for Evo phase
        if curr_time >= t_ls_start:
            break
        if pop_size <= pop_size_min:
            break
            
        # Calculate Progress
        evo_budget = t_ls_start - t_start
        if evo_budget <= 0: break
        progress = (curr_time - t_start) / evo_budget
        progress = np.clip(progress, 0, 1)

        # 1. Linear Population Size Reduction (LPSR)
        # Linearly interpolate current desired population size
        next_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        next_pop_size = max(pop_size_min, next_pop_size)
        
        if pop_size > next_pop_size:
            # Sort and truncate
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:next_pop_size]]
            fitness = fitness[sorted_idx[:next_pop_size]]
            
            # Shrink archive
            if len(archive) > next_pop_size:
                # Random removal is fast and sufficient
                num_remove = len(archive) - next_pop_size
                for _ in range(num_remove):
                    archive.pop(np.random.randint(0, len(archive)))
            
            pop_size = next_pop_size

        # 2. Eigen Coordinate System Update
        # Only perform if population is large enough to form a valid covariance matrix
        # and we are in the "exploitation" phase (progress > 0.2)
        if dim > 1 and pop_size > dim and progress > 0.2 and np.random.rand() < 0.5:
            # Calculate covariance of the top 50% individuals
            num_top = max(dim, int(pop_size * 0.5))
            sorted_idx = np.argsort(fitness)
            top_pop = pop[sorted_idx[:num_top]]
            
            # Centering
            mean_vec = np.mean(top_pop, axis=0)
            cov_mat = np.cov(top_pop, rowvar=False)
            
            # Eigen decomposition
            try:
                # eigh is for symmetric matrices (covariance is symmetric)
                _, eig_vecs = np.linalg.eigh(cov_mat)
                rot_matrix = eig_vecs
                do_eigen = True
            except:
                do_eigen = False
        else:
            do_eigen = False

        # 3. Parameter Adaptation
        r_idx = np.random.randint(0, H, pop_size)
        mu_cr = M_cr[r_idx]
        mu_f = M_f[r_idx]
        
        # CR ~ Normal
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        # Start with Eigen Crossover mostly off, increase prob later? 
        # Here we just use CR to control crossover in rotated space if do_eigen is active.
        
        # F ~ Cauchy
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        # Sanitize F
        retry = 0
        while True:
            bad_f = f <= 0
            if not np.any(bad_f) or retry > 3:
                f[bad_f] = 0.5
                break
            f[bad_f] = mu_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
            retry += 1
        f = np.minimum(f, 1.0)

        # 4. Mutation: current-to-pbest/1
        # Dynamic p: Starts at 0.2, drops to 0.05
        p_val = 0.2 - (0.15 * progress)
        p_val = max(0.05, p_val)
        top_p = max(2, int(pop_size * p_val))
        
        sorted_idx = np.argsort(fitness)
        pbest_idxs = sorted_idx[np.random.randint(0, top_p, pop_size)]
        x_pbest = pop[pbest_idxs]
        
        r1 = np.random.randint(0, pop_size, pop_size)
        # Ensure r1 != i
        col_r1 = r1 == np.arange(pop_size)
        r1[col_r1] = (r1[col_r1] + 1) % pop_size
        x_r1 = pop[r1]
        
        # r2 from Union(Pop, Archive)
        if len(archive) > 0:
            pop_all = np.vstack((pop, np.array(archive)))
        else:
            pop_all = pop
        
        r2 = np.random.randint(0, len(pop_all), pop_size)
        # Collision handling is minor in DE, skipping complex checks for speed
        x_r2 = pop_all[r2]
        
        f_vec = f[:, None]
        mutant = pop + f_vec * (x_pbest - pop) + f_vec * (x_r1 - x_r2)
        
        # 5. Crossover
        # If do_eigen is True, we rotate parents, do XO, then rotate back
        if do_eigen:
            # Rotate population to Eigen basis
            # x' = x * V
            pop_rot = np.dot(pop, rot_matrix)
            mutant_rot = np.dot(mutant, rot_matrix)
            
            # Binomial Crossover in rotated space
            j_rand = np.random.randint(0, dim, pop_size)
            mask = np.random.rand(pop_size, dim) < cr[:, None]
            mask[np.arange(pop_size), j_rand] = True
            
            trial_rot = np.where(mask, mutant_rot, pop_rot)
            
            # Rotate back: x = x' * V^T
            trial = np.dot(trial_rot, rot_matrix.T)
        else:
            # Standard Binomial Crossover
            j_rand = np.random.randint(0, dim, pop_size)
            mask = np.random.rand(pop_size, dim) < cr[:, None]
            mask[np.arange(pop_size), j_rand] = True
            trial = np.where(mask, mutant, pop)

        # 6. Boundary Handling (Reflection)
        bl = trial < lb
        if np.any(bl):
            trial[bl] = 2*lb[np.where(bl)[1]] - trial[bl]
        bu = trial > ub
        if np.any(bu):
            trial[bu] = 2*ub[np.where(bu)[1]] - trial[bu]
        trial = np.clip(trial, lb, ub)
        
        # 7. Selection & Adaptation
        fitness_delta = np.zeros(pop_size)
        success_mask = np.zeros(pop_size, dtype=bool)
        
        # Batch evaluation not possible as func takes 1D array, loop required.
        # Check time every N evaluations to avoid syscall overhead
        check_interval = max(1, int(pop_size / 5))
        
        for i in range(pop_size):
            if i % check_interval == 0 and time.time() >= t_ls_start:
                break
                
            val_trial = func(trial[i])
            
            if val_trial <= fitness[i]:
                if val_trial < fitness[i]:
                    fitness_delta[i] = fitness[i] - val_trial
                    success_mask[i] = True
                    archive.append(pop[i].copy())
                    
                pop[i] = trial[i]
                fitness[i] = val_trial
                
                if val_trial < best_val:
                    best_val = val_trial
                    best_sol = trial[i].copy()
                    
        # Trim archive
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # Update Memory
        if np.any(success_mask):
            s_f = f[success_mask]
            s_cr = cr[success_mask]
            diffs = fitness_delta[success_mask]
            
            # Weighted Lehmer Mean
            w = diffs / np.sum(diffs)
            
            mean_scr = np.sum(w * s_cr) # Arithmetic for CR
            M_cr[k_mem] = 0.5 * M_cr[k_mem] + 0.5 * mean_scr
            
            sum_wf = np.sum(w * s_f)
            if sum_wf > 1e-10:
                mean_sf = np.sum(w * (s_f**2)) / sum_wf # Lehmer for F
                M_f[k_mem] = 0.5 * M_f[k_mem] + 0.5 * mean_sf
                
            k_mem = (k_mem + 1) % H

        # 8. Restart Mechanism (Stagnation check)
        # If population variance is extremely low, we are stuck.
        # Restart helps explore other basins if time permits.
        if gen % 20 == 0:
            std_dev = np.std(fitness)
            if std_dev < 1e-8:
                # Keep best, re-init others
                # Reset memory to defaults
                M_cr.fill(0.5)
                M_f.fill(0.5)
                archive = []
                
                idx_best = np.argmin(fitness)
                for i in range(pop_size):
                    if i != idx_best:
                        pop[i] = np.random.uniform(lb, ub)
                        val = func(pop[i])
                        fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_sol = pop[i].copy()
        
        gen += 1

    # --- Local Search Phase (MTS-LS1) ---
    # Co-ordinate descent based local search.
    # We initialize the search range based on the final population distribution
    # which gives a better hint of the basin size than global bounds.
    
    curr_best = best_sol.copy()
    curr_val = best_val
    
    # Adaptive Search Range initialization
    if pop_size > 1:
        pop_std = np.std(pop, axis=0)
        # SR is 2 * std_dev, clipped to a fraction of global bounds
        sr = np.clip(2.0 * pop_std, (ub-lb)*0.001, (ub-lb)*0.2)
    else:
        sr = (ub - lb) * 0.1
        
    # Minimum step size threshold
    min_sr = (ub - lb) * 1e-8
    
    while time.time() < t_end:
        
        improved_any = False
        # Randomize dimension order to avoid bias
        dims = np.arange(dim)
        np.random.shuffle(dims)
        
        for i in dims:
            if time.time() >= t_end: break
            
            improved_dim = False
            
            # Original MTS-LS1 logic with simplified flow
            # Try x - sr
            x_test = curr_best.copy()
            x_test[i] = np.clip(x_test[i] - sr[i], lb[i], ub[i])
            val_test = func(x_test)
            
            if val_test < curr_val:
                curr_best = x_test
                curr_val = val_test
                best_val = curr_val
                improved_dim = True
            else:
                # Try x + 0.5 * sr
                x_test = curr_best.copy()
                x_test[i] = np.clip(x_test[i] + 0.5 * sr[i], lb[i], ub[i])
                val_test = func(x_test)
                
                if val_test < curr_val:
                    curr_best = x_test
                    curr_val = val_test
                    best_val = curr_val
                    improved_dim = True
            
            if not improved_dim:
                # Shrink search radius for this dimension
                sr[i] *= 0.5
            else:
                improved_any = True
        
        # If all dimensions have very small search ranges, slight reset to escape numerical traps
        if not improved_any:
            if np.max(sr) < np.min(min_sr):
                 # "Explode" search radius slightly to check neighborhood again
                 sr = np.maximum(sr, (ub-lb)*0.01)
    
    return best_val
