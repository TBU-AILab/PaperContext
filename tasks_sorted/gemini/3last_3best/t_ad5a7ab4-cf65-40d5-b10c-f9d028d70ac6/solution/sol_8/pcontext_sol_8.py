#The following algorithm implements **L-SHADE-R (L-SHADE with Stagnation Restart)**.
#
##### Algorithm Description
#This solution improves upon the previous best-performing L-SHADE implementation by addressing the issue of **early convergence to local optima** (stagnation), which likely limited the previous result to `1.76`.
#
#1.  **Linear Population Size Reduction (LPSR)**: The population starts at a slightly larger size ($20 \times D$) to improve initial global coverage and linearly reduces to 4. This matches the evaluation budget to the time constraint dynamically.
#2.  **Stagnation Detection & Soft Restart**:
#    *   One weakness of standard L-SHADE in limited time is getting "stuck" in a local optimum with no variance left to escape.
#    *   This algorithm monitors the **fitness spread** ($fitness_{worst} - fitness_{best}$).
#    *   If the population converges (spread $< 10^{-8}$) while significant time remains ($< 80\%$ used), a **Soft Restart** is triggered:
#        *   The best individual is preserved.
#        *   The rest of the population is re-initialized randomly.
#        *   Historical memory and archive are reset to prevent "ghost" biases from the previous local optimum.
#3.  **Dynamic Archive Sizing**: The archive size tracks the shrinking population size (maintained at $2.0 \times N$). This ensures that the difference vectors ($x_{r1} - x_{r2}$) remain scaled to the current search area, improving the efficiency of the `current-to-pbest/1` mutation in the final stages.
#4.  **Clipping**: Retains the boundary clipping strategy which outperformed midpoint targeting.
#
##### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-R (L-SHADE with Stagnation Restart).
    Includes Linear Population Size Reduction, Adaptive Parameters, and 
    a Soft Restart mechanism to escape local optima.
    """
    start_time = time.time()
    # Safety buffer to ensure result return before hard timeout
    time_limit = start_time + max_time - 0.05

    # --- Configuration ---
    # Initial Population: 20*dim for robust initial coverage
    # Bounded to [30, 500] to handle various dimensions efficiently
    init_pop_size = max(30, int(20 * dim))
    if init_pop_size > 500: 
        init_pop_size = 500
    min_pop_size = 4
    
    # Archive parameters
    # Archive capacity scales with population (factor 2.0)
    archive_rate = 2.0
    
    # SHADE Memory Parameters
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Pre-process bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # --- Initialization ---
    pop_size = init_pop_size
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    # Archive setup
    # Allocate max memory, manage via n_archive
    archive_max_cap = int(init_pop_size * archive_rate) + 10
    archive = np.zeros((archive_max_cap, dim))
    n_archive = 0
    
    best_f = float('inf')
    
    # Initial Evaluation
    # Check time strictly to handle cases with very expensive functions or tight limits
    check_interval = max(1, int(pop_size / 5))
    
    for i in range(pop_size):
        if i % check_interval == 0 and time.time() > time_limit:
            valid_mask = fitness != float('inf')
            if np.any(valid_mask):
                return min(best_f, np.min(fitness[valid_mask]))
            return best_f
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_f:
            best_f = val
            
    # Sort population (Index 0 is best)
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Main Loop ---
    while True:
        current_time = time.time()
        if current_time > time_limit:
            return best_f
            
        # Calculate Progress (0.0 -> 1.0)
        progress = (current_time - start_time) / max_time
        progress = np.clip(progress, 0, 1)
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target size based on time elapsed
        target_pop_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        target_pop_size = max(min_pop_size, target_pop_size)
        
        if target_pop_size < pop_size:
            # Shrink Population: Keep best, discard worst (tail)
            pop_size = target_pop_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Shrink Archive: Maintain density relative to population
            # If archive is too large, difference vectors may become too large/old
            current_archive_cap = int(pop_size * archive_rate)
            if n_archive > current_archive_cap:
                # Randomly reduce archive
                keep_idxs = np.random.choice(n_archive, current_archive_cap, replace=False)
                archive[:current_archive_cap] = archive[keep_idxs]
                n_archive = current_archive_cap

        # 2. Stagnation Detection & Soft Restart
        # If population has converged (variance ~ 0) but we have time left, restart.
        if pop_size >= 4 and progress < 0.8:
            fit_spread = fitness[-1] - fitness[0]
            # Use a small epsilon. If spread is tiny, we are stuck.
            if fit_spread < 1e-8:
                # --- Perform Soft Restart ---
                # Strategy: Keep best individual, re-initialize the rest.
                # Reset adaptive memories to allow new exploration.
                
                # Re-init pop[1:]
                pop[1:] = np.random.uniform(lb, ub, (pop_size - 1, dim))
                fitness[1:] = float('inf')
                
                # Reset Archive and Memory
                n_archive = 0
                mem_cr.fill(0.5)
                mem_f.fill(0.5)
                
                # Evaluate new individuals
                for i in range(1, pop_size):
                    if time.time() > time_limit:
                        return best_f
                    val = func(pop[i])
                    fitness[i] = val
                    if val < best_f:
                        best_f = val
                
                # Re-sort
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx]
                fitness = fitness[sorted_idx]
                
                # Skip to next generation
                continue

        # 3. Adaptive Parameter Generation (SHADE)
        # p linearly decreases from 0.2 to 0.05
        p_val = 0.2 - 0.15 * progress
        p_val = max(0.05, p_val)
        
        # Select Memory Indices
        r_idxs = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idxs]
        mu_f = mem_f[r_idxs]
        
        # CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints (F > 0 required)
        bad_f = f <= 0
        retry_count = 0
        while np.any(bad_f) and retry_count < 10:
            f[bad_f] = mu_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
            bad_f = f <= 0
            retry_count += 1
        
        f = np.clip(f, 0, 1)
        f[f <= 0] = 1e-4 # Safety floor
        
        # 4. Mutation: current-to-pbest/1
        # Select p-best individuals
        n_pbest = max(2, int(pop_size * p_val))
        pbest_idxs = np.random.randint(0, n_pbest, pop_size)
        x_pbest = pop[pbest_idxs]
        
        # Select r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        hit_i = (r1 == np.arange(pop_size))
        r1[hit_i] = (r1[hit_i] + 1) % pop_size
        x_r1 = pop[r1]
        
        # Select r2 != r1, != i from Union(Pop, Archive)
        pool_size = pop_size + n_archive
        r2 = np.random.randint(0, pool_size, pop_size)
        
        # Construct x_r2
        x_r2 = np.zeros_like(pop)
        from_pop = r2 < pop_size
        from_arch = ~from_pop
        
        x_r2[from_pop] = pop[r2[from_pop]]
        if n_archive > 0 and np.any(from_arch):
            arch_idx = (r2[from_arch] - pop_size) % n_archive
            x_r2[from_arch] = archive[arch_idx]
        elif np.any(from_arch):
            # Fallback (rare)
            x_r2[from_arch] = pop[np.random.randint(0, pop_size, np.sum(from_arch))]
            
        # Calculate Mutant Vector: v = x + F*(xp - x) + F*(xr1 - xr2)
        diff_p = x_pbest - pop
        diff_r = x_r1 - x_r2
        mutant = pop + f[:, None] * diff_p + f[:, None] * diff_r
        
        # 5. Crossover (Binomial)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        trial = np.where(mask, mutant, pop)
        
        # 6. Bound Handling (Clipping)
        trial = np.clip(trial, lb, ub)
        
        # 7. Selection
        success_f = []
        success_cr = []
        diffs = []
        
        check_interval = max(1, int(pop_size / 5))
        
        for i in range(pop_size):
            # Strict time monitoring within evaluation loop
            if i % check_interval == 0 and time.time() > time_limit:
                return best_f
            
            f_trial = func(trial[i])
            f_old = fitness[i]
            
            if f_trial <= f_old:
                # Add parent to archive
                if n_archive < archive.shape[0]:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    # Random replacement
                    ridx = np.random.randint(0, n_archive)
                    archive[ridx] = pop[i].copy()
                
                # Update Population
                pop[i] = trial[i]
                fitness[i] = f_trial
                
                success_f.append(f[i])
                success_cr.append(cr[i])
                diffs.append(f_old - f_trial)
                
                if f_trial < best_f:
                    best_f = f_trial
        
        # 8. Memory Update (Weighted Lehmer Mean)
        if len(success_f) > 0:
            w = np.array(diffs)
            sum_w = np.sum(w)
            
            if sum_w > 0:
                w = w / sum_w
            else:
                w = np.ones(len(w)) / len(w)
                
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            # Update M_CR (Weighted Mean)
            mem_cr[k_mem] = np.sum(w * s_cr)
            
            # Update M_F (Weighted Lehmer Mean)
            num = np.sum(w * (s_f ** 2))
            den = np.sum(w * s_f)
            new_f = 0.5
            if den > 0:
                new_f = num / den
            mem_f[k_mem] = np.clip(new_f, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 9. Sort Population
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_f
