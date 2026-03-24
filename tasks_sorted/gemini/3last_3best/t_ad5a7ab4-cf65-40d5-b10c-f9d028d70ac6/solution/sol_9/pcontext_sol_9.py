#The following algorithm implements **L-SHADE-E (L-SHADE with Expanded Archive)**.
#
##### Algorithm Description
#This solution builds upon the best-performing L-SHADE implementation (Algorithm 3, which achieved 1.76) by addressing a critical theoretical limitation identified in its archive management, while preserving the successful Linear Population Size Reduction (LPSR) and Clipping strategies.
#
#1.  **Expanded Dynamic Archive**:
#    *   **Improvement**: Previous algorithms (2 and 3) aggressively restricted the external archive size to match the current population size ($|A| \le N$). In standard L-SHADE and its successful variants (like jSO), the archive size is typically maintained at a rate of **2.0 to 2.6 times the population size**.
#    *   **Effect**: By allowing the archive to be larger than the population ($|A| \approx 2.6 \times N$), the algorithm maintains a richer history of "good" past solutions. This significantly improves the diversity of the difference vectors ($x_{r1} - x_{r2}$) used in the `current-to-pbest/1` mutation, preventing premature convergence of the mutation directions even as the population shrinks.
#
#2.  **Optimized Population Sizing**:
#    *   Uses an initial population size of **$18 \times D$** (capped at 300). This balances initial exploration coverage with the need to perform enough generations within the limited time budget.
#
#3.  **Refined Parameter Adaptation**:
#    *   **P-best**: The greediness parameter $p$ adapts from **0.2** down to **0.11** (instead of 0.05). Keeping the final $p$ slightly higher prevents the mutation strategy from becoming solely focused on the very best individual too early, maintaining a small degree of exploration around the optimum to refine precision.
#    *   **Memory**: Uses a history size of $H=6$ with standard SHADE weighted memory updates.
#
#4.  **Time Management**:
#    *   Includes strict time checks within the initialization and selection loops to ensuring the result is returned exactly within the `max_time` limit without timeout.
#
##### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-E (Expanded Archive).
    Features Linear Population Size Reduction, Expanded Archive (2.6x),
    and SHADE adaptive parameters.
    """
    start_time = time.time()
    # Safety buffer to ensure return before hard timeout
    time_limit = start_time + max_time - 0.05

    # --- Configuration ---
    # Population Sizing
    # Standard L-SHADE uses ~18*D. We cap at 300 to ensure sufficient generations
    # run within the time limit for expensive functions.
    init_pop_size = int(18 * dim)
    if init_pop_size < 30: init_pop_size = 30
    if init_pop_size > 300: init_pop_size = 300
    
    min_pop_size = 4
    
    # Archive parameters
    # A rate of 2.6 allows the archive to hold more history than the population,
    # providing better difference vectors for mutation.
    archive_rate = 2.6
    
    # SHADE Memory Parameters
    H = 6
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
    # Maximum memory allocation needed
    max_archive_memory = int(init_pop_size * archive_rate) + 10
    archive = np.zeros((max_archive_memory, dim))
    n_archive = 0
    
    best_f = float('inf')
    
    # Initial Evaluation
    # Check time strictly to prevent timeout during initial heavy evaluation
    check_interval = max(1, int(pop_size / 5))
    
    for i in range(pop_size):
        if i % check_interval == 0 and time.time() > time_limit:
            # Return best found so far if time is up
            valid_mask = fitness != float('inf')
            if np.any(valid_mask):
                return np.min(fitness[valid_mask])
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
        target_pop_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        target_pop_size = max(min_pop_size, target_pop_size)
        
        if target_pop_size < pop_size:
            # Shrink Population: Keep best, discard worst
            pop_size = target_pop_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Shrink Archive
            # Archive capacity scales with CURRENT population size (2.6 * N)
            current_archive_cap = int(pop_size * archive_rate)
            if n_archive > current_archive_cap:
                # Randomly reduce archive to current capacity
                keep_idxs = np.random.choice(n_archive, current_archive_cap, replace=False)
                archive[:current_archive_cap] = archive[keep_idxs]
                n_archive = current_archive_cap
        
        # Define current archive capacity for this generation
        current_archive_cap = int(pop_size * archive_rate)

        # 2. Adaptive Parameter Generation (SHADE)
        # p linearly decreases from 0.2 to 0.11
        p_val = 0.2 - 0.09 * progress
        p_val = max(0.05, p_val)
        
        # Select Memory Indices
        r_idxs = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idxs]
        mu_f = mem_f[r_idxs]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints (F > 0 required)
        bad_f = f <= 0
        retry_count = 0
        while np.any(bad_f) and retry_count < 10:
            f[bad_f] = mu_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
            bad_f = f <= 0
            retry_count += 1
            
        f = np.clip(f, 0, 1)
        f[f <= 0] = 0.001 # Safety floor
        
        # 3. Mutation: current-to-pbest/1
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
        
        # Simple collision fix for r2 == r1
        hit_r1 = (r2 == r1)
        r2[hit_r1] = (r2[hit_r1] + 1) % pool_size
        
        # Construct x_r2
        x_r2 = np.zeros((pop_size, dim))
        from_pop = r2 < pop_size
        from_arch = ~from_pop
        
        x_r2[from_pop] = pop[r2[from_pop]]
        if n_archive > 0:
            arch_idx = (r2[from_arch] - pop_size) % n_archive
            x_r2[from_arch] = archive[arch_idx]
        elif np.any(from_arch):
            # Fallback (rare race condition or empty archive logic)
            x_r2[from_arch] = pop[np.random.randint(0, pop_size, np.sum(from_arch))]
            
        # Calculate Mutant Vector: v = x + F*(xp - x) + F*(xr1 - xr2)
        diff_p = x_pbest - pop
        diff_r = x_r1 - x_r2
        mutant = pop + f[:, None] * diff_p + f[:, None] * diff_r
        
        # 4. Crossover (Binomial)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        trial = np.where(mask, mutant, pop)
        
        # 5. Bound Constraints (Clipping)
        trial = np.clip(trial, lb, ub)
        
        # 6. Selection & Archive Update
        success_f = []
        success_cr = []
        diffs = []
        
        check_interval = max(1, int(pop_size / 5))
        
        for i in range(pop_size):
            if i % check_interval == 0 and time.time() > time_limit:
                return best_f
                
            f_trial = func(trial[i])
            f_old = fitness[i]
            
            if f_trial <= f_old:
                # Add parent to archive
                if n_archive < current_archive_cap:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    # Random replacement if full
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
                    
        # 7. Update SHADE Memory (Weighted Lehmer Mean)
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
            
        # 8. Sort Population for next generation
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_f
