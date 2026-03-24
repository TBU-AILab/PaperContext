#The following algorithm implements **L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution)**, refined with **Clipping** for boundary handling and **jSO-inspired parameter adaptation**.
#
#### Algorithm Description
#This approach optimizes the previous L-SHADE and SHADE attempts by combining their strongest features while correcting the weaknesses identified in the results (specifically the boundary handling and population sizing).
#
#1.  **Linear Population Size Reduction (LPSR)**: The population starts at a moderate size ($15 \times D$) to balance exploration with the limited time budget, and linearly reduces to 4. This ensures the algorithm transitions from global search to rapid local convergence as time runs out.
#2.  **Clipping for Boundaries**: Unlike the previous "Midpoint" approach which performed poorly, this implementation uses **Clipping** (setting out-of-bound values to the bound). This allows the algorithm to effectively reach optima located on the edges of the search space, which was a likely failure mode of the previous attempt.
#3.  **Adaptive p-best**: The greediness of the mutation strategy adapts over time. The parameter $p$ (controlling the size of the best-solutions pool) decreases linearly from $0.20$ to $0.05$, increasing selection pressure towards the end of the run.
#4.  **Robust Time Management**: The algorithm checks the time budget not just per generation but within the evaluation loop, ensuring it utilizes strictly 100% of the available time without timing out or terminating prematurely.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Clipping and Linear Population Reduction.
    """
    start_time = time.time()
    # Safety buffer to return strictly before max_time
    limit_time = start_time + max_time - 0.05

    # --- Parameters ---
    # Population Size
    # Initial size: 15 * dim (Compromise between exploration and speed)
    init_pop_size = max(30, int(15 * dim))
    min_pop_size = 4
    
    # SHADE Memory Parameters
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Archive
    archive_size = init_pop_size
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # Pre-process bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # --- Initialization ---
    pop_size = init_pop_size
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    best_f = float('inf')
    best_x = None

    # Initial Evaluation
    # Check time strictly inside loop to avoid timeout during large initial pop
    check_interval = max(1, int(pop_size / 5))
    
    for i in range(pop_size):
        if i % check_interval == 0 and time.time() > limit_time:
             # Return best found so far
             valid = fitness != float('inf')
             if np.any(valid):
                 return np.min(fitness[valid])
             return best_f
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_f:
            best_f = val
            best_x = pop[i].copy()
            
    # Sort population for p-best selection (Best at index 0)
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Main Loop ---
    while True:
        current_time = time.time()
        if current_time > limit_time:
            return best_f
            
        # Calculate Progress (0.0 to 1.0)
        progress = (current_time - start_time) / max_time
        progress = np.clip(progress, 0, 1)
        
        # 1. Linear Population Size Reduction
        new_pop_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        new_pop_size = max(min_pop_size, new_pop_size)
        
        if new_pop_size < pop_size:
            # Reduce population (keep best, discard worst)
            pop_size = new_pop_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Reduce Archive to match pop_size if needed
            if n_archive > pop_size:
                # Remove random elements to maintain density
                keep_idxs = np.random.choice(n_archive, pop_size, replace=False)
                archive[:pop_size] = archive[keep_idxs]
                n_archive = pop_size
        
        # 2. Adaptive Parameters
        # p linearly decreases from 0.2 to 0.05 to shift from exploration to exploitation
        p_val = 0.2 - 0.15 * progress
        p_val = max(0.05, p_val)
        
        # Memory Selection
        r_idxs = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idxs]
        mu_f = mem_f[r_idxs]
        
        # CR generation ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F generation ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Check F bounds
        # If F > 1 clip to 1. If F <= 0, regenerate.
        bad_f = f <= 0
        retry = 0
        while np.any(bad_f) and retry < 10:
            f[bad_f] = mu_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
            bad_f = f <= 0
            retry += 1
        
        f = np.clip(f, 0, 1)
        f[f <= 0] = 0.001 # Safety floor
        
        # 3. Mutation (current-to-pbest/1)
        # Select p-best individuals
        n_pbest = max(2, int(pop_size * p_val))
        pbest_idxs = np.random.randint(0, n_pbest, pop_size)
        x_pbest = pop[pbest_idxs]
        
        # Select r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        hit_i = (r1 == np.arange(pop_size))
        r1[hit_i] = (r1[hit_i] + 1) % pop_size
        x_r1 = pop[r1]
        
        # Select r2 from Union(Pop, Archive) != r1, != i
        pool_size = pop_size + n_archive
        r2 = np.random.randint(0, pool_size, pop_size)
        
        # Construct x_r2
        x_r2 = np.zeros_like(pop)
        from_pop = r2 < pop_size
        from_arch = ~from_pop
        
        x_r2[from_pop] = pop[r2[from_pop]]
        if n_archive > 0:
            arch_idx = r2[from_arch] - pop_size
            arch_idx = arch_idx % n_archive
            x_r2[from_arch] = archive[arch_idx]
        else:
            # Fallback if archive empty (should be rare)
            fallback_idx = np.random.randint(0, pop_size, np.sum(from_arch))
            x_r2[from_arch] = pop[fallback_idx]
            
        # Calculate Mutant Vector
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        diff_p = x_pbest - pop
        diff_r = x_r1 - x_r2
        mutant = pop + f[:, None] * diff_p + f[:, None] * diff_r
        
        # 4. Crossover (Binomial)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        trial = np.where(mask, mutant, pop)
        
        # 5. Bound Constraints (Clipping)
        # Clipping works better for optima at boundaries than midpoint
        trial = np.clip(trial, lb, ub)
        
        # 6. Selection
        success_f = []
        success_cr = []
        diffs = []
        
        check_interval = max(1, int(pop_size / 5))
        
        for i in range(pop_size):
            # Check time periodically inside the loop
            if (i % check_interval == 0) and (time.time() > limit_time):
                return best_f
                
            f_trial = func(trial[i])
            f_old = fitness[i]
            
            if f_trial <= f_old:
                # Add parent to archive before replacement
                if n_archive < archive_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    # Randomly replace an archive member
                    ridx = np.random.randint(0, archive_size)
                    archive[ridx] = pop[i].copy()
                
                # Update Population
                pop[i] = trial[i]
                fitness[i] = f_trial
                
                success_f.append(f[i])
                success_cr.append(cr[i])
                diffs.append(f_old - f_trial)
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial[i].copy()
                    
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
            
            # Weighted Mean for CR
            new_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = new_cr
            
            # Weighted Lehmer Mean for F
            num = np.sum(w * (s_f ** 2))
            den = np.sum(w * s_f)
            new_f = 0.5
            if den > 0:
                new_f = num / den
            mem_f[k_mem] = np.clip(new_f, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 8. Sort Population (Maintenance for p-best)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_f
