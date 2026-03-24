#The following algorithm implements **L-SHADE-RSP (Linear Population Size Reduction Success-History Adaptive Differential Evolution)**, refined for time-constrained optimization.
#
#### Algorithm Description
#This solution is an evolution of the successful L-SHADE approach (Algorithm 1), incorporating specific tunings to improve convergence and robustness:
#
#1.  **Optimized Linear Population Size Reduction (LPSR)**:
#    *   Starts with a population size of **$18 \times D$** (following CEC benchmark recommendations for L-SHADE), capped at 300 to ensure sufficient generations can run within the time limit.
#    *    Linearly reduces the population size to **4** based on the elapsed time. This forces the algorithm to transition from exploration to fine-grained exploitation (local search) as the deadline approaches.
#    *   **Archive Shrinking**: Crucially, the external archive size also shrinks dynamically to match the population size. This prevents "ghost" solutions from early exploration phases from disrupting the convergence of the difference vectors in the final stages.
#
#2.  **Adaptive Parameters (SHADE)**:
#    *   Uses a historical memory ($H=6$) to adapt Mutation ($F$) and Crossover ($CR$) rates.
#    *   Updates memory using a **Weighted Lehmer Mean** of successful parameters, giving more weight to updates that produced larger fitness improvements.
#
#3.  **Boundary Handling (Clipping)**:
#    *   Retains the **Clipping** strategy (setting out-of-bound values to the bounds), as previous results showed this significantly outperforms midpoint targeting for this specific problem class.
#
#4.  **Greedy Mutation Strategy**:
#    *   Uses `current-to-pbest/1` mutation.
#    *   The $p$ parameter (defining the "top best" pool) adapts linearly from **0.20** down to **0.05**. This broadens the search initially and focuses strictly on the elite solutions at the end.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-RSP with time-based Linear Population Size Reduction.
    """
    start_time = time.time()
    # Safety buffer to ensure return before hard timeout
    time_limit = start_time + max_time - 0.05

    # --- Configuration ---
    # Population Sizing
    # Start with 18 * dim (Standard for L-SHADE), capped to ensure generations run
    init_pop_size = max(30, int(18 * dim))
    if init_pop_size > 300:
        init_pop_size = 300
    
    min_pop_size = 4

    # SHADE Memory Parameters
    H = 6  # History size
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
    
    # Archive: Stores parent vectors that were improved upon
    # Size tracks population size
    archive = np.zeros((init_pop_size, dim))
    n_archive = 0

    best_f = float('inf')
    
    # Evaluate Initial Population
    # Check time strictly to prevent timeout during initial heavy evaluation
    check_interval = max(1, int(pop_size / 5))

    for i in range(pop_size):
        if i % check_interval == 0 and time.time() > time_limit:
             # Return best found so far
             valid_mask = fitness != float('inf')
             if np.any(valid_mask):
                 return np.min(fitness[valid_mask])
             return best_f
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_f:
            best_f = val

    # Sort population by fitness (Best at index 0)
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]

    # --- Main Loop ---
    while True:
        current_time = time.time()
        if current_time > time_limit:
            return best_f

        # Calculate Progress (0.0 to 1.0)
        progress = (current_time - start_time) / max_time
        progress = min(1.0, max(0.0, progress))

        # 1. Linear Population Size Reduction
        new_pop_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        new_pop_size = max(min_pop_size, new_pop_size)

        if new_pop_size < pop_size:
            # Shrink Population: Keep best, discard worst (tail)
            pop_size = new_pop_size
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Shrink Archive: Must track pop_size to maintain convergence pressure
            if n_archive > pop_size:
                # Randomly reduce archive
                keep_idxs = np.random.choice(n_archive, pop_size, replace=False)
                archive[:pop_size] = archive[keep_idxs]
                n_archive = pop_size

        # 2. Adaptive Parameter Generation
        # p-best parameter linearly decreases from 0.2 to 0.05
        p_val = 0.2 - 0.15 * progress
        p_val = max(0.05, p_val)

        # Select random memory index for each individual
        r_idxs = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idxs]
        mu_f = mem_f[r_idxs]

        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)

        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints
        # F > 1 -> clip to 1. F <= 0 -> regenerate.
        bad_f = f <= 0
        retry_count = 0
        while np.any(bad_f) and retry_count < 20:
            f[bad_f] = mu_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
            bad_f = f <= 0
            retry_count += 1
        
        f = np.clip(f, 0, 1)
        f[f <= 0] = 1e-4 # Safety floor

        # 3. Mutation: current-to-pbest/1
        # Select p-best individuals (top p%)
        n_pbest = max(2, int(pop_size * p_val))
        pbest_idxs = np.random.randint(0, n_pbest, pop_size)
        x_pbest = pop[pbest_idxs] # pop is already sorted

        # Select r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        hit_i = (r1 == np.arange(pop_size))
        r1[hit_i] = (r1[hit_i] + 1) % pop_size
        x_r1 = pop[r1]

        # Select r2 != r1, != i from Union(Pop, Archive)
        pool_size = pop_size + n_archive
        r2 = np.random.randint(0, pool_size, pop_size)
        
        # Build x_r2
        x_r2 = np.zeros((pop_size, dim))
        from_pop = r2 < pop_size
        from_arch = ~from_pop
        
        x_r2[from_pop] = pop[r2[from_pop]]
        if n_archive > 0:
            # Map index to archive range
            arch_idx = r2[from_arch] - pop_size
            # Safety modulo
            x_r2[from_arch] = archive[arch_idx % n_archive]
        else:
            # Fallback if archive empty (rare)
            x_r2[from_arch] = pop[np.random.randint(0, pop_size, np.sum(from_arch))]

        # Compute Mutant Vector: v = x + F*(xp - x) + F*(xr1 - xr2)
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

        # 6. Selection and Archive Update
        success_f = []
        success_cr = []
        diffs = []

        # Recalculate check interval as pop_size changes
        check_interval = max(1, int(pop_size / 5))

        for i in range(pop_size):
            if (i % check_interval == 0) and (time.time() > time_limit):
                return best_f
            
            f_trial = func(trial[i])
            f_old = fitness[i]

            if f_trial <= f_old:
                # Add parent to archive (replace random if full)
                # Archive max size is current pop_size
                if n_archive < pop_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    ridx = np.random.randint(0, pop_size)
                    archive[ridx] = pop[i].copy()
                
                # Update Population
                pop[i] = trial[i]
                fitness[i] = f_trial
                
                success_f.append(f[i])
                success_cr.append(cr[i])
                diffs.append(f_old - f_trial)

                if f_trial < best_f:
                    best_f = f_trial
        
        # 7. Update SHADE Memory
        if len(success_f) > 0:
            w = np.array(diffs)
            sum_w = np.sum(w)
            
            # Weighted Mean based on fitness improvement
            if sum_w > 0:
                w = w / sum_w
            else:
                w = np.ones(len(w)) / len(w)
            
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)

            # Update M_CR (Weighted Mean)
            new_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = new_cr
            
            # Update M_F (Weighted Lehmer Mean)
            num = np.sum(w * (s_f ** 2))
            den = np.sum(w * s_f)
            new_f = 0.5
            if den > 0:
                new_f = num / den
            mem_f[k_mem] = np.clip(new_f, 0, 1)

            k_mem = (k_mem + 1) % H

        # 8. Sort Population for next generation (Maintenance for p-best)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

    return best_f
