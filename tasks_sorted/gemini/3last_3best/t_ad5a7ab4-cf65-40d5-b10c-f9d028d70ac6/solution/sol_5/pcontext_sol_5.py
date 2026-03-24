#The following algorithm implements **L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution)**, specifically tuned for time-constrained black-box optimization.
#
##### Algorithm Description
#L-SHADE is an advanced evolution of the SHADE algorithm that has won several IEEE CEC competitions. This implementation includes specific improvements to address the previous suboptimal performance:
#1.  **Linear Population Size Reduction (LPSR)**: The population size ($N$) starts large ($18 \times D$) to explore the landscape and linearly decreases to a minimum of 4 as the time budget is consumed. This forces the algorithm to shift from exploration to exploitation dynamically.
#2.  **Midpoint-Target Boundary Correction**: Instead of simply clipping values to bounds (which can cause stagnation at the edges) or regenerating them (random), this implementation sets out-of-bound parameters to the midpoint between the parent value and the bound. This preserves the search direction while strictly enforcing constraints.
#3.  **Adaptive Parameters**: $F$ (Mutation) and $CR$ (Crossover) are adapted using historical memory of successful updates. $F$ uses a Cauchy distribution and $CR$ uses a Normal distribution, guided by a Weighted Lehmer Mean of successful historical values.
#4.  **Dynamic p-best Mutation**: The greediness of the mutation strategy (`current-to-pbest/1`) is adjusted over time. The portion of best individuals ($p$) used for guidance shrinks from $20\%$ to $5\%$, increasing convergence pressure in the final stages.
#
##### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with midpoint-target boundary correction
    and linear population size reduction based on time budget.
    """
    start_time = time.time()
    # Safety buffer to ensure return before hard timeout
    time_limit = start_time + max_time - 0.05
    
    # --- Configuration ---
    # Initial population size: Based on CEC benchmarks (18 * dim)
    # Capped at 350 to ensure sufficient generations within limited time
    pop_size_init = min(350, max(30, int(18 * dim)))
    pop_size_min = 4
    
    # SHADE Memory Parameters
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Pre-process bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # --- Initialization ---
    pop_size = pop_size_init
    # Initialize population randomly within bounds
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    # Archive for maintaining diversity (stores successful parents)
    # Capacity tracks population size
    archive = np.zeros((pop_size_init, dim))
    n_archive = 0
    
    best_val = float('inf')
    
    # Evaluate Initial Population
    # Check time strictly during initialization
    for i in range(pop_size):
        if time.time() > time_limit:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            
    # Sort population by fitness (best first) for p-best selection
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Main Loop ---
    while True:
        t_now = time.time()
        if t_now > time_limit:
            return best_val
            
        # Calculate Progress (0.0 to 1.0)
        # Plan to finish reduction at 95% of time to allow final convergence
        elapsed = t_now - start_time
        progress = min(1.0, elapsed / (max_time * 0.95))
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate new target size
        target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        target_size = max(pop_size_min, target_size)
        
        # Resize if necessary
        if pop_size > target_size:
            pop_size = target_size
            # Since pop is sorted, we discard the worst individuals (tail of array)
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Resize Archive: remove random elements if archive is too big relative to pop
            if n_archive > pop_size:
                keep_idxs = np.random.choice(n_archive, pop_size, replace=False)
                archive[:pop_size] = archive[keep_idxs]
                n_archive = pop_size
                
        # 2. Parameter Generation (Adaptive)
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idxs]
        m_f = mem_f[r_idxs]
        
        # CR ~ Normal(m_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(m_f, 0.1)
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        # F constraint handling: if F > 1 clip to 1; if F <= 0 regenerate
        bad_f = f <= 0
        retry = 0
        while np.any(bad_f) and retry < 5:
            f[bad_f] = m_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
            bad_f = f <= 0
            retry += 1
        f = np.clip(f, 0, 1)
        f[f <= 0] = 0.01 # Safety floor
        
        # 3. Mutation: current-to-pbest/1
        # p linearly decreases from 0.2 to 0.05 to increase exploitation pressure
        p_val = 0.2 - 0.15 * progress
        p_val = max(0.05, p_val)
        
        # Select p-best targets
        n_pbest = max(2, int(pop_size * p_val))
        pbest_idxs = np.random.randint(0, n_pbest, pop_size)
        x_pbest = pop[pbest_idxs]
        
        # Select r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        hit_i = (r1 == np.arange(pop_size))
        r1[hit_i] = (r1[hit_i] + 1) % pop_size
        x_r1 = pop[r1]
        
        # Select r2 != r1, r2 != i from Union(Pop, Archive)
        pool_size = pop_size + n_archive
        r2 = np.random.randint(0, pool_size, pop_size)
        
        # Handle r2 collisions (approximate for speed)
        hit_r1 = (r2 == r1)
        r2[hit_r1] = (r2[hit_r1] + 1) % pool_size
        hit_i_r2 = (r2 == np.arange(pop_size))
        r2[hit_i_r2] = (r2[hit_i_r2] + 1) % pool_size
        
        # Construct x_r2
        x_r2 = np.zeros((pop_size, dim))
        from_pop = r2 < pop_size
        from_arch = ~from_pop
        
        x_r2[from_pop] = pop[r2[from_pop]]
        if n_archive > 0:
            arch_idx = r2[from_arch] - pop_size
            arch_idx = arch_idx % n_archive # Safety wrap
            x_r2[from_arch] = archive[arch_idx]
            
        # Compute Mutant Vector: v = x + F*(xp - x) + F*(xr1 - xr2)
        diff_p = x_pbest - pop
        diff_r = x_r1 - x_r2
        mutant = pop + f[:, None] * diff_p + f[:, None] * diff_r
        
        # 4. Crossover (Binomial)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        trial = np.where(mask, mutant, pop)
        
        # 5. Bound Handling (Midpoint Target)
        # If out of bounds, set to (bound + parent) / 2
        # This is more robust than clipping or random reinitialization
        low_mask = trial < lb
        trial[low_mask] = (lb[np.where(low_mask)[1]] + pop[low_mask]) / 2.0
        
        high_mask = trial > ub
        trial[high_mask] = (ub[np.where(high_mask)[1]] + pop[high_mask]) / 2.0
        
        # 6. Selection and Memory Updates
        success_f = []
        success_cr = []
        diffs = []
        
        for i in range(pop_size):
            if time.time() > time_limit:
                return best_val
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Successful update
                # Add parent to archive
                if n_archive < pop_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                elif n_archive > 0:
                    # Replace random archive member
                    ridx = np.random.randint(0, n_archive)
                    archive[ridx] = pop[i].copy()
                
                diff = fitness[i] - f_trial
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                success_f.append(f[i])
                success_cr.append(cr[i])
                diffs.append(diff)
                
                if f_trial < best_val:
                    best_val = f_trial
                    
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
            
            # Update M_F
            mean_f = np.sum(w * (s_f**2)) / (np.sum(w * s_f) + 1e-15)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            # Update M_CR
            mean_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 8. Sort Population for next generation (Maintenance of p-best)
        s_idx = np.argsort(fitness)
        pop = pop[s_idx]
        fitness = fitness[s_idx]

    return best_val
