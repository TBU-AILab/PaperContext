#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution)**.
#
#### Algorithm Description
#SHADE is a state-of-the-art improvement over standard Differential Evolution (DE) designed to perform effectively without manual parameter tuning. It achieves this through:
#1.  **History-Based Parameter Adaptation**: Instead of fixed $F$ (mutation factor) and $CR$ (crossover rate) values, the algorithm maintains a memory of settings that recently led to successful updates. New parameters are sampled from distributions centered on these successful historical values.
#2.  **Current-to-pbest Mutation with Archive**:
#    *   **Guidance**: The population is guided towards the top $p\%$ of best individuals ("p-best"), balancing greedy convergence with exploration.
#    *   **Diversity (Archive)**: To prevent premature convergence, the difference vector ($x_{r1} - x_{r2}$) draws the second component ($x_{r2}$) from a union of the current population and an **external archive** of inferior solutions. This preserves diversity even as the population clusters.
#3.  **Robustness**: This method typically converges faster and deeper than standard DE, making it highly effective for minimizing functions within a limited time.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive 
    Differential Evolution) with external archive and historical memory.
    """
    start_time = time.time()
    # Use 98% of available time to ensure we return result strictly before timeout
    time_limit = start_time + max_time * 0.98

    # --- Configuration ---
    # Population size: Standard robust sizing (approx 10-20 * dim)
    pop_size = max(30, int(10 * dim))
    
    # SHADE Memory Parameters
    H = 5  # Size of historical memory
    mem_M_CR = np.full(H, 0.5) # Memory for Crossover Rate (init 0.5)
    mem_M_F = np.full(H, 0.5)  # Memory for Mutation Factor (init 0.5)
    k_mem = 0 # Current memory index to update

    # External Archive
    # Stores parent vectors replaced by better offspring to maintain diversity
    archive = np.zeros((pop_size, dim))
    n_archive = 0  # Current number of items in archive

    # Pre-process bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # --- Initialization ---
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')

    # Evaluate Initial Population
    for i in range(pop_size):
        if time.time() > time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # --- Main Optimization Loop ---
    while True:
        if time.time() > time_limit:
            return best_val

        # 1. Parameter Adaptation
        # For each individual, pick a random index from memory
        idx_r = np.random.randint(0, H, pop_size)
        mu_cr = mem_M_CR[idx_r]
        mu_f = mem_M_F[idx_r]

        # Generate CR ~ Normal(mu_cr, 0.1), clipped to [0, 1]
        CR = np.random.normal(mu_cr, 0.1)
        CR = np.clip(CR, 0, 1)

        # Generate F ~ Cauchy(mu_f, 0.1)
        # Note: numpy doesn't have loc/scale for standard_cauchy, so we scale manually
        F = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints: 
        # If F > 1, clip to 1. If F <= 0, regenerate (retry until > 0)
        bad_f_mask = F <= 0
        while np.any(bad_f_mask):
            n_bad = np.sum(bad_f_mask)
            # Resample for invalid values
            F[bad_f_mask] = mu_f[bad_f_mask] + 0.1 * np.random.standard_cauchy(n_bad)
            bad_f_mask = F <= 0
        F = np.clip(F, 0, 1) # Clip > 1 is valid, but <= 0 is not allowed for DE

        # 2. Mutation: DE/current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Sort population by fitness to find p-best
        sorted_indices = np.argsort(fitness)
        
        # p is a random value in [2/NP, 0.2] to balance exploration/exploitation
        p_val = np.random.uniform(2/pop_size, 0.2)
        top_p_count = int(max(2, pop_size * p_val))
        
        # Select p-best individuals (one for each target)
        pbest_pool_idxs = sorted_indices[:top_p_count]
        pbest_chosen_idxs = np.random.choice(pbest_pool_idxs, pop_size)
        x_pbest = pop[pbest_chosen_idxs]

        # Select r1: Random individual from population != current
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        # Handle collision (r1 cannot be same as i)
        collision = (r1_idxs == np.arange(pop_size))
        r1_idxs[collision] = (r1_idxs[collision] + 1) % pop_size
        x_r1 = pop[r1_idxs]

        # Select r2: Random from Union(Population, Archive) != r1, != i
        # Pool size = current pop + current archive items
        pool_size = pop_size + n_archive
        r2_idxs = np.random.randint(0, pool_size, pop_size)
        
        # Construct x_r2 array based on indices
        x_r2 = np.empty((pop_size, dim))
        
        # If index < pop_size, take from pop. Else take from archive.
        from_pop_mask = r2_idxs < pop_size
        from_arch_mask = ~from_pop_mask
        
        if np.any(from_pop_mask):
            x_r2[from_pop_mask] = pop[r2_idxs[from_pop_mask]]
        if np.any(from_arch_mask):
            # Archive indices mapped: idx - pop_size
            arch_real_idxs = r2_idxs[from_arch_mask] - pop_size
            x_r2[from_arch_mask] = archive[arch_real_idxs]

        # Compute Difference Vectors
        diff_pbest = x_pbest - pop
        diff_r1_r2 = x_r1 - x_r2
        
        # Expand F for broadcasting (pop_size, 1)
        F_col = F.reshape(-1, 1)
        
        # Calculate Mutant Vector
        mutant = pop + F_col * diff_pbest + F_col * diff_r1_r2

        # 3. Crossover (Binomial)
        mask = np.random.rand(pop_size, dim) < CR.reshape(-1, 1)
        # Standard DE guarantee: at least one parameter comes from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask, mutant, pop)
        
        # 4. Bound Constraints
        trial = np.clip(trial, lb, ub)

        # 5. Selection and History Update
        scr = [] # Successful CRs
        sf = []  # Successful Fs
        diffs = [] # Fitness improvements
        
        # Evaluate one by one to check time strictly
        for i in range(pop_size):
            if time.time() > time_limit:
                return best_val
            
            f_trial = func(trial[i])
            f_old = fitness[i]
            
            # Greedy Selection
            if f_trial <= f_old:
                # Add parent to archive before replacing
                parent_copy = pop[i].copy()
                if n_archive < pop_size:
                    archive[n_archive] = parent_copy
                    n_archive += 1
                else:
                    # Replace random individual in archive
                    rand_idx = np.random.randint(0, pop_size)
                    archive[rand_idx] = parent_copy
                
                # Update Population
                pop[i] = trial[i]
                fitness[i] = f_trial
                
                # Record success for memory update
                scr.append(CR[i])
                sf.append(F[i])
                diffs.append(f_old - f_trial)
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial

        # 6. Update Historical Memory
        if len(scr) > 0:
            w = np.array(diffs)
            total_diff = np.sum(w)
            
            # If all improvements are 0, use uniform weights
            if total_diff == 0:
                w = np.ones(len(w)) / len(w)
            else:
                w = w / total_diff
            
            s_cr = np.array(scr)
            s_f = np.array(sf)
            
            # Update M_CR (Weighted Mean)
            new_m_cr = np.sum(w * s_cr)
            mem_M_CR[k_mem] = new_m_cr
            
            # Update M_F (Weighted Lehmer Mean)
            # sum(w * f^2) / sum(w * f)
            denom = np.sum(w * s_f)
            if denom == 0:
                new_m_f = 0.5
            else:
                new_m_f = np.sum(w * (s_f ** 2)) / denom
            
            mem_M_F[k_mem] = np.clip(new_m_f, 0, 1)
            
            # Advance memory index
            k_mem = (k_mem + 1) % H

    return best_val
