#The following algorithm implements **L-SHADE (Linear Population Size Reduction Success-History based Adaptive Differential Evolution)**, adapted to work within a strict time limit.
#
#### Key Improvements
#1.  **Linear Population Size Reduction (LPSR)**: The algorithm starts with a larger population to explore the search space and linearly reduces the population size as time progresses. This allows it to transition from global exploration to rapid local convergence, drastically improving efficiency compared to fixed-population algorithms.
#2.  **Adaptive Parameters (SHADE)**: It uses a historical memory to adapt the Mutation Factor ($F$) and Crossover Rate ($CR$) for each individual, removing the need for manual parameter tuning.
#3.  **Weighted Lehmer Mean**: The memory update mechanism uses a weighted Lehmer mean, giving more weight to parameter values that produced larger fitness improvements.
#4.  **Time-Aware Progress**: The LPSR schedule is dynamically calculated based on the elapsed time relative to `max_time`, ensuring the algorithm optimally utilizes the available budget.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Time-Based Linear Population Size Reduction.
    """
    start_time = time.time()
    
    # --- Parameters ---
    # Initial Population Size (N_init)
    # L-SHADE typically uses 18 * dim. We clamp it to a safe range [30, 300]
    # to handle various time budgets and dimensions effectively.
    np_init = int(18 * dim)
    np_init = np.clip(np_init, 30, 300)
    
    # Minimum Population Size (N_min)
    np_min = 4
    
    # SHADE Memory Size (H)
    H = 6
    
    # Archive Expansion Rate
    arc_rate = 2.6
    
    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    pop_size = np_init
    
    # Initialize Population
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, np.inf)
    
    # Initial Evaluation
    best_val = np.inf
    best_sol = np.zeros(dim)
    
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
            
    # Initialize Memory for Adaptive Parameters (F and CR)
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0
    
    # Initialize External Archive
    # Max capacity is based on initial population, but logical size changes.
    max_arc_capacity = int(np_init * arc_rate)
    archive = np.zeros((max_arc_capacity, dim))
    n_archive = 0
    
    # --- Main Loop ---
    while True:
        curr_time = time.time()
        elapsed = curr_time - start_time
        if elapsed >= max_time:
            return best_val
            
        # Calculate Progress (0.0 to 1.0)
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # ---------------------------------------------------------------------
        # 1. Linear Population Size Reduction (LPSR)
        # ---------------------------------------------------------------------
        # Calculate target population size based on time progress
        plan_pop_size = int(round((np_min - np_init) * progress + np_init))
        if plan_pop_size < np_min:
            plan_pop_size = np_min
            
        if pop_size > plan_pop_size:
            # We need to reduce the population.
            # Sort individuals by fitness (best first)
            sort_idx = np.argsort(fitness)
            
            # Keep only the top 'plan_pop_size' individuals
            keep_idx = sort_idx[:plan_pop_size]
            population = population[keep_idx]
            fitness = fitness[keep_idx]
            
            # Dynamically resize Archive capacity: |A| = round(N * arc_rate)
            new_arc_cap = int(plan_pop_size * arc_rate)
            if n_archive > new_arc_cap:
                # Randomly remove elements to fit new capacity
                keep_indices = np.random.choice(n_archive, new_arc_cap, replace=False)
                # Compact the archive
                archive[:new_arc_cap] = archive[keep_indices]
                n_archive = new_arc_cap
                
            pop_size = plan_pop_size

        # ---------------------------------------------------------------------
        # 2. Parameter Generation
        # ---------------------------------------------------------------------
        # Pick random memory slot for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_f = mem_M_F[r_idx]
        m_cr = mem_M_CR[r_idx]
        
        # Generate F using Cauchy Distribution: Location=m_f, Scale=0.1
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F: if <= 0 retry, if > 1 clip to 1
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            count = np.sum(bad_mask)
            # Regenerate bad values using their specific m_f
            f[bad_mask] = m_f[bad_mask] + 0.1 * np.random.standard_cauchy(count)
        f = np.clip(f, 0, 1)
        
        # Generate CR using Normal Distribution: Mean=m_cr, Std=0.1
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # ---------------------------------------------------------------------
        # 3. Mutation: current-to-pbest/1
        # ---------------------------------------------------------------------
        # Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        
        # p-best rate: p = 0.11 (top 11%)
        p_num = max(2, int(pop_size * 0.11))
        top_p_indices = sorted_indices[:p_num]
        
        # Select pbest
        pbest_indices = np.random.choice(top_p_indices, pop_size)
        x_pbest = population[pbest_indices]
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        collisions = (r1_indices == np.arange(pop_size))
        while np.any(collisions):
            r1_indices[collisions] = np.random.randint(0, pop_size, np.sum(collisions))
            collisions = (r1_indices == np.arange(pop_size))
        x_r1 = population[r1_indices]
        
        # Select r2 (distinct from i and r1, from Union(Population, Archive))
        n_union = pop_size + n_archive
        r2_indices = np.random.randint(0, n_union, pop_size)
        
        collisions = (r2_indices == np.arange(pop_size)) | (r2_indices == r1_indices)
        while np.any(collisions):
            r2_indices[collisions] = np.random.randint(0, n_union, np.sum(collisions))
            collisions = (r2_indices == np.arange(pop_size)) | (r2_indices == r1_indices)
            
        # Construct x_r2
        x_r2 = np.zeros((pop_size, dim))
        mask_pop = r2_indices < pop_size
        mask_arch = ~mask_pop
        
        x_r2[mask_pop] = population[r2_indices[mask_pop]]
        if np.any(mask_arch):
            # Archive indices are offset by pop_size
            arch_idx = r2_indices[mask_arch] - pop_size
            x_r2[mask_arch] = archive[arch_idx]
            
        # Calculate Mutant Vector
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        mutant = population + f[:, None] * (x_pbest - population) + f[:, None] * (x_r1 - x_r2)
        
        # ---------------------------------------------------------------------
        # 4. Crossover (Binomial) & Bounds
        # ---------------------------------------------------------------------
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        
        # Ensure at least one dimension is mutated
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        trial = np.clip(trial, min_b, max_b)
        
        # ---------------------------------------------------------------------
        # 5. Selection and Memory Update
        # ---------------------------------------------------------------------
        success_mem_f = []
        success_mem_cr = []
        success_diff_fit = []
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Evaluation Loop
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
            
            t_val = func(trial[i])
            
            if t_val <= fitness[i]:
                # Improvement or equal: trial replaces target
                
                # Add target to archive before replacement
                if n_archive < int(pop_size * arc_rate):
                    archive[n_archive] = population[i].copy()
                    n_archive += 1
                elif n_archive > 0:
                    # Archive full, replace random
                    ridx = np.random.randint(0, n_archive)
                    archive[ridx] = population[i].copy()
                
                # Record success info for memory update (only if strictly better)
                if t_val < fitness[i]:
                    success_mem_f.append(f[i])
                    success_mem_cr.append(cr[i])
                    success_diff_fit.append(fitness[i] - t_val)
                
                new_population[i] = trial[i]
                new_fitness[i] = t_val
                
                if t_val < best_val:
                    best_val = t_val
                    best_sol = trial[i].copy()
        
        population = new_population
        fitness = new_fitness
        
        # Update Memory (Weighted Lehmer Mean)
        if len(success_mem_f) > 0:
            s_f = np.array(success_mem_f)
            s_cr = np.array(success_mem_cr)
            s_df = np.array(success_diff_fit)
            
            total_diff = np.sum(s_df)
            if total_diff > 0:
                w = s_df / total_diff
                
                # Weighted Lehmer Mean for F
                mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                
                # Weighted Mean for CR
                mean_cr = np.sum(w * s_cr)
                
                # Update memory slot
                mem_M_F[k_mem] = 0.5 * mem_M_F[k_mem] + 0.5 * mean_f
                mem_M_CR[k_mem] = 0.5 * mem_M_CR[k_mem] + 0.5 * mean_cr
                
                k_mem = (k_mem + 1) % H

    return best_val
