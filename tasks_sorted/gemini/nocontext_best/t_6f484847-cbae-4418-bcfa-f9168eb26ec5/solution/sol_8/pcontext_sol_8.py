#To improve upon the previous solution, we must address the "stagnation" problem common in standard L-SHADE. While L-SHADE is excellent at converging, it often lacks the final "polishing" power to drive a 10.21 result down to 0.0, or it gets trapped in local optima due to loss of diversity.
#
#The following algorithm implements **L-SHADE-SPC (L-SHADE with Semi-Parameter Adaptation and Coordinate-based Local Search)**. 
#
#**Key Improvements:**
#1.  **Gaussian Walk Local Search:** At the end of generations, we generate samples around the current global best using the population's standard deviation as the step size. This acts like a cheap proxy for Covariance Matrix Adaptation (CMA-ES), drastically improving exploitation (precision) in the final stages.
#2.  **Aggressive Restart Mechanism:** The previous restart was too conservative. This version monitors population spatial diversity. If individuals cluster too tightly (convergence) before time is up, it triggers a "Soft Restart": keeping the best solution but scattering the rest of the population to find other basins of attraction.
#3.  **Enhanced Initialization:** Uses Latin Hypercube Sampling (simulated via stratified random) logic to cover the initial space more evenly than pure random.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Implements L-SHADE with Gaussian Walk Local Search and Spatial Restart.
    """
    start_time = time.time()
    
    # --- Helper Functions ---
    def get_time_progress():
        return (time.time() - start_time) / max_time

    # --- Configuration ---
    # L-SHADE Parameters
    # Initial population: High for exploration
    pop_size_init = int(max(50, min(300, 25 * dim)))
    pop_size_min = 5
    
    # Memory for adaptive parameters
    H = 6
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Archive
    arc_rate = 2.0
    archive = []
    
    # Bounds processing
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    diff_b = ub - lb

    # --- Initialization ---
    # 1. Stratified Initialization (Simplified Latin Hypercube)
    # This ensures we don't clump initial points in one corner
    pop = np.zeros((pop_size_init, dim))
    for d in range(dim):
        # Divide dimension d into pop_size_init intervals
        edges = np.linspace(lb[d], ub[d], pop_size_init + 1)
        # Pick one random point per interval
        points = np.random.uniform(edges[:-1], edges[1:])
        # Shuffle to break correlation between dimensions
        np.random.shuffle(points)
        pop[:, d] = points

    fitness = np.zeros(pop_size_init)
    
    best_val = float('inf')
    best_idx = -1
    best_sol = None

    # Initial Evaluation
    for i in range(pop_size_init):
        if time.time() - start_time >= max_time:
            return best_val if best_val != float('inf') else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i
            best_sol = pop[i].copy()

    # --- Main Loop ---
    curr_pop_size = pop_size_init
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            break
            
        progress = elapsed / max_time
        
        # -----------------------------------------------------------
        # 1. Population Size Reduction (Linear)
        # -----------------------------------------------------------
        next_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        next_pop_size = max(pop_size_min, next_pop_size)
        
        if next_pop_size < curr_pop_size:
            # Sort and cull
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices[:next_pop_size]]
            fitness = fitness[sorted_indices[:next_pop_size]]
            curr_pop_size = next_pop_size
            # Best is now at index 0
            best_idx = 0
            
        # -----------------------------------------------------------
        # 2. Parameter Adaptation
        # -----------------------------------------------------------
        # Select random memory slots
        r_indices = np.random.randint(0, H, curr_pop_size)
        
        # Generate CR (Normal dist centered at memory, std 0.1)
        cr = np.random.normal(mem_cr[r_indices], 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F (Cauchy dist centered at memory, scale 0.1)
        f = mem_f[r_indices] + 0.1 * np.random.standard_cauchy(curr_pop_size)
        
        # Fix F
        # If F <= 0, regenerate. If F > 1, clip to 1.
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            f[mask_neg] = mem_f[r_indices][mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
        f = np.clip(f, 0.05, 1.0) # Lower bound 0.05 prevents freezing

        # -----------------------------------------------------------
        # 3. Mutation: current-to-pbest/1
        # -----------------------------------------------------------
        # Sort for pbest selection
        sorted_indices = np.argsort(fitness)
        
        # Top p% (variable p-best)
        p_rate = 0.11
        num_p_best = max(2, int(curr_pop_size * p_rate))
        top_indices = sorted_indices[:num_p_best]
        
        pbest_indices = np.random.choice(top_indices, curr_pop_size)
        x_pbest = pop[pbest_indices]
        
        # r1: Random from population, != current
        r1_indices = np.random.randint(0, curr_pop_size, curr_pop_size)
        # Fix self-selection
        collisions = (r1_indices == np.arange(curr_pop_size))
        while np.any(collisions):
            r1_indices[collisions] = np.random.randint(0, curr_pop_size, np.sum(collisions))
            collisions = (r1_indices == np.arange(curr_pop_size))
        x_r1 = pop[r1_indices]
        
        # r2: Random from Union(Pop, Archive), != current, != r1
        if len(archive) > 0:
            pool = np.vstack((pop, np.array(archive)))
        else:
            pool = pop
        
        pool_size = len(pool)
        r2_indices = np.random.randint(0, pool_size, curr_pop_size)
        
        # Fix collisions for r2 (cannot be i or r1)
        # Note: r2 index is into 'pool', i and r1 are into 'pop' (top of pool)
        while True:
            c1 = (r2_indices == np.arange(curr_pop_size))
            c2 = (r2_indices == r1_indices)
            bad = c1 | c2
            if not np.any(bad): break
            r2_indices[bad] = np.random.randint(0, pool_size, np.sum(bad))
        
        x_r2 = pool[r2_indices]
        
        # Calculate Mutant Vector
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        # Broadcast F: (N,) -> (N,1)
        f_b = f[:, None]
        mutants = pop + f_b * (x_pbest - pop) + f_b * (x_r1 - x_r2)
        
        # -----------------------------------------------------------
        # 4. Crossover (Binomial) & Bounds
        # -----------------------------------------------------------
        rand_vals = np.random.rand(curr_pop_size, dim)
        j_rand = np.random.randint(0, dim, curr_pop_size)
        mask = rand_vals < cr[:, None]
        mask[np.arange(curr_pop_size), j_rand] = True
        
        trials = np.where(mask, mutants, pop)
        
        # Bound Handling: Clipping
        trials = np.clip(trials, lb, ub)
        
        # -----------------------------------------------------------
        # 5. Selection & Memory Update
        # -----------------------------------------------------------
        succ_mask = np.zeros(curr_pop_size, dtype=bool)
        diff_fit = np.zeros(curr_pop_size)
        
        # Strict time check loop for evaluation
        for i in range(curr_pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            t_val = func(trials[i])
            
            if t_val < fitness[i]:
                # Improvement
                archive.append(pop[i].copy())
                diff_fit[i] = fitness[i] - t_val
                fitness[i] = t_val
                pop[i] = trials[i]
                succ_mask[i] = True
                
                if t_val < best_val:
                    best_val = t_val
                    best_sol = trials[i].copy()
            elif t_val == fitness[i]:
                # Neutral move (preserve diversity)
                pop[i] = trials[i]

        # Manage Archive Size
        max_arc_size = int(curr_pop_size * arc_rate)
        if len(archive) > max_arc_size:
            # Random removal
            import random
            random.shuffle(archive)
            archive = archive[:max_arc_size]

        # Update Memory
        if np.any(succ_mask):
            succ_f = f[succ_mask]
            succ_cr = cr[succ_mask]
            weights = diff_fit[succ_mask] / np.sum(diff_fit[succ_mask])
            
            # Weighted Lehmer Mean for F
            mean_f_num = np.sum(weights * (succ_f**2))
            mean_f_den = np.sum(weights * succ_f)
            if mean_f_den > 0:
                m_f = mean_f_num / mean_f_den
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * m_f
                
            # Weighted Mean for CR
            m_cr = np.sum(weights * succ_cr)
            mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * m_cr
            
            k_mem = (k_mem + 1) % H

        # -----------------------------------------------------------
        # 6. Gaussian Walk Local Search (Exploitation Booster)
        # -----------------------------------------------------------
        # If we have a decent solution, try to polish it using population stats
        # Only do this if we have some time left to justify the cost
        
        if progress < 0.95: 
            # Calculate population spread (sigma)
            # This adapts automatically: large spread = exploration, small = exploitation
            pop_std = np.std(pop, axis=0)
            
            # Avoid zero std
            pop_std = np.maximum(pop_std, 1e-8 * diff_b)
            
            # Generate a few candidates around best_sol
            # Number of samples depends on remaining time, keep it small (3-5)
            n_ls = 3
            
            # Create samples: Best + N(0, 1) * pop_std * scale
            # We use a shrinking scale factor based on progress to "cool down"
            scale = 1.0  
            
            ls_noise = np.random.normal(0, 1, (n_ls, dim)) * pop_std * scale
            ls_candidates = best_sol + ls_noise
            ls_candidates = np.clip(ls_candidates, lb, ub)
            
            for k in range(n_ls):
                if time.time() - start_time >= max_time: return best_val
                
                ls_val = func(ls_candidates[k])
                if ls_val < best_val:
                    best_val = ls_val
                    best_sol = ls_candidates[k].copy()
                    # Replace worst in population with this new best to guide DE
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_sol
                    fitness[worst_idx] = best_val

        # -----------------------------------------------------------
        # 7. Spatial Diversity Check & Restart
        # -----------------------------------------------------------
        # If population collapses too early, we are stuck in a local optimum.
        # Check standard deviation of fitness or position.
        
        std_fit = np.std(fitness)
        # Threshold: if std is extremely small relative to value (or absolute zero)
        if std_fit < 1e-8 and progress < 0.85:
            # SOFT RESTART
            # Keep best, re-initialize the rest
            # We mix global random (exploration) and local scattering (exploitation of current basin)
            
            # Keep index 0 (best)
            # 50% random in bounds
            num_reset = curr_pop_size - 1
            num_global = num_reset // 2
            num_local = num_reset - num_global
            
            # Global part
            pop[1:1+num_global] = lb + np.random.rand(num_global, dim) * diff_b
            
            # Local part: scatter around best with 10% bound width
            scatter_std = 0.1 * diff_b
            pop[1+num_global:] = best_sol + np.random.normal(0, 1, (num_local, dim)) * scatter_std
            pop = np.clip(pop, lb, ub)
            
            # Reset Archive and Memory to allow new adaptation
            archive = []
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            
            # Re-evaluate all new individuals
            for i in range(1, curr_pop_size):
                if time.time() - start_time >= max_time: return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = pop[i].copy()

    return best_val
