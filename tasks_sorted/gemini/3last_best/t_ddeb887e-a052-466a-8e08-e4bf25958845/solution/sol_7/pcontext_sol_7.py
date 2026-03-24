#The output values and codes for the last generated algorithms indicate that while **JADE** (Output ~12.61) and **OBL-SHADE** (Output ~12.61) performed well, they likely got stuck in a local optimum near 12.6, unable to reach the theoretical minimum (0). The failure of other methods to improve on this suggests the need for a hybrid approach that combines the robust global search of **SHADE** with a mechanism for fine-grained **Local Search** (Memetic Algorithm) to exploit the best basin of attraction found.
#
#The following algorithm implements **Memetic SHADE with Restarts**.
#It improves upon the previous best (JADE) by:
#1.  **SHADE Adaptation:** Uses Success-History adaptation for $F$ and $CR$ parameters, which is superior to JADE's single mean for complex landscapes.
#2.  **Pattern Search (Local Search):** Periodically performs a lightweight coordinate descent on the global best solution. This specifically addresses the issue of getting "stuck" close to the optimum by greedily refining the best candidate along each dimension.
#3.  **Elitist Restarts:** When the population stagnates (low variance) or fails to improve for many generations, it triggers a restart but keeps the refined global best, effectively performing iterated local search.
#4.  **Vectorized Implementation:** Ensures maximum generation count within the time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using a Memetic SHADE algorithm (SHADE + Pattern Search) with Restarts.
    
    Key Features:
    1. SHADE (Success-History Adaptive Differential Evolution): Robust global search.
    2. Pattern Search: Local search on the best solution to refine accuracy.
    3. Restart Mechanism: Escapes local optima when population converges.
    """
    start_time = time.time()
    
    # --- Configuration ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Population Size:
    # A size of ~20*dim allows sufficient diversity for SHADE dynamics.
    # We clip it to [60, 200] to balance exploration vs speed.
    pop_size = int(np.clip(20 * dim, 60, 200))
    
    # SHADE Parameters
    H = 6  # History memory size
    mem_M_cr = np.full(H, 0.5) # Memory for CR
    mem_M_f = np.full(H, 0.5)  # Memory for F
    k_mem = 0
    archive = [] # External archive for diversity
    
    # Global Best Tracking
    best_fit = float('inf')
    best_sol = None
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.zeros(pop_size)
    
    # Initial Evaluation
    for i in range(pop_size):
        if time.time() - start_time >= max_time: return best_fit
        val = func(pop[i])
        fitness[i] = val
        if val < best_fit:
            best_fit = val
            best_sol = pop[i].copy()
            
    # Sort Population (Best individual at index 0)
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Main Loop ---
    gen = 0
    stagnation_counter = 0
    
    while True:
        # Check Time Budget
        if time.time() - start_time >= max_time:
            return best_fit
        
        gen += 1
        
        # 1. Parameter Generation (Vectorized)
        # Randomly select memory indices
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_M_cr[r_idx]
        m_f = mem_M_f[r_idx]
        
        # Generate CR ~ Normal(M_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(M_f, 0.1)
        # Using tan(pi*(rand-0.5)) approx.
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Handle F bounds: retry if <= 0, clip if > 1
        mask_bad = f <= 0
        while np.any(mask_bad):
            f[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_bad)) - 0.5))
            mask_bad = f <= 0
        f = np.minimum(f, 1.0)
        
        # 2. Mutation: current-to-pbest/1
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # p-best selection: random from top p% (p in [2/N, 0.2])
        p_vals = np.random.uniform(2/pop_size, 0.2, pop_size)
        p_indices = (p_vals * pop_size).astype(int)
        p_indices = np.maximum(p_indices, 1) # Ensure at least top 1
        
        # Vectorized selection of pbest indices from sorted population
        # (Since pop is sorted, index 0 is best. Random int < p_index gives a top individual)
        rand_ranks = np.floor(np.random.rand(pop_size) * p_indices).astype(int)
        x_pbest = pop[rand_ranks]
        
        # Select r1 (from Pop, distinct from i)
        r1 = np.random.randint(0, pop_size, pop_size)
        mask_self = (r1 == np.arange(pop_size))
        r1[mask_self] = (r1[mask_self] + 1) % pop_size
        x_r1 = pop[r1]
        
        # Select r2 (from Union(Pop, Archive), distinct from i, r1)
        if len(archive) > 0:
            # Convert archive to array (fast for small size)
            arr_arch = np.array(archive)
            union_pop = np.vstack((pop, arr_arch))
        else:
            union_pop = pop
            
        n_union = len(union_pop)
        r2 = np.random.randint(0, n_union, pop_size)
        
        # Collision handling for r2 (Check vs i and r1)
        mask_coll = (r2 == np.arange(pop_size)) | (r2 == r1)
        if np.any(mask_coll):
            r2[mask_coll] = np.random.randint(0, n_union, np.sum(mask_coll))
        x_r2 = union_pop[r2]
        
        # Compute Mutant Vectors
        mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        mask_cross = np.random.rand(pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size) # Force at least one dim
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, pop)
        
        # Bound Constraints (Clipping)
        trial = np.clip(trial, min_b, max_b)
        
        # 4. Selection & Evaluation
        improved = False
        success_diffs = []
        success_cr = []
        success_f = []
        
        for i in range(pop_size):
            if time.time() - start_time >= max_time: return best_fit
            
            f_trial = func(trial[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                if f_trial < fitness[i]:
                    improved = True
                    # Add replaced parent to archive
                    if len(archive) < pop_size:
                        archive.append(pop[i].copy())
                    else:
                        archive[np.random.randint(0, pop_size)] = pop[i].copy()
                        
                    # Store success data for SHADE adaptation
                    success_diffs.append(fitness[i] - f_trial)
                    success_cr.append(cr[i])
                    success_f.append(f[i])
                    
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_sol = trial[i].copy()
        
        if improved:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # 5. SHADE Memory Update
        if len(success_diffs) > 0:
            w = np.array(success_diffs)
            w = w / np.sum(w) # Normalize weights
            
            s_cr = np.array(success_cr)
            s_f = np.array(success_f)
            
            # Weighted Mean CR
            mean_cr = np.sum(w * s_cr)
            
            # Weighted Lehmer Mean F
            mean_f = np.sum(w * s_f**2) / (np.sum(w * s_f) + 1e-15)
            
            mem_M_cr[k_mem] = mean_cr
            mem_M_f[k_mem] = np.clip(mean_f, 0, 1)
            k_mem = (k_mem + 1) % H
            
        # 6. Sort Population (Crucial for p-best selection)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # 7. Memetic Step: Pattern Search (Local Search) on Best Solution
        # Run periodically (e.g., every 20 gens) or if stagnating
        # This helps refine the solution to high precision
        if gen % 20 == 0 or stagnation_counter > 5:
            # Step size based on current population spread
            pop_std = np.std(pop, axis=0)
            step_size = np.mean(pop_std) * 0.5
            # Clamp step size to prevent it from vanishing too early
            if step_size < 1e-9: step_size = 0.005 * np.mean(diff_b)
            
            # Coordinate Descent
            for d in range(dim):
                if time.time() - start_time >= max_time: return best_fit
                
                # Try positive step
                old_val = best_sol[d]
                new_val = np.clip(old_val + step_size, min_b[d], max_b[d])
                
                if abs(new_val - old_val) > 1e-13:
                    best_sol[d] = new_val
                    val = func(best_sol)
                    if val < best_fit:
                        best_fit = val
                        # Sync with population (index 0 is best)
                        pop[0] = best_sol.copy()
                        fitness[0] = best_fit
                        continue # Success, move to next dimension
                    else:
                        best_sol[d] = old_val # Revert
                
                # Try negative step
                new_val = np.clip(old_val - step_size, min_b[d], max_b[d])
                if abs(new_val - old_val) > 1e-13:
                    best_sol[d] = new_val
                    val = func(best_sol)
                    if val < best_fit:
                        best_fit = val
                        pop[0] = best_sol.copy()
                        fitness[0] = best_fit
                    else:
                        best_sol[d] = old_val # Revert

        # 8. Restart Mechanism
        # If population diversity is extremely low or stagnation is prolonged
        fit_range = fitness[-1] - fitness[0]
        if fit_range < 1e-9 or np.std(fitness) < 1e-9 or stagnation_counter > 25:
            # Re-initialize population for a fresh search
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol.copy() # Elitism: Keep the best found so far
            
            # Evaluate new population
            fitness[0] = best_fit
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time: return best_fit
                val = func(pop[i])
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_sol = pop[i].copy()
                    
            # Reset SHADE adaptation memories to allow new parameter exploration
            mem_M_cr.fill(0.5)
            mem_M_f.fill(0.5)
            archive = []
            stagnation_counter = 0
            
            # Re-sort
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]

    return best_fit
