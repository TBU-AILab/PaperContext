#The output values and codes for the last generated algorithms indicate that while **JADE** (Output ~12.61) performed best, it likely got stuck in a local optimum significantly above the theoretical global minimum (usually 0). The failure of **SHADE-R** (Output ~29.46) and **IPO-CMA-ES** (Output ~139.98) suggests that while advanced strategies are theoretically sound, their overhead or specific parameter tuning in a constrained Python implementation can be detrimental compared to a simpler, high-frequency loop.
#
#However, to bridge the gap from 12.61 to 0, we need a more aggressive global exploration strategy than standard random restarts.
#
#The following algorithm implements **OBL-SHADE (Opposition-Based Learning SHADE with Restarts)**.
#It improves upon the JADE implementation by:
#1.  **Opposition-Based Initialization & Restarts:** Instead of random restarts, it generates an "opposite" population ($X_{opp} = \min + \max - X$) and selects the fittest individuals from the union of Random and Opposite populations. This drastically improves the probability of starting in a promising basin of attraction.
#2.  **SHADE Parameter Adaptation:** It uses a history memory ($H$) to adapt $F$ and $CR$, which is generally more robust than JADE's single mean adaptation for complex landscapes.
#3.  **Simplified Memory & Archive:** Keeps the diversity maintenance of the archive but streamlines the implementation to maximize generations per second.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using OBL-SHADE (Success-History Adaptive DE with 
    Opposition-Based Learning Restarts).
    
    Key Strategies:
    1. OBL (Opposition-Based Learning) for Initialization and Restarts.
    2. SHADE parameter adaptation (History based).
    3. 'current-to-pbest/1' mutation with external Archive.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Population size: Balanced for speed vs diversity.
    # ~100-150 is usually robust for dims 10-50.
    pop_size = int(np.clip(20 * dim, 60, 150))
    
    # SHADE Memory Parameters
    H = 6
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Archive
    archive = []
    
    # Global Best
    best_fit = float('inf')
    best_sol = None
    
    # --- Helper: OBL Population Generation ---
    def get_obl_population(p_size, current_best=None):
        # 1. Random Generation
        # pop_rand = min + rand * (max - min)
        pop_rand = min_b + np.random.rand(p_size, dim) * diff_b
        
        # Elitism: Inject current best into random population if restarting
        if current_best is not None:
            pop_rand[0] = current_best.copy()
        
        # 2. Opposition Generation
        # pop_opp = min + max - pop_rand
        pop_opp = min_b + max_b - pop_rand
        
        # Bounds Check for Opposite Population
        # If opposite is out of bounds, replace with random value
        mask_out = (pop_opp < min_b) | (pop_opp > max_b)
        if np.any(mask_out):
            # Generate random fix for strictly the out-of-bound elements
            # (Vectorized replacement)
            rand_fix = min_b + np.random.rand(p_size, dim) * diff_b
            pop_opp = np.where(mask_out, rand_fix, pop_opp)
            
        return pop_rand, pop_opp

    # --- Initialization Phase (OBL) ---
    p_rand, p_opp = get_obl_population(pop_size)
    
    # Evaluate Random + Opposite (2 * pop_size)
    combined_pop = np.vstack((p_rand, p_opp))
    combined_fit = np.zeros(len(combined_pop))
    
    for i in range(len(combined_pop)):
        if time.time() - start_time >= max_time: return best_fit
        val = func(combined_pop[i])
        combined_fit[i] = val
        if val < best_fit:
            best_fit = val
            best_sol = combined_pop[i].copy()
            
    # Select best N individuals to form initial population
    sorted_idx = np.argsort(combined_fit)
    pop = combined_pop[sorted_idx[:pop_size]]
    fitness = combined_fit[sorted_idx[:pop_size]]
    
    # --- Main Loop ---
    while True:
        # Check time overhead
        if time.time() - start_time >= max_time: return best_fit
        
        # 1. Parameter Generation (Vectorized)
        # Randomly select memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idx]
        m_f = mem_f[r_idx]
        
        # CR ~ Normal(M_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_f, 0.1)
        # Retry if F <= 0, Clip if F > 1
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Correct F <= 0 (Vectorized retry)
        mask_bad_f = f <= 0
        while np.any(mask_bad_f):
            count = np.sum(mask_bad_f)
            # Re-sample specific indices
            f[mask_bad_f] = m_f[mask_bad_f] + 0.1 * np.random.standard_cauchy(count)
            mask_bad_f = f <= 0
            
        f = np.minimum(f, 1.0)
        
        # 2. Mutation: current-to-pbest/1
        # V = X + F*(Xpbest - X) + F*(Xr1 - Xr2)
        
        # p-best selection: Random individual from top p% (p in [2/N, 0.2])
        p_val = np.random.uniform(2/pop_size, 0.2, pop_size)
        top_indices = (p_val * pop_size).astype(int)
        top_indices = np.maximum(top_indices, 1) # Ensure at least top 1
        
        # Generate indices for pbest, r1, r2
        # pbest indices
        idx_pbest = np.array([np.random.randint(0, lim) for lim in top_indices])
        x_pbest = pop[idx_pbest]
        
        # r1 indices (from Pop, != i)
        idx_r1 = np.random.randint(0, pop_size, pop_size)
        # Simple rotation to avoid self-collision
        mask_self = (idx_r1 == np.arange(pop_size))
        idx_r1[mask_self] = (idx_r1[mask_self] + 1) % pop_size
        x_r1 = pop[idx_r1]
        
        # r2 indices (from Union(Pop, Archive), != i, != r1)
        if len(archive) > 0:
            arr_arch = np.array(archive)
            union_pop = np.vstack((pop, arr_arch))
        else:
            union_pop = pop
            
        n_union = len(union_pop)
        idx_r2 = np.random.randint(0, n_union, pop_size)
        
        # Collision handling for r2 (Check against r1 and self)
        # Note: idx_r2 refers to union. If < pop_size, it's in pop.
        mask_r2_self = (idx_r2 == np.arange(pop_size))
        mask_r2_r1 = (idx_r2 < pop_size) & (idx_r2 == idx_r1)
        mask_bad = mask_r2_self | mask_r2_r1
        
        # One retry pass for collisions is usually sufficient for DE robustness
        if np.any(mask_bad):
            idx_r2[mask_bad] = np.random.randint(0, n_union, np.sum(mask_bad))
            
        x_r2 = union_pop[idx_r2]
        
        # Compute Mutant Vectors
        # pop is already sorted by fitness from end of last loop
        mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        # mask = rand < CR
        cross_mask = np.random.rand(pop_size, dim) < cr[:, None]
        # Ensure at least one dimension taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # Bound Constraints (Clipping)
        trial = np.clip(trial, min_b, max_b)
        
        # 4. Selection & Evaluation
        new_pop = pop.copy()
        new_fit = fitness.copy()
        
        success_diffs = []
        success_cr = []
        success_f = []
        
        for i in range(pop_size):
            if time.time() - start_time >= max_time: return best_fit
            
            f_trial = func(trial[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                new_pop[i] = trial[i]
                new_fit[i] = f_trial
                
                if f_trial < fitness[i]:
                    # Add replaced parent to archive
                    if len(archive) < pop_size:
                        archive.append(pop[i].copy())
                    else:
                        # Random replacement
                        archive[np.random.randint(0, pop_size)] = pop[i].copy()
                        
                    # Store success info
                    success_diffs.append(fitness[i] - f_trial)
                    success_cr.append(cr[i])
                    success_f.append(f[i])
                    
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_sol = trial[i].copy()
                    
        pop = new_pop
        fitness = new_fit
        
        # 5. Memory Update (SHADE)
        if len(success_diffs) > 0:
            diffs = np.array(success_diffs)
            scr = np.array(success_cr)
            sf = np.array(success_f)
            
            # Weighted improvement
            weights = diffs / np.sum(diffs)
            
            # Update Memory CR (Weighted Mean)
            mem_cr[k_mem] = np.sum(weights * scr)
            
            # Update Memory F (Weighted Lehmer Mean)
            mem_f[k_mem] = np.sum(weights * sf**2) / (np.sum(weights * sf) + 1e-15)
            
            k_mem = (k_mem + 1) % H
            
        # 6. Sort Population (Crucial for p-best)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # 7. Restart Check (Stagnation)
        # If diversity is lost (std dev low) or range is tiny
        if np.std(fitness) < 1e-8 or (fitness[-1] - fitness[0]) < 1e-8:
            # Trigger OBL Restart
            
            # Reset adaptation memories to default
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            archive = [] # Clear archive to escape local basin
            
            # Generate new OBL population
            # We inject best_sol to ensure we don't lose the global best
            p_rand, p_opp = get_obl_population(pop_size, current_best=best_sol)
            
            combined_pop = np.vstack((p_rand, p_opp))
            combined_fit = np.zeros(len(combined_pop))
            
            # Evaluate new batch
            for i in range(len(combined_pop)):
                if time.time() - start_time >= max_time: return best_fit
                
                # Optimization: if we injected best_sol at index 0, we know its fitness
                if i == 0 and best_sol is not None:
                     combined_fit[i] = best_fit
                else:
                    val = func(combined_pop[i])
                    combined_fit[i] = val
                    if val < best_fit:
                        best_fit = val
                        best_sol = combined_pop[i].copy()
                        
            # Selection for restart
            sorted_idx = np.argsort(combined_fit)
            pop = combined_pop[sorted_idx[:pop_size]]
            fitness = combined_fit[sorted_idx[:pop_size]]

    return best_fit
