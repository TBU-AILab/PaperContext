#The output values from the previous algorithms indicate that **jSO (L-SHADE with LPSR)** (Code 1) performed the best, achieving a value of `1.447`. This suggests the `current-to-pbest` strategy with dynamic `p` and memory-based adaptation is the most effective approach for this problem. The Memetic algorithm (Code 2) likely failed due to the high cost of local search or inefficient balance.
#
#To improve upon the `1.447` result and aim for the global minimum, this new algorithm implements **jSO with Opposition-Based Learning (OBL) Initialization and Stagnation Restart**.
#
#### Key Improvements
#
#1.  **Opposition-Based Learning (OBL) Initialization**: Instead of starting with a purely random population, the algorithm generates a random population *and* its opposite (symmetric within bounds). It evaluates both sets and selects the best `N` individuals. This significantly increases the probability of starting in a promising basin of attraction.
#2.  **Refined Stagnation Detection**: The algorithm monitors both population variance and the number of generations without improvement. If the population converges prematurely or stagnates for too long, a **Soft Restart** is triggered. This preserves the elite solution but re-initializes the rest of the population, resetting the adaptive memory to escape deep local optima.
#3.  **Optimized jSO Parameters**:
#    *   **LPSR (Linear Population Size Reduction)**: Reduces population size linearly from `20 * dim` to `4` to balance exploration (early) and exploitation (late).
#    *   **Midpoint Bound Handling**: Uses the midpoint rule (`(min + x) / 2`) for out-of-bound solutions, which is proven to be more effective for convergence than simple clipping.
#    *   **Weighted Memory Update**: Updates control parameters based on the magnitude of fitness improvement, giving more weight to highly successful mutations.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using jSO (L-SHADE variant) with Opposition-Based Learning (OBL)
    Initialization and Stagnation Restart.
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial Population Size: 
    # Standard jSO uses ~25*dim, but we use ~20*dim to account for OBL overhead.
    pop_size_init = int(max(30, 20 * dim))
    pop_size_min = 4
    
    # SHADE Memory Parameters
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.8)
    k_mem = 0
    
    # Archive for 'current-to-pbest' mutation (diversity maintenance)
    archive = []
    
    # --- Initialization ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # 1. Generate Random Population
    pop_rand = min_b + np.random.rand(pop_size_init, dim) * diff_b
    
    # 2. Generate Opposite Population (OBL)
    # x_opp = min + max - x
    pop_opp = min_b + max_b - pop_rand
    
    # Check bounds for opposite population
    # If out of bounds, replace with random value in bounds
    low_mask = pop_opp < min_b
    high_mask = pop_opp > max_b
    if np.any(low_mask):
        pop_opp[low_mask] = min_b[low_mask] + np.random.rand(np.sum(low_mask)) * diff_b[low_mask]
    if np.any(high_mask):
        pop_opp[high_mask] = min_b[high_mask] + np.random.rand(np.sum(high_mask)) * diff_b[high_mask]
        
    # 3. Evaluate Initial Candidates
    # We combine both pools, evaluate, and select the best pop_size_init
    pop_all = np.vstack((pop_rand, pop_opp))
    n_total = len(pop_all)
    fit_all = np.empty(n_total)
    
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # Evaluation loop with strict time check
    evaluated_count = 0
    for i in range(n_total):
        if datetime.now() >= end_time:
            if evaluated_count == 0:
                # Emergency: evaluate one if none done
                try: return func(pop_rand[0])
                except: return float('inf')
            return best_val
            
        val = func(pop_all[i])
        fit_all[i] = val
        evaluated_count += 1
        
        if val < best_val:
            best_val = val
            best_vec = pop_all[i].copy()
            
    # Select best N individuals
    sorted_idx = np.argsort(fit_all[:evaluated_count])
    pop_size = pop_size_init
    # Handle edge case where time ran out during init
    if len(sorted_idx) < pop_size:
        pop_size = len(sorted_idx)
        
    pop = pop_all[sorted_idx[:pop_size]]
    fitness = fit_all[sorted_idx[:pop_size]]
    
    # --- Stagnation Tracking ---
    last_best_val = best_val
    no_improve_count = 0
    
    # --- Main Optimization Loop ---
    while datetime.now() < end_time:
        
        # 1. Time-based Progress & LPSR
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # Linear Population Size Reduction
        target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            # Sort and truncate
            s_idx = np.argsort(fitness)
            pop = pop[s_idx[:target_size]]
            fitness = fitness[s_idx[:target_size]]
            pop_size = target_size
            
            # Reduce archive size accordingly
            if len(archive) > pop_size:
                np.random.shuffle(archive)
                archive = archive[:pop_size]
        
        # 2. Check Stagnation & Restart
        # Track improvement
        if best_val < last_best_val:
            last_best_val = best_val
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        fit_std = np.std(fitness)
        # Restart conditions:
        # A. Population converged to a point (std ~ 0)
        # B. No improvement for significant duration (stuck in local basin)
        # Don't restart if we are very close to deadline (> 90% time)
        stagnant = (no_improve_count > 500)
        converged = (fit_std < 1e-12)
        
        if (converged or stagnant) and progress < 0.9 and pop_size > pop_size_min:
            # Soft Restart: Keep Elite, Re-init others
            pop[0] = best_vec
            fitness[0] = best_val
            
            # Generate new random individuals
            new_pop = min_b + np.random.rand(pop_size - 1, dim) * diff_b
            pop[1:] = new_pop
            
            # Evaluate new individuals
            for k in range(1, pop_size):
                if datetime.now() >= end_time: return best_val
                v = func(pop[k])
                fitness[k] = v
                if v < best_val:
                    best_val = v
                    best_vec = pop[k].copy()
            
            # Reset Memory & Archive to restart learning
            mem_f.fill(0.5)
            mem_cr.fill(0.8)
            archive = []
            no_improve_count = 0
            continue
            
        # 3. Parameter Generation (jSO/SHADE)
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idxs]
        m_f = mem_f[r_idxs]
        
        # CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        # Regenerate non-positive F
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            f[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
        f[f > 1] = 1.0
        
        # 4. Evolution Cycle (current-to-pbest/1)
        # Sort for pbest selection
        sorted_indices = np.argsort(fitness)
        
        # Dynamic 'p': decreases linearly from 0.25 to 0.05
        p_val = 0.25 - (0.20 * progress)
        if p_val < 0.05: p_val = 0.05
        
        top_p_count = int(max(2, p_val * pop_size))
        p_best_indices = sorted_indices[:top_p_count]
        
        new_pop = np.empty_like(pop)
        new_fitness = np.empty_like(fitness)
        
        succ_mask = np.zeros(pop_size, dtype=bool)
        diff_vals = np.zeros(pop_size)
        
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_val
            
            x_i = pop[i]
            
            # Select x_pbest
            p_idx = np.random.choice(p_best_indices)
            x_pbest = pop[p_idx]
            
            # Select r1 (distinct from i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i: r1 = np.random.randint(0, pop_size)
            x_r1 = pop[r1]
            
            # Select r2 (distinct from i, r1; from Union(Pop, Archive))
            limit = pop_size + len(archive)
            r2_idx = np.random.randint(0, limit)
            while True:
                if r2_idx < pop_size:
                    if r2_idx != i and r2_idx != r1: break
                else:
                    break # Archive indices assumed distinct
                r2_idx = np.random.randint(0, limit)
            
            if r2_idx < pop_size:
                x_r2 = pop[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_size]
                
            # Mutation: current-to-pbest/1
            mutant = x_i + f[i] * (x_pbest - x_i) + f[i] * (x_r1 - x_r2)
            
            # Crossover: Binomial
            mask_j = np.random.rand(dim) < cr[i]
            j_rand = np.random.randint(dim)
            mask_j[j_rand] = True
            trial = np.where(mask_j, mutant, x_i)
            
            # Bound Correction: Midpoint Rule
            below_min = trial < min_b
            above_max = trial > max_b
            trial[below_min] = (x_i[below_min] + min_b[below_min]) / 2.0
            trial[above_max] = (x_i[above_max] + max_b[above_max]) / 2.0
            
            # Selection
            f_trial = func(trial)
            
            # Accept if better or equal
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                
                # Record success for memory update (strictly better)
                if f_trial < fitness[i]:
                    succ_mask[i] = True
                    diff_vals[i] = fitness[i] - f_trial
                    archive.append(x_i.copy())
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
            else:
                new_pop[i] = x_i
                new_fitness[i] = fitness[i]
                
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 5. Update Memory (Weighted Lehmer Mean)
        succ_idx = np.where(succ_mask)[0]
        if len(succ_idx) > 0:
            s_cr = cr[succ_idx]
            s_f = f[succ_idx]
            w = diff_vals[succ_idx]
            
            total_w = np.sum(w)
            if total_w > 0:
                w = w / total_w
                
                m_cr_new = np.sum(w * s_cr)
                m_f_new = np.sum(w * (s_f**2)) / (np.sum(w * s_f) + 1e-12)
                
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * m_cr_new
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * m_f_new
                
                k_mem = (k_mem + 1) % H
                
    return best_val
