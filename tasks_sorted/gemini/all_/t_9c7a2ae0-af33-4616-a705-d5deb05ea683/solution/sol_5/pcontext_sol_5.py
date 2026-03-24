#The output values from the previous algorithms indicate that **L-SHADE with Linear Population Size Reduction (Code 2)** performed best, achieving a value of `1.459`. The other algorithms (Memetic L-SHADE with coordinate descent, JADE, jDE, Standard DE) performed significantly worse, likely due to inefficient local search strategies or lack of adaptive pressure.
#
#To improve upon the `1.459` result and push towards the global minimum (0.0), this new algorithm implements a pure, optimized version of **jSO (iL-SHADE with specific parameter adaptation)**. jSO is an enhanced version of L-SHADE that introduces a specific deterministic reduction in the `p` parameter (controlling greediness) and refined bound handling rules, designed specifically to solve the convergence issues L-SHADE faces in the late stages of optimization.
#
#### Key Improvements
#1.  **jSO Parameter Adaptation**:
#    *   **Dynamic `p` value**: The `current-to-pbest` strategy's greedy parameter `p` linearly decreases from `0.25` (exploration) to `0.05` (exploitation) based on time. This forces the algorithm to focus on the best solutions as the deadline approaches.
#    *   **Initial Weights**: Memory for mutation factor $F$ and crossover $CR$ are initialized to $0.5$ and $0.8$ respectively (as per jSO guidelines), encouraging higher crossover rates initially.
#2.  **Midpoint Bound Correction**: Instead of simply clipping values to bounds (which piles solutions on the edge), this algorithm uses the midpoint rule (`(min + x) / 2`). This preserves the direction of the search vector while keeping it valid.
#3.  **Refined Restart Mechanism**: The algorithm monitors population variance. If all individuals converge to a local basin (variance $\approx 0$) before time runs out, it triggers a **Soft Restart**—keeping the single best solution but re-initializing the rest of the population and, crucially, resetting the adaptation memory to restore exploration capability.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using jSO (L-SHADE variant with specific parameter adaptation)
    and Linear Population Size Reduction (LPSR).
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial Population Size: jSO suggests ~25 * dim
    pop_size_init = int(max(30, 25 * dim))
    pop_size_min = 4
    
    # SHADE Memory Parameters
    H = 5
    # Initialize memory: M_F=0.5, M_CR=0.8 (jSO defaults)
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.8)
    k_mem = 0
    
    # Archive for 'current-to-pbest' mutation
    archive = []
    
    # --- Initialization ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    pop_size = pop_size_init
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        if datetime.now() >= end_time: return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while datetime.now() < end_time:
        
        # 1. Time-based Progress Calculation
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # 2. Linear Population Size Reduction (LPSR)
        target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        target_size = max(pop_size_min, target_size)
        
        if pop_size > target_size:
            # Sort by fitness and truncate the worst
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:target_size]]
            fitness = fitness[sorted_idx[:target_size]]
            pop_size = target_size
            
            # Resize archive to match current population size
            if len(archive) > pop_size:
                np.random.shuffle(archive)
                archive = archive[:pop_size]

        # 3. Stagnation Check & Soft Restart
        # If population variance is extremely low, we are stuck in a basin.
        # Restart allows finding a better basin if time permits.
        fit_range = np.max(fitness) - np.min(fitness)
        if fit_range < 1e-9 and pop_size > pop_size_min:
            # Keep global best (Elitism)
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_vec
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Reset Memory to encourage fresh exploration
            mem_f.fill(0.5)
            mem_cr.fill(0.8)
            archive = []
            
            # Re-evaluate
            for i in range(1, pop_size):
                if datetime.now() >= end_time: return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            continue

        # 4. Parameter Generation (jSO/SHADE)
        # Select random memory indices
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idxs]
        m_f = mem_f[r_idxs]
        
        # Generate CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(M_F, 0.1)
        # Use simple regeneration for F <= 0
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            f[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
        f[f > 1] = 1.0
        
        # 5. Evolution Cycle
        # Calculate dynamic p value for current-to-pbest
        # Linearly decreases from 0.25 to 0.05 based on progress
        p_val = 0.25 - (0.20 * progress)
        if p_val < 0.05: p_val = 0.05
        
        sorted_indices = np.argsort(fitness)
        top_p_count = int(max(2, p_val * pop_size))
        p_best_indices = sorted_indices[:top_p_count]
        
        new_pop = np.empty_like(pop)
        new_fitness = np.empty_like(fitness)
        
        # Tracking successes for memory update
        succ_mask = np.zeros(pop_size, dtype=bool)
        diff_f_vals = np.zeros(pop_size)
        
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_val
            
            x_i = pop[i]
            
            # Select x_pbest
            x_pbest = pop[np.random.choice(p_best_indices)]
            
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
                    break # Archive indices are effectively distinct
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
            # If out of bounds, set to midpoint between old value and bound
            # This is superior to clipping for convergence speed.
            below_min = trial < min_b
            above_max = trial > max_b
            trial[below_min] = (x_i[below_min] + min_b[below_min]) / 2.0
            trial[above_max] = (x_i[above_max] + max_b[above_max]) / 2.0
            
            # Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                
                # If strictly better, record success for parameter adaptation
                if f_trial < fitness[i]:
                    succ_mask[i] = True
                    diff_f_vals[i] = fitness[i] - f_trial
                    archive.append(x_i.copy())
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
            else:
                new_pop[i] = x_i
                new_fitness[i] = fitness[i]
        
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Limit
        if len(archive) > pop_size:
            # Random removal
            num_remove = len(archive) - pop_size
            for _ in range(num_remove):
                archive.pop(np.random.randint(0, len(archive)))
        
        # 6. Update Memory (Weighted Lehmer Mean)
        succ_idx = np.where(succ_mask)[0]
        if len(succ_idx) > 0:
            s_cr = cr[succ_idx]
            s_f = f[succ_idx]
            w = diff_f_vals[succ_idx]
            
            # Normalize weights
            total_imp = np.sum(w)
            if total_imp > 0:
                w = w / total_imp
                
                # Update M_CR: Weighted Arithmetic Mean
                mean_cr = np.sum(w * s_cr)
                
                # Update M_F: Weighted Lehmer Mean (sum(w*f^2)/sum(w*f))
                mean_f = np.sum(w * (s_f ** 2)) / (np.sum(w * s_f) + 1e-12)
                
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                
                k_mem = (k_mem + 1) % H

    return best_val
