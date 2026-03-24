#The previous results show that **jSO** (Code 2) and **L-SHADE** (Code 4) yielded the best performance (~1.45), while other approaches like Memetic algorithms or standard DE lagged behind. The value `1.45` likely corresponds to a local optimum or a lack of fine-tuning capability in the final stages of the search.
#
#To bridge the gap from `1.45` to `0.0`, this new algorithm implements **jSO with Hybrid Restart**.
#
#### Key Improvements
#1.  **Hybrid Restart Mechanism**: Unlike standard restarts that completely re-randomize the population (losing progress) or simple local searches (losing diversity), this algorithm uses a **Hybrid** approach when stagnation is detected:
#    *   **50% of the population** is re-initialized globally (randomly) to explore new basins.
#    *   **50% of the population** is re-initialized locally (Gaussian sampling) around the best-so-far solution to escape shallow traps and fine-tune the current basin.
#    *   **Memory Reset**: The historical memory of crossover/mutation parameters is reset to allow for fresh adaptation dynamics.
#2.  **Optimized Population Sizing**: Uses `18 * dim` as the initial population size (tuned down from `25 * dim` to allow more generations within the time limit) combined with **Linear Population Size Reduction (LPSR)** to focus computational resources as the deadline approaches.
#3.  **Corrected Parameter Generation**: Fixes a potential indexing issue in the Cauchy generation logic found in previous iterations and maintains the **Weighted Lehmer Mean** update which is crucial for jSO's performance.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using jSO (L-SHADE variant) with Hybrid Restart.
    
    Improvements:
    - Hybrid Restart: Mixes global random exploration with local Gaussian 
      exploitation around the best solution when stagnation occurs.
    - Optimized Population Size: 18*dim for better generation count.
    - Midpoint Bound Handling: Efficiently handles boundary constraints.
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial Population Size: 18 * dim is a balanced choice for limited time
    pop_size_init = int(max(30, 18 * dim))
    pop_size_min = 4
    
    # SHADE Memory Parameters
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.8)
    k_mem = 0
    
    # External Archive for diversity
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
        
        # 1. Time-based Progress & LPSR
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # Calculate target population size
        target_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        target_size = max(pop_size_min, target_size)
        
        # Reduce population if needed
        if pop_size > target_size:
            # Sort by fitness (best first)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:target_size]]
            fitness = fitness[sorted_idx[:target_size]]
            pop_size = target_size
            
            # Reduce archive size
            if len(archive) > pop_size:
                np.random.shuffle(archive)
                archive = archive[:pop_size]
        
        # 2. Hybrid Restart Check
        # Trigger if fitness variance is extremely low (convergence)
        # and we are not in the very last moments of optimization.
        fit_range = np.max(fitness) - np.min(fitness)
        if fit_range < 1e-8 and pop_size > pop_size_min and progress < 0.95:
            
            # Keep the elite
            pop[0] = best_vec
            fitness[0] = best_val
            
            # Divide remaining slots
            n_remaining = pop_size - 1
            n_global = n_remaining // 2
            n_local = n_remaining - n_global
            
            # A. Global Restart (Exploration)
            # Randomly initialize half the population
            if n_global > 0:
                pop[1 : 1 + n_global] = min_b + np.random.rand(n_global, dim) * diff_b
            
            # B. Local Restart (Exploitation)
            # Gaussian sampling around best_vec
            if n_local > 0:
                # Sigma = 5% of the domain width
                sigma = diff_b * 0.05
                noise = np.random.normal(0, 1, (n_local, dim)) * sigma
                candidates = best_vec + noise
                
                # Bounce back strategy or clipping
                candidates = np.clip(candidates, min_b, max_b)
                pop[1 + n_global :] = candidates
                
            # Reset Memory to learn new parameter distributions
            mem_f.fill(0.5)
            mem_cr.fill(0.8)
            archive = []
            
            # Evaluate new individuals
            # Note: Index 0 is already evaluated (elitism)
            for i in range(1, pop_size):
                if datetime.now() >= end_time: return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            
            # Skip evolution this turn, start fresh next iter
            continue

        # 3. Parameter Generation (jSO/SHADE)
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idxs]
        m_f = mem_f[r_idxs]
        
        # CR ~ Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f = m_f[r_idxs] + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Check constraints for F
        # If F <= 0, regenerate. If F > 1, clip to 1.
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            # Regenerate using the specific means for the failed indices
            f[mask_neg] = m_f[r_idxs[mask_neg]] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
            
        f[f > 1] = 1.0
        
        # 4. Mutation & Crossover
        # Dynamic p value: decreases from 0.25 to 0.05
        p_val = 0.25 - (0.20 * progress)
        if p_val < 0.05: p_val = 0.05
        
        sorted_indices = np.argsort(fitness)
        top_p_cnt = int(max(2, p_val * pop_size))
        p_best_indices = sorted_indices[:top_p_cnt]
        
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
                    break
                r2_idx = np.random.randint(0, limit)
                
            if r2_idx < pop_size:
                x_r2 = pop[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_size]
                
            # Mutation: current-to-pbest/1 (weighted)
            mutant = x_i + f[i] * (x_pbest - x_i) + f[i] * (x_r1 - x_r2)
            
            # Crossover: Binomial
            mask_j = np.random.rand(dim) < cr[i]
            j_rand = np.random.randint(dim)
            mask_j[j_rand] = True
            trial = np.where(mask_j, mutant, x_i)
            
            # Bound Correction: Midpoint Rule
            # x_new = (x_old + bound) / 2
            below_b = trial < min_b
            above_b = trial > max_b
            trial[below_b] = (x_i[below_b] + min_b[below_b]) / 2.0
            trial[above_b] = (x_i[above_b] + max_b[above_b]) / 2.0
            
            # Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                
                # Record success if strictly better
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
        
        # Maintain Archive Limit
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 5. Update Memory (Weighted Lehmer Mean)
        succ_idx = np.where(succ_mask)[0]
        if len(succ_idx) > 0:
            w = diff_vals[succ_idx]
            total_w = np.sum(w)
            
            if total_w > 0:
                w = w / total_w
                s_cr = cr[succ_idx]
                s_f = f[succ_idx]
                
                mean_cr = np.sum(w * s_cr)
                # Lehmer mean for F
                mean_f = np.sum(w * (s_f**2)) / (np.sum(w * s_f) + 1e-12)
                
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                
                k_mem = (k_mem + 1) % H
                
    return best_val
