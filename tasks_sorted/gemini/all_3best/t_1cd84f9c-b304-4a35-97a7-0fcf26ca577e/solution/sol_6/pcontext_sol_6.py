#The best performing algorithm so far was Algorithm 4 (`8.22`), which implemented a simplified SHADE strategy combined with a restart mechanism. Algorithm 2 (L-SHADE) performed worse (`16.06`), suggesting that complex population resizing strategies might incur too much overhead or disrupt the search dynamics given the specific time constraints.
#
#To improve upon Algorithm 4, I propose **Restarter SHADE with Time-Dependent Adaptation and Local-Global Restart**.
#
#**Key Improvements:**
#1.  **Time-Dependent `p` Adaptation:** Instead of a random `p` (used in Algorithm 4) or fixed `p` (standard SHADE), this algorithm linearly reduces `p` from `0.2` (exploration) to `2/N` (exploitation) based on elapsed time. This guides the `current-to-pbest` mutation from searching for new basins to converging on the best one.
#2.  **Hybrid Restart Strategy:** When stagnation is detected (low variance or no fitness improvement for ~40 generations), the algorithm triggers a restart. Unlike previous versions that randomized everything, this version keeps the elite individual and generates **10% of the new population as small Gaussian perturbations around the elite**. This ensures that if the stagnation was due to precision issues in a good basin, the algorithm has a chance to fix it (local restart), while the remaining 90% random individuals search for better basins (global restart).
#3.  **Robust F-Parameter Generation:** Instead of simply clamping non-positive $F$ values to $0.5$, this implementation uses a retry mechanism (up to a limit) to sample from the Cauchy distribution, preserving the statistical properties of the adaptation better.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarter SHADE with Time-Dependent Adaptation.
    
    Features:
    - SHADE (Success-History based Adaptive Differential Evolution).
    - Time-dependent 'p' for 'current-to-pbest' mutation (jSO-inspired).
    - Hybrid Restart: Global exploration + Local exploitation around best.
    - Robust parameter generation and granular time management.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- 1. Configuration ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing
    # Fixed moderate size to balance speed and diversity.
    # Capped at 100 to ensure reasonable generational turnover.
    pop_size = int(max(30, min(100, 15 * dim)))
    
    # Archive Size (2.0 * pop_size is standard for SHADE)
    archive_size = int(2.0 * pop_size)
    
    # SHADE Memory Parameters
    H = 5
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0
    
    # --- 2. State Initialization ---
    population = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    archive = []
    
    best_ind = None
    best_fit = float('inf')
    
    # Stagnation Tracking
    last_best_fit = float('inf')
    stagnation_count = 0
    stagnation_limit = 40
    
    # --- 3. Helper Functions ---
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    def safe_evaluate(x):
        try:
            return func(x)
        except Exception:
            return float('inf')
            
    def get_progress():
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(1.0, elapsed / max_time)

    # --- 4. Initial Population ---
    for i in range(pop_size):
        if is_time_up(): return best_fit
        x = min_b + np.random.rand(dim) * diff_b
        population[i] = x
        val = safe_evaluate(x)
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_ind = x.copy()
            
    if is_time_up(): return best_fit

    # --- 5. Main Optimization Loop ---
    while not is_time_up():
        
        # A. Stagnation Check & Restart
        # -----------------------------
        pop_std = np.std(fitness)
        
        # Update stagnation counter (using tolerance for float comparison)
        if best_fit < last_best_fit - 1e-12:
            stagnation_count = 0
            last_best_fit = best_fit
        else:
            stagnation_count += 1
        
        progress = get_progress()
        
        # Trigger Restart if:
        # 1. Variance is negligible (convergence) OR Stagnation count exceeded
        # 2. AND We are not in the very final phase of the time budget (>90%)
        if (pop_std < 1e-9 or stagnation_count > stagnation_limit) and progress < 0.90:
            
            # Reset Memory & Archive to adapt to new landscape
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            archive = []
            stagnation_count = 0
            
            # 1. Elitism: Keep Global Best at index 0
            population[0] = best_ind
            fitness[0] = best_fit
            
            # 2. Re-initialize the rest
            for i in range(1, pop_size):
                if is_time_up(): return best_fit
                
                # Hybrid Restart Strategy:
                # 10% chance: Small Gaussian perturbation around Best (Local Search/Exploitation)
                # 90% chance: Pure Global Random (Global Search/Exploration)
                if random.random() < 0.1:
                    sigma = 0.01 * diff_b # 1% of the domain width
                    x = best_ind + np.random.normal(0, 1, dim) * sigma
                    x = np.clip(x, min_b, max_b)
                else:
                    x = min_b + np.random.rand(dim) * diff_b
                
                population[i] = x
                val = safe_evaluate(x)
                fitness[i] = val
                
                if val < best_fit:
                    best_fit = val
                    best_ind = x.copy()
            
            continue # Skip normal evolution this cycle
            
        # B. Parameter Adaptation (SHADE)
        # -------------------------------
        sorted_indices = np.argsort(fitness)
        
        # Calculate adaptive 'p' (Linear Decay)
        # Decays from 0.2 (exploration) to 2/N (exploitation)
        p_min = 2.0 / pop_size
        p_max = 0.2
        p_curr = p_max - (p_max - p_min) * progress
        p_curr = max(p_min, p_curr)
        
        # Generate F and CR from memory
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idxs]
        m_f = mem_M_F[r_idxs]
        
        # CR ~ Normal(M_CR, 0.1)
        cr_vals = np.random.normal(m_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1)
        f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Robust Retry for F <= 0 (Try up to 3 times to get positive F)
        retry_mask = f_vals <= 0
        retries = 0
        while np.any(retry_mask) and retries < 3:
            n_retry = np.sum(retry_mask)
            f_vals[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(n_retry)
            retry_mask = f_vals <= 0
            retries += 1
        
        # Fallback for remaining negative values and clamp > 1
        f_vals[f_vals <= 0] = 0.5
        f_vals = np.minimum(f_vals, 1.0)
        
        # C. Evolution Cycle
        # ------------------
        # Create Union of Population + Archive for r2 selection
        if len(archive) > 0:
            pop_archive = np.vstack((population, np.array(archive)))
        else:
            pop_archive = population
            
        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        success_sf = []
        success_scr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if is_time_up(): return best_fit
            
            x_i = population[i]
            F = f_vals[i]
            CR = cr_vals[i]
            
            # Mutation: current-to-pbest/1
            # Select pbest from top p%
            top_p = int(max(1, p_curr * pop_size))
            pbest_idx = sorted_indices[np.random.randint(0, top_p)]
            x_pbest = population[pbest_idx]
            
            # Select r1 (!= i)
            r1_idx = np.random.randint(0, pop_size)
            while r1_idx == i:
                r1_idx = np.random.randint(0, pop_size)
            x_r1 = population[r1_idx]
            
            # Select r2 (!= i, != r1) from Union
            r2_idx = np.random.randint(0, len(pop_archive))
            while r2_idx == i or (r2_idx < pop_size and r2_idx == r1_idx):
                r2_idx = np.random.randint(0, len(pop_archive))
            x_r2 = pop_archive[r2_idx]
            
            mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, x_i)
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = safe_evaluate(trial)
            
            if f_trial <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                
                # If strictly better, record success
                if f_trial < fitness[i]:
                    success_scr.append(CR)
                    success_sf.append(F)
                    diff_fitness.append(fitness[i] - f_trial)
                    archive.append(x_i.copy())
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_ind = trial.copy()
            else:
                new_population[i] = x_i
                new_fitness[i] = fitness[i]
                
        population = new_population
        fitness = new_fitness
        
        # D. Archive Maintenance
        # ----------------------
        while len(archive) > archive_size:
            archive.pop(random.randint(0, len(archive)-1))
            
        # E. Memory Update (Weighted Lehmer Mean)
        # ---------------------------------------
        if len(diff_fitness) > 0:
            w = np.array(diff_fitness)
            total_diff = np.sum(w)
            
            if total_diff > 0:
                w = w / total_diff
                
                # M_CR Update (Weighted Mean)
                m_cr_new = np.sum(w * np.array(success_scr))
                mem_M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                
                # M_F Update (Weighted Lehmer Mean)
                sf = np.array(success_sf)
                denom = np.sum(w * sf)
                if denom > 0:
                    m_f_new = np.sum(w * (sf ** 2)) / denom
                    mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                else:
                    mem_M_F[k_mem] = 0.5
                    
                k_mem = (k_mem + 1) % H

    return best_fit
