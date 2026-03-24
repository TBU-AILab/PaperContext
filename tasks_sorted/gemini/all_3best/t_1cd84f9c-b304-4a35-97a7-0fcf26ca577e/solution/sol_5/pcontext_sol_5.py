#The best performing algorithm so far was Algorithm 3 (`8.22`), which implemented a Simplified SHADE strategy with a hard restart mechanism. The "Full L-SHADE" (Algorithm 1) performed worse (`16.06`), likely due to the overhead of population resizing or overly aggressive reduction preventing recovery after restarts.
#
#To improve upon Algorithm 3, I propose **iSHADE (Improved SHADE) with Time-Dependent Adaptation and Basin Reset**.
#
#**Key Improvements over Algorithm 3:**
#1.  **Time-Dependent $p$ for Mutation:** In the `current-to-pbest` mutation strategy, the greedy parameter $p$ (controlling how elite the target vectors are) now linearly decays from `0.2` (exploration) to `2/N` (exploitation) based on the elapsed time. This mimics the behavior of advanced algorithms like jSO without the complexity of population resizing.
#2.  **Unbiased Restart:** When the population stagnates (variance $\approx$ 0), the algorithm triggers a restart. Unlike Algorithm 3, this version **resets the historical memory** ($M_F, M_{CR}$) to default values. This ensures that when the search jumps to a new random basin of attraction, it isn't biased by the parameters that were successful in the *previous* (stagnated) basin.
#3.  **Refined Population Sizing:** The population size is set to a fixed, moderate value (`12 * dim`, capped at 100). This provides a better balance between generational speed and diversity than the previous attempts.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Improved SHADE (iSHADE) with Time-Dependent Adaptation.
    
    Key Features:
    - SHADE mechanism (History-based parameter adaptation).
    - Time-dependent 'p' parameter for 'current-to-pbest' mutation (Exploration -> Exploitation).
    - Stagnation detection with Restart and Memory Reset.
    - External Archive to maintain diversity.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- 1. Configuration & Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size: Fixed, moderate size to balance speed and diversity
    # Capped at 100 to ensure sufficient generations within max_time
    pop_size = int(max(30, min(100, 12 * dim)))
    
    # SHADE Memory Parameters
    H = 5  # History size
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer
    
    # Archive (stores replaced individuals)
    archive = []
    
    # Pre-allocate arrays
    population = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    
    best_ind = None
    best_fit = float('inf')
    
    # --- 2. Helper Functions ---
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit
    
    def get_progress():
        # Returns [0.0, 1.0] representing time usage
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(1.0, elapsed / max_time)

    def safe_evaluate(x):
        try:
            return func(x)
        except Exception:
            return float('inf')
            
    # --- 3. Initialization ---
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
    
    # --- 4. Main Optimization Loop ---
    while not is_time_up():
        progress = get_progress()
        
        # A. Stagnation Check & Restart
        # If population variance is zero (converged), restart to find new basins.
        # Condition: std < 1e-8 AND we are not in the final 10% of time (save time for final polishing)
        if np.std(fitness) < 1e-8 and progress < 0.9:
            # 1. Reset Memory (crucial for unbiased search in new basin)
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            archive = []
            
            # 2. Restart Population (keep best, randomize others)
            # Find indices that are NOT the best
            # (Using float comparison tolerance is safer, but exact match usually works for copies)
            for i in range(pop_size):
                if is_time_up(): return best_fit
                
                # Skip the global best individual
                if i == np.argmin(fitness): 
                    continue
                
                # Re-initialize
                x = min_b + np.random.rand(dim) * diff_b
                population[i] = x
                val = safe_evaluate(x)
                fitness[i] = val
                
                if val < best_fit:
                    best_fit = val
                    best_ind = x.copy()
            continue

        # B. Calculate Adaptive 'p'
        # Linearly decay p from 0.2 (exploration) to 2/pop_size (exploitation)
        p_min = 2.0 / pop_size
        p_max = 0.2
        p_curr = p_max - (p_max - p_min) * progress
        p_curr = max(p_min, p_curr)
        
        # Sort population for current-to-pbest selection
        sorted_indices = np.argsort(fitness)

        # C. Generate Parameters (SHADE strategy)
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idxs]
        m_f = mem_M_F[r_idxs]
        
        # CR ~ Normal(M_CR, 0.1), clipped [0, 1]
        cr_vals = np.random.normal(m_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        # F ~ Cauchy(M_F, 0.1), clipped [0, 1]
        # If F <= 0, set to 0.5 (fallback)
        f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f_vals = np.minimum(f_vals, 1.0)
        f_vals[f_vals <= 0] = 0.5
        
        # Prepare for evolution
        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        success_f = []
        success_cr = []
        diff_fitness = []
        
        # Union of Population + Archive for r2 selection
        if len(archive) > 0:
            pop_archive = np.vstack((population, np.array(archive)))
        else:
            pop_archive = population
            
        # D. Evolution Cycle
        for i in range(pop_size):
            if is_time_up(): return best_fit
            
            # Mutation: current-to-pbest/1
            # Select pbest from top p%
            top_p_count = int(max(2, p_curr * pop_size))
            pbest_idx = sorted_indices[np.random.randint(0, top_p_count)]
            
            # Select r1 (!= i)
            r1_idx = np.random.randint(0, pop_size)
            while r1_idx == i:
                r1_idx = np.random.randint(0, pop_size)
            
            # Select r2 (!= i, != r1) from Union
            r2_idx = np.random.randint(0, len(pop_archive))
            # Note: r2_idx could point to archive (index >= pop_size) or population
            # If in population range, must check collision with i and r1
            while r2_idx == i or (r2_idx < pop_size and r2_idx == r1_idx):
                r2_idx = np.random.randint(0, len(pop_archive))
            
            x_i = population[i]
            x_pbest = population[pbest_idx]
            x_r1 = population[r1_idx]
            x_r2 = pop_archive[r2_idx]
            
            F = f_vals[i]
            CR = cr_vals[i]
            
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
                
                # Record success if strictly better
                if f_trial < fitness[i]:
                    success_f.append(F)
                    success_cr.append(CR)
                    diff_fitness.append(fitness[i] - f_trial)
                    # Add parent to archive
                    archive.append(x_i.copy())
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_ind = trial.copy()
            else:
                new_population[i] = x_i
                new_fitness[i] = fitness[i]
                
        population = new_population
        fitness = new_fitness
        
        # E. Archive Maintenance
        while len(archive) > pop_size:
            archive.pop(random.randint(0, len(archive)-1))
            
        # F. Memory Update (Weighted Lehmer Mean)
        if len(diff_fitness) > 0:
            w = np.array(diff_fitness)
            total_diff = np.sum(w)
            
            if total_diff > 0:
                w = w / total_diff
                
                # M_CR: Weighted Arithmetic Mean
                m_cr_new = np.sum(w * np.array(success_cr))
                mem_M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                
                # M_F: Weighted Lehmer Mean
                sf = np.array(success_f)
                denom = np.sum(w * sf)
                if denom > 0:
                    m_f_new = np.sum(w * (sf ** 2)) / denom
                    mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                else:
                    mem_M_F[k_mem] = 0.5
                
                k_mem = (k_mem + 1) % H

    return best_fit
