#The best performing algorithm so far was Algorithm 2 (`8.22`), which implemented a simplified SHADE strategy. Algorithm 1 (`47.96`) attempted to add Opposition-Based Learning and Bounce-Back boundary handling but performed significantly worse, likely due to the computational overhead of OBL and the bounce-back strategy preventing convergence to boundary-located optima.
#
#To improve upon Algorithm 2, I propose implementing the full **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution)** algorithm, adapted for a **time-based budget**.
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR):** Unlike Algorithm 2 (which had fixed population size after init), this version continuously reduces the population size from a large initial value (for exploration) to a minimal value (for exploitation) based on the elapsed time. This is the defining feature of L-SHADE that allows it to converge efficiently.
#2.  **Time-Based Schedule:** Standard L-SHADE uses function evaluation counts (NFE). Since we have a `max_time` constraint, I calculate the "progress" ratio (`elapsed / max_time`) to drive the population reduction.
#3.  **Adaptive Archive:** The external archive size now scales dynamically with the population size, ensuring the diversity buffer doesn't become stale or oversized relative to the active population.
#4.  **Refined Restart:** A restart is triggered only if the population stagnates (variance $\approx$ 0) *and* there is still significant time remaining ($<85\%$ progress), allowing the algorithm to escape local optima without wasting the final exploitation phase.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Time-Based Linear Population Size Reduction.
    
    Algorithm: L-SHADE (Linear Population Size Reduction - Success-History based Adaptive DE)
    Adaptation: Uses elapsed time instead of evaluation count to drive the linear reduction.
    
    Key Features:
    - Starts with a large population for exploration.
    - Linearly reduces population size based on time to focus on exploitation.
    - Uses 'current-to-pbest/1' mutation with an external archive.
    - Adaptive F and CR parameters based on historical success.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- 1. Setup & Configuration ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing (L-SHADE LPSR)
    # Start larger to explore, reduce to min_pop_size to exploit.
    # We cap the max size to ensure the algorithm gets enough generations 
    # even if the time limit is short or dimensions are high.
    max_pop_size = int(max(30, min(200, 20 * dim)))
    min_pop_size = 4
    
    pop_size = max_pop_size
    archive_size = max_pop_size 
    
    # SHADE Memory Parameters
    H = 6 # Size of the historical memory
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0
    
    # --- 2. Helper Functions ---
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit
    
    def get_progress():
        # Returns float [0, 1] indicating consumed time budget
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(1.0, elapsed / max_time)

    def safe_evaluate(x):
        try:
            return func(x)
        except Exception:
            return float('inf')

    # --- 3. Initialization ---
    population = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    
    best_ind = None
    best_fit = float('inf')
    
    # Generate initial population
    for i in range(pop_size):
        if is_time_up(): break
        x = min_b + np.random.rand(dim) * diff_b
        population[i] = x
        val = safe_evaluate(x)
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_ind = x.copy()
            
    if is_time_up(): return best_fit
    
    # Archive starts empty
    archive = []
    
    # --- 4. Main Optimization Loop ---
    while not is_time_up():
        
        # A. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on time progress
        progress = get_progress()
        target_size = int(round(max_pop_size + (min_pop_size - max_pop_size) * progress))
        target_size = max(min_pop_size, target_size)
        
        if pop_size > target_size:
            # Reduce population: Keep best 'target_size' individuals
            sort_idx = np.argsort(fitness)
            population = population[sort_idx[:target_size]]
            fitness = fitness[sort_idx[:target_size]]
            pop_size = target_size
            
            # Reduce archive size to match new pop_size
            archive_size = target_size
            while len(archive) > archive_size:
                archive.pop(random.randint(0, len(archive)-1))
        
        # B. Stagnation Check & Restart
        # If variance is near zero AND we are not near the end of time
        # We perform a soft restart to escape local optima
        if np.std(fitness) < 1e-9 and progress < 0.85:
            # Soft Restart: Keep best, regenerate others
            idx_keep = np.argmin(fitness)
            best_curr = population[idx_keep].copy()
            fit_curr = fitness[idx_keep]
            
            # Regenerate the rest
            new_block = min_b + np.random.rand(pop_size - 1, dim) * diff_b
            
            # Place back
            population[0] = best_curr
            fitness[0] = fit_curr
            population[1:] = new_block
            
            # Evaluate new randoms
            for k in range(1, pop_size):
                if is_time_up(): return best_fit
                val = safe_evaluate(population[k])
                fitness[k] = val
                if val < best_fit:
                    best_fit = val
                    best_ind = population[k].copy()
            
            # Reset Memory/Archive to adapt to new landscape
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            archive = []
            continue 

        # C. Generate Adaptive Parameters
        # Select random memory indices
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idxs]
        m_f = mem_M_F[r_idxs]
        
        # CR ~ Normal(M_CR, 0.1), clipped [0, 1]
        cr_vals = np.random.normal(m_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0.0, 1.0)
        
        # F ~ Cauchy(M_F, 0.1), clipped [0, 1], retry if <= 0 (approximated by setting to 0.5)
        f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f_vals = np.minimum(f_vals, 1.0)
        f_vals[f_vals <= 0] = 0.5 
        
        # D. Evolution Step
        sorted_indices = np.argsort(fitness)
        
        # Create pool for r2: Union(Population, Archive)
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
            # 1. Select pbest from top p% (p random in [2/N, 0.2])
            p_min = 2.0 / pop_size
            p = np.random.uniform(p_min, 0.2)
            top_p = int(p * pop_size)
            top_p = max(1, top_p)
            
            pbest_idx = sorted_indices[np.random.randint(0, top_p)]
            x_pbest = population[pbest_idx]
            
            # 2. Select r1 (!= i) from Population
            r1_idx = np.random.randint(0, pop_size)
            while r1_idx == i:
                r1_idx = np.random.randint(0, pop_size)
            x_r1 = population[r1_idx]
            
            # 3. Select r2 (!= i, != r1) from Union
            r2_idx = np.random.randint(0, len(pop_archive))
            while r2_idx == i or (r2_idx < pop_size and r2_idx == r1_idx):
                r2_idx = np.random.randint(0, len(pop_archive))
            x_r2 = pop_archive[r2_idx]
            
            # Difference vectors
            mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True
            trial = np.where(mask, mutant, x_i)
            
            # Bound Handling (Clip is more robust than bounce-back for general cases)
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = safe_evaluate(trial)
            
            if f_trial <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                
                success_sf.append(F)
                success_scr.append(CR)
                diff_fitness.append(fitness[i] - f_trial)
                
                # Add replaced parent to archive
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
        while len(archive) > archive_size:
            archive.pop(random.randint(0, len(archive)-1))
            
        # F. Memory Update (Weighted Lehmer Mean)
        if len(diff_fitness) > 0:
            success_sf = np.array(success_sf)
            success_scr = np.array(success_scr)
            diff_fitness = np.array(diff_fitness)
            
            total_diff = np.sum(diff_fitness)
            if total_diff > 0:
                weights = diff_fitness / total_diff
            else:
                weights = np.ones(len(diff_fitness)) / len(diff_fitness)
                
            # M_CR update (Weighted Arithmetic Mean)
            m_cr_new = np.sum(weights * success_scr)
            if np.isnan(m_cr_new) or m_cr_new < 0: m_cr_new = 0.5
            mem_M_CR[k_mem] = m_cr_new
            
            # M_F update (Weighted Lehmer Mean)
            denom = np.sum(weights * success_sf)
            if denom > 0:
                m_f_new = np.sum(weights * (success_sf ** 2)) / denom
            else:
                m_f_new = 0.5
            if np.isnan(m_f_new) or m_f_new <= 0: m_f_new = 0.5
            mem_M_F[k_mem] = m_f_new
            
            k_mem = (k_mem + 1) % H

    return best_fit
