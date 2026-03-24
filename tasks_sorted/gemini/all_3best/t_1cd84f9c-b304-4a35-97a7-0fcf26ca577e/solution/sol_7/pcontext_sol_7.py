The output values and codes for the last generated algorithms are as follows:
1. output value is: 2.330088369959853

 1. algorithm code is:
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


2. output value is: 8.244351875989892

 2. algorithm code is:
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


3. output value is: 16.068758131149536

 3. algorithm code is:
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


Improve.
#The best performing algorithm so far was Algorithm 4 (`2.33`), which implemented a SHADE strategy with a restart mechanism that kept the elite individual and perturbed others. The significant improvement from `16.06` to `2.33` confirms that **Restarting** is crucial for this specific problem class (likely multimodal with time constraints), and that the previous strategies were getting stuck in local optima.
#
#To beat `2.33`, I propose **Aggressive Restart SHADE with Multi-Scale Search**.
#
#**Key Improvements:**
#1.  **Multi-Scale Restart:** Algorithm 4 used a simple restart (Elite + Small Perturbation + Random). The new algorithm introduces a **"Medium" perturbation tier** (25% of population). This addresses the case where the global optimum is in a neighboring basin that requires a larger jump than the "Small" perturbation (1%) but is more probable than a completely random guess (Global).
#2.  **Aggressive Stagnation Detection:** The stagnation limit is reduced from 40 to 25 generations. Given the limited time, we want to fail fast and restart often if progress stalls, rather than grinding out minor improvements.
#3.  **Refined Restart Distribution:** The restart population is composed of: 1 Elite, 25% Small Perturbation (Precision), 25% Medium Perturbation (Local Exploration), and ~50% Global Random (Global Exploration).
#4.  **Robust Time Management:** Maintains the strict time checks and adaptive parameters of the previous best version.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Aggressive Restart SHADE with Multi-Scale Search.
    
    Improvements over previous versions:
    1. Multi-Scale Restart: When stagnated, generates new population using a mix of:
       - Global Random (Exploration)
       - Medium Gaussian perturbations around Best (Neighborhood Search)
       - Small Gaussian perturbations around Best (Precision Refinement)
    2. Aggressive Stagnation Detection: Lowers tolerance/limit to trigger restarts faster.
    3. Time-Aware p-adaptation: Adapts the greediness of mutation based on remaining time.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- 1. Configuration ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Sizing
    # Use a moderate size to allow fast iterations
    pop_size = int(max(30, min(100, 15 * dim)))
    
    # SHADE Parameters
    H = 5 # History size
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0
    archive = []
    archive_size = int(2.0 * pop_size)
    
    # --- 2. State Initialization ---
    population = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    
    best_ind = None
    best_fit = float('inf')
    
    # Stagnation tracking
    last_best_fit = float('inf')
    stagnation_count = 0
    stagnation_limit = 25 # Reduced from 40 to encourage agility
    
    # --- 3. Helpers ---
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
        
        # --- A. Stagnation & Restart Logic ---
        pop_std = np.std(fitness)
        
        # Check for improvement (tolerance 1e-12)
        if best_fit < last_best_fit - 1e-12:
            stagnation_count = 0
            last_best_fit = best_fit
        else:
            stagnation_count += 1
            
        progress = get_progress()
        
        # Restart Trigger:
        # 1. Converged (low std) OR Stagnated count
        # 2. AND enough time left (progress < 0.9)
        if (pop_std < 1e-9 or stagnation_count > stagnation_limit) and progress < 0.9:
            
            # Reset SHADE Memory
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            archive = []
            stagnation_count = 0
            
            # Preserve Elite
            population[0] = best_ind
            fitness[0] = best_fit
            
            # Multi-Scale Regeneration
            # 1. Medium Perturbation (25%): 5% of domain width
            # 2. Small Perturbation (25%): 1% of domain width
            # 3. Global Random (50%): Full exploration
            
            n_medium = int(0.25 * pop_size)
            n_small = int(0.25 * pop_size)
            # Remaining are random
            
            start_idx = 1
            
            # Generate Medium
            for i in range(start_idx, start_idx + n_medium):
                if is_time_up(): return best_fit
                x = best_ind + np.random.normal(0, 0.05, dim) * diff_b
                x = np.clip(x, min_b, max_b)
                population[i] = x
                val = safe_evaluate(x)
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_ind = x.copy()
            
            start_idx += n_medium
            
            # Generate Small
            for i in range(start_idx, start_idx + n_small):
                if is_time_up(): return best_fit
                x = best_ind + np.random.normal(0, 0.01, dim) * diff_b
                x = np.clip(x, min_b, max_b)
                population[i] = x
                val = safe_evaluate(x)
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_ind = x.copy()
                    
            start_idx += n_small
            
            # Generate Random
            for i in range(start_idx, pop_size):
                if is_time_up(): return best_fit
                x = min_b + np.random.rand(dim) * diff_b
                population[i] = x
                val = safe_evaluate(x)
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_ind = x.copy()
                    
            continue # Restart done, skip evolution
            
        # --- B. SHADE Parameter Generation ---
        sorted_indices = np.argsort(fitness)
        
        # Adaptive p for current-to-pbest
        p_min = 2.0 / pop_size
        p_max = 0.2
        # Linear decay based on time
        p_curr = p_max - (p_max - p_min) * progress
        p_curr = max(p_min, p_curr)
        
        # Generate F and CR from memory
        r_idxs = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idxs]
        m_f = mem_M_F[r_idxs]
        
        cr_vals = np.random.normal(m_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Robust Retry for F <= 0 (Try up to 3 times to get positive F)
        retry_limit = 3
        neg_mask = f_vals <= 0
        r_count = 0
        while np.any(neg_mask) and r_count < retry_limit:
            n_bad = np.sum(neg_mask)
            f_vals[neg_mask] = m_f[neg_mask] + 0.1 * np.random.standard_cauchy(n_bad)
            neg_mask = f_vals <= 0
            r_count += 1
            
        f_vals[f_vals <= 0] = 0.5
        f_vals = np.minimum(f_vals, 1.0)
        
        # --- C. Evolution ---
        if len(archive) > 0:
            pop_archive = np.vstack((population, np.array(archive)))
        else:
            pop_archive = population
            
        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        success_scr = []
        success_sf = []
        diff_fitness = []
        
        for i in range(pop_size):
            if is_time_up(): return best_fit
            
            x_i = population[i]
            F = f_vals[i]
            CR = cr_vals[i]
            
            # Mutation: current-to-pbest/1
            top_p = int(max(1, p_curr * pop_size))
            pbest_idx = sorted_indices[np.random.randint(0, top_p)]
            x_pbest = population[pbest_idx]
            
            r1_idx = np.random.randint(0, pop_size)
            while r1_idx == i:
                r1_idx = np.random.randint(0, pop_size)
            x_r1 = population[r1_idx]
            
            r2_idx = np.random.randint(0, len(pop_archive))
            while r2_idx == i or (r2_idx < pop_size and r2_idx == r1_idx):
                r2_idx = np.random.randint(0, len(pop_archive))
            x_r2 = pop_archive[r2_idx]
            
            mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            # Crossover
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
        
        # --- D. Archive Maintenance ---
        while len(archive) > archive_size:
            archive.pop(random.randint(0, len(archive)-1))
            
        # --- E. Memory Update ---
        if len(diff_fitness) > 0:
            w = np.array(diff_fitness)
            total_diff = np.sum(w)
            
            if total_diff > 0:
                w = w / total_diff
                
                m_cr_new = np.sum(w * np.array(success_scr))
                mem_M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
                
                sf = np.array(success_sf)
                denom = np.sum(w * sf)
                if denom > 0:
                    m_f_new = np.sum(w * (sf**2)) / denom
                    mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                else:
                    mem_M_F[k_mem] = 0.5
                    
                k_mem = (k_mem + 1) % H

    return best_fit