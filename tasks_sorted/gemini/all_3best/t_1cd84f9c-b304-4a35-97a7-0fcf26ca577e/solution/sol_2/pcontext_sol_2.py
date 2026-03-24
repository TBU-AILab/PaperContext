#Here is an improved algorithm. It implements a simplified version of **L-SHADE (Success-History based Adaptive Differential Evolution with Linear Population Reduction)**, combined with a **Restart Mechanism**.
#
#**Why this is better:**
#1.  **State-of-the-art Mutation (`current-to-pbest`):** Unlike the basic random mutation in previous attempts, this strategy guides the search towards the best individuals found so far (exploitation) while maintaining diversity using an external archive (exploration).
#2.  **Historical Memory:** It remembers successful $F$ and $CR$ parameters from previous generations to adapt to the function's landscape dynamically.
#3.  **Archive mechanism:** It stores recently replaced superior individuals to prevent premature convergence, providing a pool of alternative difference vectors.
#4.  **Stagnation Restart:** If the population converges to a single point (stagnation) before time runs out, it restarts the search in a new area while preserving the global best.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using a SHADE-inspired Differential Evolution algorithm.
    Includes 'current-to-pbest' mutation, parameter history memory, 
    external archive, and a restart mechanism for stagnation.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- 1. Parameters & Configuration ---
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Population size: adaptive to dimension but bounded for speed
    pop_size = int(max(30, min(100, 15 * dim)))
    
    # SHADE Memory parameters (History size H)
    H = 5
    mem_M_F = np.full(H, 0.5)   # Memory for F (scale factor)
    mem_M_CR = np.full(H, 0.5)  # Memory for CR (crossover rate)
    k_mem = 0                   # Current memory index pointer
    
    # External Archive
    archive = []
    
    # --- 2. State Variables ---
    population = np.zeros((pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    
    best_ind = None
    best_fit = float('inf')

    # --- 3. Helper Functions ---
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    def safe_evaluate(x):
        try:
            return func(x)
        except Exception:
            return float('inf')

    def initialize_population(indices):
        """Initializes specific indices of the population randomly."""
        nonlocal best_ind, best_fit
        for i in indices:
            if is_time_up(): return
            ind = min_b + np.random.rand(dim) * diff_b
            val = safe_evaluate(ind)
            population[i] = ind
            fitness[i] = val
            if val < best_fit:
                best_fit = val
                best_ind = ind.copy()

    # --- 4. Initial Setup ---
    initialize_population(range(pop_size))
    if is_time_up(): return best_fit

    # --- 5. Main Optimization Loop ---
    while not is_time_up():
        
        # A. Restart Logic (Stagnation Check)
        # If population diversity is extremely low, restart search
        if np.std(fitness) < 1e-8:
            # Keep the global best, randomize others
            indices_to_reset = [i for i in range(pop_size) if fitness[i] != best_fit]
            if not indices_to_reset: # If all are identical to best
                indices_to_reset = range(1, pop_size)
            
            initialize_population(indices_to_reset)
            archive = [] # Clear archive on restart
            if is_time_up(): return best_fit
            continue

        # B. Parameter Generation
        # Sort population to find top p-best individuals later
        sorted_indices = np.argsort(fitness)
        
        # Generate adaptive parameters for this generation
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idx]
        m_f = mem_M_F[r_idx]

        # CR: Normal distribution, clipped [0, 1]
        cr_vals = np.random.normal(m_cr, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)

        # F: Cauchy distribution. Retry if <= 0 (approximated by clamping), clip if > 1
        f_vals = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f_vals = np.minimum(f_vals, 1.0)
        f_vals[f_vals <= 0] = 0.5 # Fallback for non-positive values

        # Prepare for evolution
        success_scr = []
        success_sf = []
        fitness_diffs = []
        
        # Union of Population and Archive for mutation
        if len(archive) > 0:
            pop_archive = np.vstack((population, np.array(archive)))
        else:
            pop_archive = population

        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        update_mask = np.zeros(pop_size, dtype=bool)

        # C. Evolution Cycle (Mutation -> Crossover -> Selection)
        for i in range(pop_size):
            if is_time_up(): return best_fit

            # 1. Mutation: current-to-pbest/1
            # Select pbest from top p% (random p in [2/N, 0.2])
            p = np.random.uniform(2/pop_size, 0.2)
            top_p_cnt = int(max(2, p * pop_size))
            pbest_idx = sorted_indices[np.random.randint(0, top_p_cnt)]
            
            # Select r1 (from population, != i)
            candidates = [x for x in range(pop_size) if x != i]
            r1_idx = random.choice(candidates)
            
            # Select r2 (from union of pop + archive, != i, != r1)
            # Simplified: just pick random from union, low collision prob is acceptable
            r2_idx = np.random.randint(0, len(pop_archive))
            
            x_i = population[i]
            x_pbest = population[pbest_idx]
            x_r1 = population[r1_idx]
            x_r2 = pop_archive[r2_idx]
            
            F = f_vals[i]
            mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)

            # 2. Crossover (Binomial)
            CR = cr_vals[i]
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR
            mask[j_rand] = True # Ensure at least one change
            
            trial = np.where(mask, mutant, x_i)
            trial = np.clip(trial, min_b, max_b)

            # 3. Selection
            f_trial = safe_evaluate(trial)

            if f_trial <= fitness[i]:
                # Improvement or equal: Accept
                new_population[i] = trial
                new_fitness[i] = f_trial
                update_mask[i] = True
                
                # Store success data
                success_scr.append(CR)
                success_sf.append(F)
                fitness_diffs.append(fitness[i] - f_trial)
                
                # Add replaced parent to archive
                archive.append(x_i.copy())

                # Update Global Best
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_ind = trial.copy()
            else:
                # Keep parent
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]

        # Apply updates
        if is_time_up(): return best_fit
        population = new_population
        fitness = new_fitness

        # D. Archive Maintenance
        # If archive grows too big, remove random elements
        while len(archive) > pop_size:
            archive.pop(random.randint(0, len(archive) - 1))

        # E. Update History Memory (Weighted Lehmer Mean)
        if len(success_scr) > 0:
            success_scr = np.array(success_scr)
            success_sf = np.array(success_sf)
            fitness_diffs = np.array(fitness_diffs)
            
            # Weights proportional to fitness improvement
            total_diff = np.sum(fitness_diffs)
            weights = fitness_diffs / total_diff if total_diff > 0 else np.ones(len(fitness_diffs))/len(fitness_diffs)
            
            # Update M_CR (Weighted Mean)
            m_cr_new = np.sum(weights * success_scr)
            if np.isnan(m_cr_new): m_cr_new = 0.5
            mem_M_CR[k_mem] = m_cr_new
            
            # Update M_F (Weighted Lehmer Mean)
            denom = np.sum(weights * success_sf)
            if denom == 0:
                m_f_new = 0.5
            else:
                m_f_new = np.sum(weights * (success_sf ** 2)) / denom
            
            if np.isnan(m_f_new): m_f_new = 0.5
            mem_M_F[k_mem] = m_f_new
            
            # Advance memory pointer
            k_mem = (k_mem + 1) % H

    return best_fit
