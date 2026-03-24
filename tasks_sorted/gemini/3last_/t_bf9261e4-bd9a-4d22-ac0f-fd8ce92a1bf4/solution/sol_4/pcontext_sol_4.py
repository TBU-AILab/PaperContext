#The following algorithm implements **SHADE-R (Success-History Adaptive Differential Evolution with Restarts)**.
#
#**Improvements over previous attempts:**
#1.  **Adaptive Parameters (SHADE)**: Unlike standard DE, it automatically tunes the mutation factor ($F$) and crossover rate ($CR$) using historical success data. This allows it to adapt to different landscape types (e.g., separable vs. non-separable) without manual tuning.
#2.  **Vectorized Operations**: The implementation utilizes NumPy for bulk generation of trial vectors (mutation and crossover) and boundary handling. This minimizes Python loop overhead, allowing for significantly more function evaluations within the limited time.
#3.  **Restart Mechanism**: To address the multimodal nature of difficult problems (and improve upon the 30.85 result), the algorithm detects stagnation (low population variance). When triggered, it keeps the best solution found so far and restarts the rest of the population to explore new areas, preventing premature convergence.
#4.  **External Archive**: It maintains an archive of recently inferior solutions to maintain diversity in the "current-to-pbest" mutation strategy, balancing exploration and exploitation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE-R (Success-History Adaptive Differential Evolution with Restarts).
    """
    start_time = datetime.now()
    # Reserve 5% of time for safe return
    time_limit = timedelta(seconds=max_time * 0.95)
    
    # Pre-process bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Population size: 18*dim is standard for SHADE, but we cap it for efficiency
    pop_size = int(18 * dim)
    pop_size = max(30, min(pop_size, 150))
    
    # Memory for adaptive parameters (History size H=5)
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive for diversity maintenance (Size = 2 * pop_size)
    archive_size = int(2 * pop_size)
    archive = np.empty((archive_size, dim))
    archive_count = 0
    
    # Helper to check remaining time
    def has_time():
        return (datetime.now() - start_time) < time_limit

    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None
    
    # Initial Evaluation
    for i in range(pop_size):
        if not has_time(): return best_fitness
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # --- Main Loop ---
    while has_time():
        
        # 1. Restart Check
        # If population variance is extremely low, we are likely stuck in a local optimum.
        # Restart population, keeping only the best individual.
        if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
            # Re-initialize population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol # Keep elite
            
            # Reset fitness array
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fitness
            
            # Reset memory and archive to adapt to new region
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            archive_count = 0
            
            # Evaluate new population (skipping elite at index 0)
            for i in range(1, pop_size):
                if not has_time(): return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            continue # Skip to next generation
            
        # 2. Parameter Adaptation
        # Sort population to easily select p-best
        sorted_indices = np.argsort(fitness)
        pop_sorted = pop[sorted_indices]
        
        # Pick random memory indices
        r_indices = np.random.randint(0, H, pop_size)
        m_f = mem_f[r_indices]
        m_cr = mem_cr[r_indices]
        
        # Generate F using Cauchy distribution
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F values (must be > 0 and <= 1)
        # Vectorized retry for F <= 0
        bad_f_mask = F <= 0
        while np.any(bad_f_mask):
            count = np.sum(bad_f_mask)
            F[bad_f_mask] = m_f[bad_f_mask] + 0.1 * np.random.standard_cauchy(count)
            bad_f_mask = F <= 0
        F = np.clip(F, 0, 1) # Clip max to 1.0
        
        # Generate CR using Normal distribution
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0, 1)
        
        # 3. Mutation: DE/current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Select X_pbest: Randomly from top p% (p in [0.05, 0.2])
        p = np.random.uniform(0.05, 0.2)
        top_p_count = max(2, int(pop_size * p))
        pbest_indices = np.random.randint(0, top_p_count, pop_size)
        x_pbest = pop_sorted[pbest_indices]
        
        # Select X_r1: Random from population
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = pop[r1_indices]
        
        # Select X_r2: Random from Union(Population, Archive)
        total_avail = pop_size + archive_count
        r2_indices = np.random.randint(0, total_avail, pop_size)
        
        # Efficiently build x_r2 based on indices
        x_r2 = np.empty((pop_size, dim))
        mask_pop = r2_indices < pop_size
        mask_arc = ~mask_pop
        
        if np.any(mask_pop):
            x_r2[mask_pop] = pop[r2_indices[mask_pop]]
        if np.any(mask_arc):
            # Archive indices are offset by pop_size in r2_indices
            arc_indices = r2_indices[mask_arc] - pop_size
            x_r2[mask_arc] = archive[arc_indices]
            
        # Calculate Mutant Vectors (vectorized)
        F_col = F[:, np.newaxis]
        mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # 4. Crossover: Binomial
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR[:, np.newaxis]
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 5. Boundary Handling (Clip)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 6. Evaluation and Selection
        success_f = []
        success_cr = []
        success_diff = []
        
        for i in range(pop_size):
            if not has_time(): return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial < fitness[i]:
                # Improvement: Solution enters next generation
                
                # Add displaced solution to archive
                if archive_count < archive_size:
                    archive[archive_count] = pop[i].copy()
                    archive_count += 1
                else:
                    # Random replacement if archive full
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = pop[i].copy()
                
                # Record success info
                diff = fitness[i] - f_trial
                success_diff.append(diff)
                success_f.append(F[i])
                success_cr.append(CR[i])
                
                # Update population
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                # Update global best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    
        # 7. Update History Memory (Weighted Lehmer Mean)
        if len(success_diff) > 0:
            s_diff = np.array(success_diff)
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            # Weights based on fitness improvement
            weights = s_diff / np.sum(s_diff)
            
            # Update Memory CR (Weighted Mean)
            m_cr_new = np.sum(weights * s_cr)
            mem_cr[k_mem] = np.clip(m_cr_new, 0, 1)
            
            # Update Memory F (Weighted Lehmer Mean)
            m_f_num = np.sum(weights * (s_f ** 2))
            m_f_den = np.sum(weights * s_f)
            if m_f_den > 0:
                m_f_new = m_f_num / m_f_den
                mem_f[k_mem] = np.clip(m_f_new, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
    return best_fitness
