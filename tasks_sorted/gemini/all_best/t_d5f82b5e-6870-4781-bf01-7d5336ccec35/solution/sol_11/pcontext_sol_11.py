#The improved algorithm is a **Self-Adaptive Differential Evolution with Success-History Adaptation (SHADE)** that incorporates **Linear Parameter Adaptation** and a **Multi-Stage Restart Mechanism**.
#
#### Improvements Overview
#1.  **Adaptive Greedy Parameter ($p$)**: Instead of a fixed top-11% for `p-best` selection, the algorithm linearly adapts $p$ from `0.2` (exploration) down to `0.05` (exploitation) over the time budget. This mimics the behavior of L-SHADE without requiring complex population resizing.
#2.  **Soft Bound Handling**: Replaces strict clipping with a mean-based correction (`(limit + old) / 2`). This preserves the distributional properties of the population near boundaries better than piling mass on the edge.
#3.  **Refined Restart Logic**:
#    *   **Convergence Restart**: Triggered when population variance vanishes ($< 1e-9$). Hard reset keeping only the best solution.
#    *   **Stagnation Restart**: Triggered if no global improvement for `patience` generations. Soft reset keeping the top 20% elite and randomizing the rest, plus "jolting" the SHADE memory to escape parameter basins.
#4.  **SHADE Architecture**: Retains the robust memory-based adaptation of $F$ and $CR$ using Weighted Lehmer Mean, which proved most effective in previous attempts.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the objective function using SHADE (Success-History Adaptive Differential Evolution)
    with adaptive p-best selection, soft bound handling, and multi-stage restarts.
    """
    # Setup timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper: Time Check ---
    def check_time():
        return datetime.now() - start_time >= time_limit

    # --- Configuration ---
    # Population size: Adaptive to dimension (standard SHADE heuristic)
    # Clamped to ensure reasonable generation count within time limits
    pop_size = int(20 * dim)
    pop_size = max(30, min(100, pop_size))
    
    # Archive size: Stores replaced individuals to maintain diversity
    archive_size = int(2.5 * pop_size)
    
    # SHADE Memory Parameters
    H = 5 # History memory size
    mem_M_CR = np.full(H, 0.5) # Memory for Crossover Rate
    mem_M_F = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0                  # Memory pointer
    
    # Restart triggers
    patience = 40       # Generations without global improvement -> Soft Restart
    conv_thresh = 1e-9  # Standard deviation threshold -> Hard Restart
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    archive = []
    
    # Global Best Tracking
    global_best_val = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if check_time(): return global_best_val
        val = func(population[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val
            
    # --- Main Optimization Loop ---
    prev_global_best = global_best_val
    gen_no_improv = 0
    
    while not check_time():
        
        # 1. Calculate Progress (0.0 to 1.0)
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # 2. Adaptive p-best value
        # Linearly decay from 0.2 (explore) to 0.05 (exploit)
        p_val = 0.2 - 0.15 * progress
        num_p_best = max(2, int(pop_size * p_val))
        
        # 3. Sort Population
        sorted_indices = np.argsort(fitness)
        
        # --- Restart Logic ---
        fit_std = np.std(fitness)
        
        # A. Convergence Restart (Hard Reset)
        if fit_std < conv_thresh:
            # Keep only the single best, re-init the rest
            best_idx = sorted_indices[0]
            best_indiv = population[best_idx].copy()
            best_fit = fitness[best_idx]
            
            # Re-initialize
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_indiv
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fit
            
            # Reset Memory & Archive to allow fresh learning
            mem_M_CR.fill(0.5)
            mem_M_F.fill(0.5)
            archive = []
            gen_no_improv = 0
            
            # Evaluate new population (skip index 0)
            for i in range(1, pop_size):
                if check_time(): return global_best_val
                val = func(population[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
            
            sorted_indices = np.argsort(fitness)
            
        # B. Stagnation Restart (Soft Reset)
        elif gen_no_improv >= patience:
            # Keep top 20% elite, randomize bottom 80%
            elite_count = int(pop_size * 0.2)
            reset_indices = sorted_indices[elite_count:]
            
            for idx in reset_indices:
                if check_time(): return global_best_val
                population[idx] = min_b + np.random.rand(dim) * diff_b
                val = func(population[idx])
                fitness[idx] = val
                if val < global_best_val:
                    global_best_val = val
            
            # Perturb memory to escape parameter stagnation
            mem_M_F = np.clip(mem_M_F + np.random.uniform(-0.1, 0.1, H), 0.1, 1.0)
            mem_M_CR = np.clip(mem_M_CR + np.random.uniform(-0.1, 0.1, H), 0.0, 1.0)
            
            gen_no_improv = 0
            sorted_indices = np.argsort(fitness)

        # 4. Generate SHADE Parameters
        # Select random memory slot for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_M_CR[r_idx]
        m_f = mem_M_F[r_idx]
        
        # Generate CR: Normal(mean=M_CR, std=0.1)
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # Generate F: Cauchy(loc=M_F, scale=0.1)
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F values
        for k in range(pop_size):
            while F[k] <= 0:
                # Regenerate if <= 0
                F[k] = m_f[k] + 0.1 * np.random.standard_cauchy()
            if F[k] > 1:
                F[k] = 1.0
                
        # 5. Evolution Step
        success_F = []
        success_CR = []
        diff_fitness = []
        new_archive = []
        
        # Indices of current p-best candidates
        pbest_choices = sorted_indices[:num_p_best]
        
        for i in range(pop_size):
            if check_time(): return global_best_val
            
            x_i = population[i]
            
            # Strategy: current-to-pbest/1/bin
            
            # Select p-best
            x_pbest = population[np.random.choice(pbest_choices)]
            
            # Select r1 (distinct from i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select r2 (distinct from i, r1, from Union(Pop, Archive))
            union_len = pop_size + len(archive)
            r2 = np.random.randint(0, union_len)
            while True:
                if r2 < pop_size:
                    if r2 == i or r2 == r1:
                        r2 = np.random.randint(0, union_len)
                        continue
                break
            
            if r2 < pop_size:
                x_r2 = population[r2]
            else:
                x_r2 = archive[r2 - pop_size]
            
            # Mutation
            mutant = x_i + F[i] * (x_pbest - x_i) + F[i] * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            mask = np.random.rand(dim) < CR[i]
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            trial = np.where(mask, mutant, x_i)
            
            # Bound Handling (Soft Correction)
            # If value violates bound, set to mean of bound and old value
            # This prevents piling up at the edges
            low_viol = trial < min_b
            trial[low_viol] = (min_b[low_viol] + x_i[low_viol]) * 0.5
            high_viol = trial > max_b
            trial[high_viol] = (max_b[high_viol] + x_i[high_viol]) * 0.5
            
            # Evaluate
            f_trial = func(trial)
            
            # Selection
            if f_trial <= fitness[i]:
                # Add parent to archive candidates
                new_archive.append(x_i.copy())
                
                # Record success for memory update
                if f_trial < fitness[i]:
                    success_CR.append(CR[i])
                    success_F.append(F[i])
                    diff_fitness.append(fitness[i] - f_trial)
                
                # Update population
                fitness[i] = f_trial
                population[i] = trial
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    
        # 6. Update Archive
        for vec in new_archive:
            if len(archive) < archive_size:
                archive.append(vec)
            else:
                # Random replacement
                archive[np.random.randint(0, archive_size)] = vec
                
        # 7. Update SHADE Memory (Weighted Lehmer Mean)
        if len(success_F) > 0:
            s_F = np.array(success_F)
            s_CR = np.array(success_CR)
            df = np.array(diff_fitness)
            
            # Weights proportional to fitness improvement
            total_diff = np.sum(df)
            if total_diff > 0:
                weights = df / total_diff
            else:
                weights = np.ones(len(df)) / len(df)
            
            # Mean F (Weighted Lehmer Mean)
            mean_F = np.sum(weights * (s_F ** 2)) / (np.sum(weights * s_F) + 1e-10)
            mem_M_F[k_mem] = np.clip(mean_F, 0.1, 1.0)
            
            # Mean CR (Weighted Arithmetic Mean)
            mean_CR = np.sum(weights * s_CR)
            mem_M_CR[k_mem] = np.clip(mean_CR, 0.0, 1.0)
            
            # Advance memory pointer
            k_mem = (k_mem + 1) % H
            
        # 8. Update Stagnation Counter
        if global_best_val < prev_global_best:
            gen_no_improv = 0
            prev_global_best = global_best_val
        else:
            gen_no_improv += 1
            
    return global_best_val
