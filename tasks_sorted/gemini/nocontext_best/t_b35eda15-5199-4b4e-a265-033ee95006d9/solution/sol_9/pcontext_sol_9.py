#To improve upon the JADE algorithm, we will implement **L-SHADE (Success-History based Adaptive Differential Evolution with Linear Population Size Reduction)**.
#
#**Why this is better:**
#1.  **Linear Population Size Reduction (LPSR):** The previous algorithm used a fixed population size. L-SHADE starts with a large population (to explore the global landscape) and linearly reduces the population size as time progresses. This forces the algorithm to shift from **Exploration** to **Exploitation**, converging much tighter on the minimum in the final stages.
#2.  **External Archive:** It utilizes an archive of inferior solutions recently replaced. This preserves diversity information without needing to re-evaluate them, allowing for more robust mutation directions.
#3.  **Historical Memory:** Instead of a single moving average for parameters $F$ and $C_r$, L-SHADE maintains a history memory ($H$). This prevents a single outlier generation from distorting the adaptation process.
#
#Here is the fully operational, self-contained Python code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Implementation of L-SHADE (Linear Success-History Adaptive Differential Evolution).
    Features:
    1. LPSR: Linearly reduces population size over time to shift from exploration to exploitation.
    2. External Archive: Uses rejected parents to maintain diversity in mutation.
    3. Historical Memory: Adapts F and Cr parameters based on a history of success.
    """
    
    # --- Configuration ---
    # Initial Population size (Starting high for exploration)
    # R_Ninit is typically 18, but we cap it for very high dimensions to stay within time limits
    N_init = int(round(max(20, 18 * dim)))
    if N_init > 500: N_init = 500
    
    # Minimum population size (Ending low for exploitation)
    N_min = 4 
    
    # History memory size
    H = 6
    
    # Time management
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- State Variables ---
    pop_size = N_init
    # Memory for adaptive parameters (M_CR, M_F) initialized to 0.5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer
    
    # Archive for inferior solutions (starts empty)
    archive = []
    
    # --- Initialization: Latin Hypercube Sampling ---
    # Ensures grid-like coverage of the space initially
    population = np.empty((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(0, 1, pop_size + 1)
        offsets = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(offsets)
        population[:, d] = min_b[d] + offsets * diff_b[d]
        
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    best_val = float('inf')
    best_vec = None
    
    for i in range(pop_size):
        if datetime.now() >= end_time: return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()

    # Max function evaluations budget estimation (dynamic based on speed)
    # We use time ratio to drive LPSR
    
    # --- Main Loop ---
    while True:
        # Check Time
        now = datetime.now()
        if now >= end_time:
            return best_val
            
        # --- 1. Linear Population Size Reduction (LPSR) ---
        # Calculate progress ratio (0.0 to 1.0)
        elapsed = (now - start_time).total_seconds()
        ratio = elapsed / max_time
        if ratio > 1.0: ratio = 1.0
        
        # Calculate new target population size
        new_pop_size = int(round((N_min - N_init) * ratio + N_init))
        
        # If we need to reduce population
        if new_pop_size < pop_size:
            # Sort by fitness (worst at the end)
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Truncate
            remove_count = pop_size - new_pop_size
            # Simply drop the worst (last indices)
            population = population[:-remove_count]
            fitness = fitness[:-remove_count]
            
            # Resize Archive if it exceeds new population size
            if len(archive) > new_pop_size:
                # Randomly remove elements to fit size
                del archive[new_pop_size:]
                
            pop_size = new_pop_size
            
            # Resync best vector just in case
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_val:
                best_val = fitness[best_idx]
                best_vec = population[best_idx].copy()
                
        # --- 2. Parameter Generation ---
        # Pick random memory indices
        r_idx = np.random.randint(0, H, pop_size)
        m_cr_selected = mem_cr[r_idx]
        m_f_selected = mem_f[r_idx]
        
        # Generate CR ~ Normal(M_CR, 0.1)
        cr_vals = np.random.normal(m_cr_selected, 0.1)
        cr_vals = np.clip(cr_vals, 0, 1)
        # Ensure CR is not negative (though clip handles it, logic dictates 0-1)
        
        # Generate F ~ Cauchy(M_F, 0.1)
        # Cauchy: loc + scale * tan(pi * (rand - 0.5)) or standard_cauchy
        f_vals = m_f_selected + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints
        # If F > 1, clamp to 1
        f_vals[f_vals > 1] = 1.0
        # If F <= 0, regenerate until > 0 (or simple clamp)
        # Standard SHADE regenerates, but for speed we clamp to small epsilon
        f_vals[f_vals <= 0] = 0.1
        
        # --- 3. Mutation: current-to-pbest/1 with Archive ---
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        sorted_pop = population[sorted_indices]
        
        # p is a random value between 2/pop_size and 0.2 (usually) 
        # SHADE logic: p_best index chosen from top max(2, pop_size * p)
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        top_cnt = int(max(2, pop_size * p_val))
        pbest_idxs = np.random.randint(0, top_cnt, pop_size)
        x_pbest = sorted_pop[pbest_idxs]
        
        # r1: Random from population (distinct from i)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        # We ignore self-collision check for vectorization speed benefit
        x_r1 = population[r1_idxs]
        
        # r2: Random from Union(Population, Archive)
        # Prepare union
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((population, archive_np))
        else:
            union_pop = population
            
        r2_idxs = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_idxs]
        
        # Mutation Equation: V = X_current + F(X_pbest - X_current) + F(X_r1 - X_r2)
        # Reshape F for broadcasting
        F_col = f_vals[:, np.newaxis]
        mutants = population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)
        
        # --- 4. Crossover (Binomial) ---
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr_vals[:, np.newaxis]
        
        # Force at least one dimension
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutants, population)
        
        # Bound Constraints: Bounce back or Clip?
        # Clipping is safer for general cases
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # --- 5. Selection ---
        success_mask = np.zeros(pop_size, dtype=bool)
        diff_fitness = np.zeros(pop_size)
        
        new_archive_candidates = []
        
        for i in range(pop_size):
            # Check time periodically inside the loop if pop is huge
            if i % 50 == 0 and datetime.now() >= end_time:
                 return best_val
                 
            f_trial = func(trial_pop[i])
            
            # Greedy selection
            if f_trial <= fitness[i]:
                # Improvement
                diff = fitness[i] - f_trial
                
                # Add parent to archive
                new_archive_candidates.append(population[i].copy())
                
                # Update population
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                success_mask[i] = True
                diff_fitness[i] = diff
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()
                    
        # Update Archive
        if len(new_archive_candidates) > 0:
            archive.extend(new_archive_candidates)
            # Maintain Archive Size <= pop_size
            if len(archive) > pop_size:
                # Randomly remove excess
                num_excess = len(archive) - pop_size
                # Fast removal: shuffle and slice
                np.random.shuffle(archive)
                archive = archive[:pop_size]

        # --- 6. Update History Memory ---
        if np.any(success_mask):
            # Get successful F and CR
            succ_f = f_vals[success_mask]
            succ_cr = cr_vals[success_mask]
            df = diff_fitness[success_mask]
            
            # Calculate weights based on fitness improvement
            total_df = np.sum(df)
            if total_df == 0:
                weights = np.ones(len(df)) / len(df)
            else:
                weights = df / total_df
                
            # Weighted Lehmer Mean for F
            # mean_L = sum(w * f^2) / sum(w * f)
            num_f = np.sum(weights * (succ_f ** 2))
            den_f = np.sum(weights * succ_f)
            if den_f == 0: 
                new_mu_f = 0.5
            else:
                new_mu_f = num_f / den_f
            
            # Weighted Mean for CR
            new_mu_cr = np.sum(weights * succ_cr)
            
            # Clamping (optional but good for stability)
            new_mu_f = np.clip(new_mu_f, 0, 1)
            new_mu_cr = np.clip(new_mu_cr, 0, 1)
            
            # Update Memory at index k
            mem_f[k_mem] = new_mu_f
            mem_cr[k_mem] = new_mu_cr
            
            # Increment pointer
            k_mem = (k_mem + 1) % H

    return best_val
