#Here is the fully operational, self-contained Python implementation.
#
##### Improvements:
#1.  **L-SHADE Algorithm**: Implements the **Success-History based Adaptive Differential Evolution** (L-SHADE), which is superior to standard JADE. It maintains a historical memory ($M_{CR}, M_F$) of successful control parameters to adapt to the fitness landscape over time.
#2.  **External Archive**: An archive of inferior solutions is maintained to preserve population diversity. The mutation strategy `current-to-pbest/1` uses this archive to generate difference vectors, preventing premature convergence.
#3.  **Weighted Parameter Adaptation**: Instead of simple averages, the algorithm updates its parameter memory using a **Weighted Lehmer Mean** based on fitness improvement magnitudes. This ensures that parameters generating larger fitness gains have more influence on future generations.
#4.  **Stagnation Detection & Restart**: Monitors population variance. If the search stagnates (low variance), it triggers a partial restart—keeping elite solutions while re-initializing the rest to explore new basins of attraction.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Success-History based Adaptive Differential Evolution)
    with External Archive and Restart Mechanism.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Standard L-SHADE heuristic (18 * dim), clamped for safety
    pop_size = int(max(30, 18 * dim))
    
    # Archive Settings
    archive = []
    max_archive_size = pop_size
    
    # Memory for Adaptive Parameters (H=6)
    h_mem = 6
    m_cr = np.full(h_mem, 0.5) # Memory for Crossover Rate
    m_f = np.full(h_mem, 0.5)  # Memory for Scaling Factor
    k_mem = 0                  # Memory index pointer
    
    # Bounds Pre-processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    global_best_val = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return global_best_val
            
        val = func(population[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val

    # --- Main Optimization Loop ---
    while True:
        # Time Check
        if time.time() - start_time >= max_time:
            break
        
        # Sort population by fitness (needed for p-best selection)
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # --- Restart Strategy ---
        # If population diversity (std dev) is too low, we are likely stuck in a local optimum.
        # Trigger restart if we have sufficient time left.
        if np.std(fitness) < 1e-10 and (time.time() - start_time) < (0.85 * max_time):
            # Keep the top 10% (Elite), re-initialize the rest
            num_keep = int(max(1, 0.1 * pop_size))
            
            # Re-distribute the rest of the population
            population[num_keep:] = min_b + np.random.rand(pop_size - num_keep, dim) * diff_b
            
            # Re-evaluate new individuals
            for i in range(num_keep, pop_size):
                if time.time() - start_time >= max_time: return global_best_val
                val = func(population[i])
                fitness[i] = val
                if val < global_best_val:
                    global_best_val = val
            
            # Clear archive to avoid pulling towards old local optimum
            archive = []
            # Continue to next iteration immediately
            continue

        # --- Generate Control Parameters (F and CR) ---
        # Pick random memory slot for each individual
        r_idx = np.random.randint(0, h_mem, pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        f = np.clip(f, 0, 1) # Clip to [0, 1] usually. 
        f[f <= 0] = 0.5      # Safety clamp if Cauchy generates negative/zero
        
        # --- Mutation: current-to-pbest/1 ---
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Select p-best from top p% (p is random in [2/N, 0.2])
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        top_p_count = int(max(1, p_val * pop_size))
        
        # Union of Population and Archive for r2 selection
        if len(archive) > 0:
            pop_archive = np.vstack((population, np.array(archive)))
        else:
            pop_archive = population
        len_pa = len(pop_archive)
        
        # Generate Indices
        idx_pbest = np.random.randint(0, top_p_count, pop_size)
        idx_r1 = np.random.randint(0, pop_size, pop_size)
        idx_r2 = np.random.randint(0, len_pa, pop_size)
        
        # Resolve index conflicts (r1 != i, r2 != i, r1 != r2)
        # Using a simple iterative fix
        for i in range(pop_size):
            while idx_r1[i] == i:
                idx_r1[i] = np.random.randint(0, pop_size)
            while idx_r2[i] == i or idx_r2[i] == idx_r1[i]:
                idx_r2[i] = np.random.randint(0, len_pa)
                
        # Calculate Mutant Vectors
        x_i = population
        x_pbest = population[idx_pbest]
        x_r1 = population[idx_r1]
        x_r2 = pop_archive[idx_r2]
        
        f_vec = f[:, np.newaxis]
        mutant = x_i + f_vec * (x_pbest - x_i) + f_vec * (x_r1 - x_r2)
        mutant = np.clip(mutant, min_b, max_b)
        
        # --- Crossover (Binomial) ---
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        mask = rand_vals < cr[:, np.newaxis]
        mask[np.arange(pop_size), j_rand] = True # Ensure at least 1 dim changes
        
        trial = np.where(mask, mutant, x_i)
        
        # --- Selection & Update ---
        successful_cr = []
        successful_f = []
        fitness_diff = []
        new_archive_sol = []
        
        for i in range(pop_size):
            if time.time() - start_time >= max_time: return global_best_val
            
            f_new = func(trial[i])
            f_old = fitness[i]
            
            if f_new <= f_old:
                # Add inferior parent to archive candidates
                new_archive_sol.append(population[i].copy())
                
                # Update population
                population[i] = trial[i]
                fitness[i] = f_new
                
                # Collect success data (only if strictly better for weights)
                diff = f_old - f_new
                if diff > 0:
                    successful_cr.append(cr[i])
                    successful_f.append(f[i])
                    fitness_diff.append(diff)
                
                if f_new < global_best_val:
                    global_best_val = f_new

        # --- Archive Maintenance ---
        if new_archive_sol:
            archive.extend(new_archive_sol)
            if len(archive) > max_archive_size:
                # Remove random elements to maintain size
                excess_count = len(archive) - max_archive_size
                idxs_to_remove = np.random.choice(len(archive), excess_count, replace=False)
                # Sort descending to delete efficiently
                for idx in sorted(idxs_to_remove, reverse=True):
                    del archive[idx]

        # --- Update Historical Memory (L-SHADE) ---
        if len(successful_f) > 0:
            s_cr = np.array(successful_cr)
            s_f = np.array(successful_f)
            w = np.array(fitness_diff)
            
            # Normalize weights
            if np.sum(w) > 0:
                w = w / np.sum(w)
            else:
                w = np.ones(len(w)) / len(w)
            
            # Weighted Lehmer Mean for F
            mean_f_lehmer = np.sum(w * s_f**2) / (np.sum(w * s_f) + 1e-10)
            
            # Weighted Arithmetic Mean for CR
            mean_cr = np.sum(w * s_cr)
            
            # Update memory slot
            m_f[k_mem] = mean_f_lehmer
            m_cr[k_mem] = mean_cr
            
            k_mem = (k_mem + 1) % h_mem

    return global_best_val
