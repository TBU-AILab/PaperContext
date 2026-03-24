#The following algorithm implements **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution)**, adapted for a time-limited environment.
#
#**Key Improvements over the previous implementation:**
#1.  **Linear Population Size Reduction (LPSR):** Instead of fixed restarts or static population sizes, this algorithm starts with a large population to maximize exploration and linearly reduces it over time to maximize exploitation. This is a state-of-the-art technique in Differential Evolution (CEC winners) that naturally forces convergence without needing complex restart heuristics.
#2.  **Time-Adaptive Schedule:** Since the number of function evaluations is unknown, the reduction schedule is mapped dynamically to the `max_time` budget.
#3.  **Midpoint Boundary Correction:** Instead of clipping (which stacks solutions at bounds), this uses midpoint reflection `(current + bound) / 2`, preserving population diversity near the edges.
#4.  **Robust Parameter Adaptation:** Utilizes the standard SHADE memory mechanism with weighted Lehmer means to adapt mutation ($F$) and crossover ($CR$) rates based on successful improvements.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History based Adaptive Differential Evolution).
    
    The population size decreases linearly from N_init to N_min based on 
    elapsed time, transitioning from exploration to exploitation.
    """
    start_time = time.time()
    
    # --- 1. Initialization & Parameters ---
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    
    # Population Sizing
    # L-SHADE typically uses N_init = 18 * dim. 
    # We cap N_init to avoid excessive overhead in high dimensions within short time limits.
    init_pop_size = int(max(30, min(18 * dim, 200)))
    min_pop_size = 4
    
    curr_pop_size = init_pop_size
    
    # Initialize Population (Uniform Random)
    pop = lower_b + np.random.rand(curr_pop_size, dim) * (upper_b - lower_b)
    fitness = np.full(curr_pop_size, float('inf'))
    
    # Global Best Tracking
    global_best_fitness = float('inf')
    
    # Evaluate Initial Population
    for i in range(curr_pop_size):
        # Time safeguard
        if time.time() - start_time > max_time:
            # If we time out during init, return best found so far
            return global_best_fitness if global_best_fitness != float('inf') else func(pop[0])
            
        val = func(pop[i])
        fitness[i] = val
        if val < global_best_fitness:
            global_best_fitness = val
            
    # Sort population (required for p-best selection)
    sorted_indices = np.argsort(fitness)
    pop = pop[sorted_indices]
    fitness = fitness[sorted_indices]
    
    # --- 2. SHADE Memory Initialization ---
    H_mem = 6  # Memory size
    M_CR = np.full(H_mem, 0.5) # Memory for Crossover Rate
    M_F = np.full(H_mem, 0.5)  # Memory for Scaling Factor
    k_mem = 0  # Memory index pointer
    
    # External Archive (stores inferior solutions to preserve diversity)
    archive = [] 
    
    # --- 3. Main Optimization Loop ---
    while True:
        curr_time = time.time()
        elapsed = curr_time - start_time
        
        # Check termination
        if elapsed >= max_time:
            break
            
        # --- A. Linear Population Size Reduction (LPSR) ---
        # Calculate progress ratio (0.0 to 1.0)
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # Calculate target population size
        target_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        target_size = max(min_pop_size, target_size)
        
        # Reduce population if needed
        if curr_pop_size > target_size:
            reduce_count = curr_pop_size - target_size
            # Since pop is sorted by fitness, we remove the worst (last indices)
            pop = pop[:-reduce_count]
            fitness = fitness[:-reduce_count]
            curr_pop_size = target_size
            
            # Archive size tracks population size in L-SHADE
            if len(archive) > curr_pop_size:
                # Randomly remove elements from archive to match new size
                keep_indices = np.random.choice(len(archive), curr_pop_size, replace=False)
                archive = [archive[k] for k in keep_indices]

        # --- B. Parameter Generation ---
        # Select random memory slot for each individual
        r_idx = np.random.randint(0, H_mem, size=curr_pop_size)
        m_cr = M_CR[r_idx]
        m_f = M_F[r_idx]
        
        # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # Generate F: Cauchy(m_f, 0.1)
        # If F > 1 -> 1. If F <= 0 -> Regenerate.
        F = m_f + 0.1 * np.random.standard_cauchy(size=curr_pop_size)
        retry_mask = F <= 0
        while np.any(retry_mask):
            F[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(size=np.sum(retry_mask))
            retry_mask = F <= 0
        F = np.clip(F, 0.0, 1.0)
        
        # --- C. Mutation: current-to-pbest/1 ---
        # Select p-best individuals (top p%)
        p_best_rate = 0.11
        num_p_best = max(2, int(curr_pop_size * p_best_rate))
        
        # Indices for p-best vectors
        p_best_indices = np.random.randint(0, num_p_best, size=curr_pop_size)
        x_pbest = pop[p_best_indices]
        
        # Indices for r1 (random from population)
        r1_indices = np.random.randint(0, curr_pop_size, size=curr_pop_size)
        x_r1 = pop[r1_indices]
        
        # Indices for r2 (Union of Population and Archive)
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, len(union_pop), size=curr_pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Calculate Mutant Vectors: v = x + F*(xp - x) + F*(xr1 - xr2)
        x_curr = pop
        diff1 = x_pbest - x_curr
        diff2 = x_r1 - x_r2
        F_col = F[:, np.newaxis] # Reshape for broadcasting
        
        mutant = x_curr + F_col * diff1 + F_col * diff2
        
        # --- D. Boundary Correction (Midpoint Reflection) ---
        # If outside bounds, place halfway between current parent and bound
        mask_l = mutant < lower_b
        if np.any(mask_l):
            mutant = np.where(mask_l, (x_curr + lower_b) / 2.0, mutant)
            
        mask_u = mutant > upper_b
        if np.any(mask_u):
            mutant = np.where(mask_u, (x_curr + upper_b) / 2.0, mutant)
            
        # --- E. Crossover (Binomial) ---
        rand_vals = np.random.rand(curr_pop_size, dim)
        mask_cross = rand_vals < CR[:, np.newaxis]
        # Ensure at least one parameter is taken from mutant
        j_rand = np.random.randint(0, dim, size=curr_pop_size)
        mask_cross[np.arange(curr_pop_size), j_rand] = True
        
        trial_pop = np.where(mask_cross, mutant, pop)
        
        # --- F. Selection & Memory Update ---
        success_F = []
        success_CR = []
        diff_f = []
        
        # Evaluate Trial Vectors
        for i in range(curr_pop_size):
            # Check time strictly inside the loop
            if time.time() - start_time > max_time:
                return global_best_fitness
                
            f_trial = func(trial_pop[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                # If strictly better, record success info
                if f_trial < fitness[i]:
                    archive.append(pop[i].copy())
                    diff_f.append(fitness[i] - f_trial)
                    success_F.append(F[i])
                    success_CR.append(CR[i])
                    
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial
                
                # Update Population
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
        
        # Maintain Archive Size <= Current Pop Size
        while len(archive) > curr_pop_size:
            rem_idx = np.random.randint(0, len(archive))
            archive.pop(rem_idx)
            
        # Update Memory (M_F, M_CR) using Weighted Lehmer Mean
        if len(diff_f) > 0:
            fit_diffs = np.array(diff_f)
            weights = fit_diffs / np.sum(fit_diffs)
            
            s_F = np.array(success_F)
            s_CR = np.array(success_CR)
            
            # Lehmer Mean for F
            mean_F_lehmer = np.sum(weights * (s_F**2)) / np.sum(weights * s_F)
            M_F[k_mem] = mean_F_lehmer
            
            # Weighted Mean for CR
            mean_CR_w = np.sum(weights * s_CR)
            M_CR[k_mem] = mean_CR_w
            
            k_mem = (k_mem + 1) % H_mem
            
        # Resort population for next generation's p-best selection
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
    return global_best_fitness
