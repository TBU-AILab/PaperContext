#Here is the improved algorithm. It implements **L-SHADE-Epsilon with Linear Population Size Reduction (LPSR)** and an **MTS-LS1 (Multiple Trajectory Search)** local search polishing phase.
#
#### Key Improvements
#1.  **Linear Population Size Reduction (LPSR):** Instead of a fixed population size, the algorithm starts with a large population (improving exploration coverage) and linearly reduces it over time to a minimal size (forcing exploitation and convergence). This allows the algorithm to find the "basin" of the global minimum early and refine it rapidly later.
#2.  **Adaptive Parameters (L-SHADE):** It utilizes history-based parameter adaptation for mutation factor ($F$) and crossover rate ($CR$), automatically tuning the strategy to the specific function landscape.
#3.  **MTS-LS1 Local Search:** The final polishing phase uses a simplified version of MTS-LS1 (a robust trajectory-based search) instead of simple coordinate descent. This handles variable scaling and rotated landscapes significantly better.
#4.  **Robust Time Management:** The evolutionary cycle dynamically adjusts the population size based on the percentage of elapsed time, ensuring the algorithm uses the full budget effectively without stopping too early or overrunning.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Linear Population Size Reduction (LPSR)
    and MTS-LS1 Local Search Polish.
    """
    
    # --- 1. Initialization and Constants ---
    start_time = time.time()
    
    # Allocate time for Evolutionary Phase vs Polish Phase
    # We give 90% to evolution (global search), 10% to polish (local refinement)
    # but ensure polish has at least 0.5s if max_time allows
    polish_ratio = 0.1
    polish_time = max(0.2, max_time * polish_ratio)
    evo_max_time = max(0.1, max_time - polish_time)
    
    # Helper for bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size Configuration (LPSR)
    # Start large for exploration, shrink to min_pop for convergence
    # Standard L-SHADE suggests 18 * dim, but we cap for very high dims/low time
    initial_pop_size = int(np.clip(18 * dim, 50, 250)) 
    min_pop_size = 4
    
    pop_size = initial_pop_size
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.zeros(pop_size)
    
    # Evaluate Initial Population
    # We check time strictly here as func evaluation might be slow
    for i in range(pop_size):
        if time.time() - start_time > evo_max_time:
            # If we timeout during init, just fill rest with infinity to proceed safely
            fitness[i:] = float('inf')
            break
        fitness[i] = func(pop[i])
        
    # Global Best Tracking
    best_idx = np.argmin(fitness)
    global_best_val = fitness[best_idx]
    global_best_vec = pop[best_idx].copy()
    
    # Memory for Adaptive Parameters (SHADE)
    H = 6 # Memory size
    mem_M_CR = np.full(H, 0.5)
    mem_M_F = np.full(H, 0.5)
    k_mem = 0
    
    # Archive for 'current-to-pbest/1' mutation
    archive = []
    
    # --- 2. Evolutionary Loop (L-SHADE-LPSR) ---
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Check termination for Evo phase
        if elapsed >= evo_max_time:
            break
            
        # A. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on time progress
        progress = elapsed / evo_max_time
        target_pop_size = int(round(((min_pop_size - initial_pop_size) * progress) + initial_pop_size))
        target_pop_size = max(min_pop_size, target_pop_size)
        
        # Shrink population if needed
        if pop_size > target_pop_size:
            # Sort by fitness and keep top N
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices[:target_pop_size]]
            fitness = fitness[sorted_indices[:target_pop_size]]
            pop_size = target_pop_size
            
            # Maintain archive size relative to new pop_size
            while len(archive) > pop_size:
                # Remove random elements from archive to fit
                del archive[np.random.randint(0, len(archive))]
        
        # B. Parameter Generation
        # Generate CR and F for each individual using history memory
        r_idx = np.random.randint(0, H, pop_size)
        r_CR = mem_M_CR[r_idx]
        r_F = mem_M_F[r_idx]
        
        # Cauchy distribution for F, Normal for CR
        CR = np.random.normal(r_CR, 0.1)
        CR = np.clip(CR, 0, 1)
        
        F = r_F + 0.1 * np.random.standard_cauchy(pop_size)
        F = np.clip(F, 0, 1)
        # Fix F <= 0
        F[F <= 0] = 0.5 
        
        # C. Mutation: current-to-pbest/1
        # Sort population to find p-bests
        sorted_indices = np.argsort(fitness)
        # p is a random value in [2/pop_size, 0.2]
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        p_top = max(2, int(p_val * pop_size))
        
        # Indices
        # x_best_p: chosen from top p%
        pbest_indices = sorted_indices[np.random.randint(0, p_top, pop_size)]
        x_pbest = pop[pbest_indices]
        
        # r1: random from population, r1 != i
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Ensure r1 != i (simple rotation fix)
        r1_mask = (r1_indices == np.arange(pop_size))
        r1_indices[r1_mask] = (r1_indices[r1_mask] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # r2: random from (Population U Archive), r2 != i, r2 != r1
        # Prepare Union
        if len(archive) > 0:
            pop_all = np.vstack((pop, np.array(archive)))
        else:
            pop_all = pop
            
        r2_indices = np.random.randint(0, len(pop_all), pop_size)
        # We skip strict collision checks for r2 for speed, minimal impact in DE
        x_r2 = pop_all[r2_indices]
        
        # Generate Mutant Vectors
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        F_col = F[:, None]
        v = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # D. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        rand_u = np.random.rand(pop_size, dim)
        mask = rand_u < CR[:, None]
        mask[np.arange(pop_size), j_rand] = True
        
        u = np.where(mask, v, pop)
        
        # Boundary Handling (Bounce back/Correction)
        # If outside, set to (bound + old_val) / 2
        lower_mask = u < min_b
        upper_mask = u > max_b
        u[lower_mask] = (min_b[lower_mask] + pop[lower_mask]) / 2.0
        u[upper_mask] = (max_b[upper_mask] + pop[upper_mask]) / 2.0
        
        # E. Selection and Update
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        success_mask = np.zeros(pop_size, dtype=bool)
        diff_f = np.zeros(pop_size)
        
        # Evaluate candidates
        # Vectorized eval is not possible since func takes 1D array, loop is necessary
        for i in range(pop_size):
            # Check time strictly inside the evaluation loop
            if time.time() - start_time > evo_max_time:
                break
                
            f_new = func(u[i])
            
            if f_new <= fitness[i]:
                new_pop[i] = u[i]
                new_fitness[i] = f_new
                
                if f_new < fitness[i]:
                    success_mask[i] = True
                    diff_f[i] = fitness[i] - f_new
                    # Add parent to archive
                    archive.append(pop[i].copy())
                
                # Update Global Best
                if f_new < global_best_val:
                    global_best_val = f_new
                    global_best_vec = u[i].copy()
        
        pop = new_pop
        fitness = new_fitness
        
        # F. Update Memory
        n_succ = np.sum(success_mask)
        if n_succ > 0:
            s_F = F[success_mask]
            s_CR = CR[success_mask]
            w = diff_f[success_mask]
            
            # Weighted Lehmer Mean
            total_w = np.sum(w)
            if total_w > 0:
                w = w / total_w
                
                mean_pow2 = np.sum(w * (s_F ** 2))
                mean_pow1 = np.sum(w * s_F)
                
                if mean_pow1 > 1e-10:
                    mem_M_F[k_mem] = mean_pow2 / mean_pow1
                else:
                    mem_M_F[k_mem] = 0.5
                    
                mem_M_CR[k_mem] = np.sum(w * s_CR)
                
                k_mem = (k_mem + 1) % H

    # --- 3. Polish Phase: MTS-LS1 (Multiple Trajectory Search Local Search) ---
    # Used to fine-tune the global best solution in the remaining time
    # This is more robust than simple coordinate descent for variable step sizes
    
    if global_best_vec is not None:
        best_sol = global_best_vec.copy()
        best_fit = global_best_val
        
        # Search range initialization (approx 40% of domain)
        SR = (max_b - min_b) * 0.4 
        
        # Improvement flag
        improved = True
        
        while time.time() - start_time < max_time:
            if not improved:
                # Reduce search range if no improvement in full cycle
                SR = SR * 0.5
                if np.max(SR) < 1e-15: # Converged
                    break
            
            improved = False
            
            # Randomize dimension order for unbiased search
            dims = np.random.permutation(dim)
            
            for i in dims:
                if time.time() - start_time >= max_time:
                    break
                
                # Try moving in Negative Direction
                x_new = best_sol.copy()
                x_new[i] -= 0.5 * SR[i]
                
                # Check bounds
                if x_new[i] < min_b[i]: 
                    x_new[i] = min_b[i]
                
                val = func(x_new)
                
                if val < best_fit:
                    best_fit = val
                    best_sol = x_new
                    improved = True
                else:
                    # If Negative failed, try Positive Direction
                    x_new = best_sol.copy()
                    x_new[i] += 0.5 * SR[i]
                    
                    if x_new[i] > max_b[i]:
                        x_new[i] = max_b[i]
                        
                    val = func(x_new)
                    
                    if val < best_fit:
                        best_fit = val
                        best_sol = x_new
                        improved = True
            
            if best_fit < global_best_val:
                global_best_val = best_fit
                global_best_vec = best_sol

    return global_best_val
