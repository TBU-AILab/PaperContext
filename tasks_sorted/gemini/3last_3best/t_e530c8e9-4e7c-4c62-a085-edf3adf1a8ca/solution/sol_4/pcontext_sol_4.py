#The following is a fully operational, self-contained implementation of an advanced optimization algorithm.
#
#### Algorithm Explanation
#This algorithm implements **L-SHADE (Linear Success-History Adaptive Differential Evolution)**, which is widely considered one of the most powerful improvements over standard Differential Evolution. It significantly outperforms the previous algorithms (DE and jDE) by addressing their parameter tuning and convergence speed issues.
#
#Key improvements included in this implementation:
#1.  **Success-History Parameter Adaptation**: Instead of random or fixed parameters, it maintains a memory ($M_{CR}, M_F$) of successful $F$ (mutation) and $CR$ (crossover) values. It learns from the optimization process to generate parameters that work best for the specific landscape.
#2.  **Linear Population Size Reduction (LPSR)**: The algorithm starts with a large population to explore the search space and linearly reduces the population size as time progresses. This ensures maximum exploration early on and maximum exploitation (refinement) towards the deadline.
#3.  **External Archive**: It maintains an archive of inferior solutions recently replaced by better ones. These are used in the mutation operator `current-to-pbest/1` to preserve diversity and prevent premature convergence.
#4.  **Robust Restart**: If the population collapses (converges) before the time limit, it saves the best solution and performs a "soft restart" to hunt for other potential global minima.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Success-History Adaptive Differential Evolution)
    with Linear Population Size Reduction and a robust restart mechanism.
    """
    start_time = time.time()
    
    # --- Helper Functions ---
    def trim_archive(arch, target_size):
        if len(arch) > target_size:
            # Remove random elements to fit size
            idxs = np.random.choice(len(arch), len(arch) - target_size, replace=False)
            return np.delete(arch, idxs, axis=0)
        return arch

    def get_cauchy(loc, scale, size):
        # Generate Cauchy distributed values: loc + scale * standard_cauchy
        return loc + scale * np.random.standard_cauchy(size)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initial Population Size (N_init)
    # Start strictly larger for exploration, but cap for performance
    N_init = int(np.clip(dim * 18, 50, 200))
    # Final Population Size (N_min)
    N_min = 4
    
    current_pop_size = N_init
    
    # Initialize Population
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    # Evaluate initial population
    global_best_val = float('inf')
    global_best_vec = np.zeros(dim)
    
    for i in range(current_pop_size):
        if (time.time() - start_time) >= max_time:
            return global_best_val if global_best_val != float('inf') else 0.0 # Fallback
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_vec = pop[i].copy()

    # --- SHADE Memory Initialization ---
    memory_size = 5
    M_CR = np.full(memory_size, 0.5)
    M_F = np.full(memory_size, 0.5)
    k_mem = 0 # Memory index pointer
    
    # Archive for mutated vectors (maintains diversity)
    archive = np.empty((0, dim))
    
    # --- Main Loop ---
    while True:
        # Time check
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed >= max_time:
            return global_best_val
            
        # 1. Linear Population Size Reduction (LPSR) based on Time
        # Calculate allowed generations is hard, so we use time ratio
        # Only reduce if we aren't already at min
        if current_pop_size > N_min:
            # Estimate reduction based on time progress
            time_ratio = elapsed / max_time
            plan_size = int(round((N_min - N_init) * time_ratio + N_init))
            
            if current_pop_size > plan_size:
                # Reduction needed: Remove worst individuals
                sort_indices = np.argsort(fitness)
                n_remove = current_pop_size - max(N_min, plan_size)
                
                # Keep best, discard worst
                keep_indices = sort_indices[:current_pop_size - n_remove]
                
                pop = pop[keep_indices]
                fitness = fitness[keep_indices]
                current_pop_size = len(pop)
                
                # Resize archive to fit new pop size
                archive = trim_archive(archive, current_pop_size)

        # Sort population (needed for p-best selection)
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        
        # 2. Parameter Generation
        # Generate CR and F for each individual based on memory
        # Randomly select memory index for each individual
        r_idxs = np.random.randint(0, memory_size, current_pop_size)
        
        # CR: Normal distribution, clipped [0, 1]
        m_cr_selected = M_CR[r_idxs]
        CR = np.random.normal(m_cr_selected, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        # In SHADE, if CR is close to 0, it is often folded to 0, but clip is fine.
        
        # F: Cauchy distribution, clipped [0, 1] (resampled if <= 0)
        m_f_selected = M_F[r_idxs]
        F = get_cauchy(m_f_selected, 0.1, current_pop_size)
        
        # Handle F constraints (F > 1 -> 1, F <= 0 -> regenerate)
        # Vectorized regeneration for F <= 0
        bad_f = F <= 0
        while np.any(bad_f):
            F[bad_f] = get_cauchy(m_f_selected[bad_f], 0.1, np.sum(bad_f))
            bad_f = F <= 0
        F = np.clip(F, 0.0, 1.0) # Clip upper bound to 1.0

        # 3. Mutation: current-to-pbest/1
        # v = x + F(x_pbest - x) + F(x_r1 - x_r2)
        # x_r2 is selected from Union(Population, Archive)
        
        # p-best selection: top p% (randomized p in [2/N, 0.2])
        p_min = 2.0 / current_pop_size
        p_i = np.random.uniform(p_min, 0.2, current_pop_size)
        p_best_indices = (p_i * current_pop_size).astype(int)
        p_best_indices = np.clip(p_best_indices, 0, current_pop_size - 1)
        
        # Create vectors
        x_pbest = pop[p_best_indices] # Sorted pop, so index i is i-th best
        x_curr = pop # Aligned
        
        # Select r1 (distinct from current)
        r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
        # Fix r1 == i collisions
        collisions = (r1_indices == np.arange(current_pop_size))
        r1_indices[collisions] = (r1_indices[collisions] + 1) % current_pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Union(Pop, Archive))
        union_pop = pop
        if len(archive) > 0:
            union_pop = np.vstack((pop, archive))
            
        r2_indices = np.random.randint(0, len(union_pop), current_pop_size)
        # Fix r2 collisions (simplified check against r1 and i not strictly enforced for speed, 
        # DE is robust to minor overlaps in r2)
        x_r2 = union_pop[r2_indices]
        
        # Calculate Mutation Vectors
        # Expand F for broadcasting
        F_col = F[:, None]
        mutant = x_curr + F_col * (x_pbest - x_curr) + F_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        cross_mask = np.random.rand(current_pop_size, dim) < CR[:, None]
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, current_pop_size)
        cross_mask[np.arange(current_pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        trial = np.clip(trial, min_b, max_b)
        
        # 5. Selection & Memory Update Preparations
        success_mask = np.zeros(current_pop_size, dtype=bool)
        diff_fitness = np.zeros(current_pop_size)
        
        # Evaluate trials
        # We can't vectorize func call easily, so loop
        trial_fitness = np.empty(current_pop_size)
        
        for i in range(current_pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            f_trial = func(trial[i])
            trial_fitness[i] = f_trial
            
            if f_trial < fitness[i]:
                success_mask[i] = True
                diff_fitness[i] = fitness[i] - f_trial
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trial[i].copy()
            else:
                success_mask[i] = False

        # 6. Update Population and Archive
        # Identify successful indices
        s_idxs = np.where(success_mask)[0]
        
        if len(s_idxs) > 0:
            # Add replaced individuals to archive
            # Candidates to enter archive are pop[s_idxs] (the old parents)
            candidates = pop[s_idxs]
            
            # Add to archive
            if len(archive) == 0:
                archive = candidates
            else:
                archive = np.vstack((archive, candidates))
            
            # Trim archive to max size (pop_size)
            archive = trim_archive(archive, current_pop_size)
            
            # Update population
            pop[s_idxs] = trial[s_idxs]
            fitness[s_idxs] = trial_fitness[s_idxs]
            
            # 7. Update Memory (Weighted Lehmer Mean)
            # Collect successful F and CR
            S_F = F[s_idxs]
            S_CR = CR[s_idxs]
            df = diff_fitness[s_idxs]
            
            # Calculate weights based on fitness improvement
            total_df = np.sum(df)
            if total_df > 0:
                weights = df / total_df
                
                # Update M_CR (Weighted Mean)
                if np.max(S_CR) > 0: # Avoid div by zero
                    mean_scr = np.sum(weights * S_CR)
                    M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_scr
                
                # Update M_F (Weighted Lehmer Mean)
                mean_sf = np.sum(weights * (S_F ** 2)) / np.sum(weights * S_F)
                M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_sf
                
                # Increment memory pointer
                k_mem = (k_mem + 1) % memory_size

        # 8. Stagnation Check / Restart
        # If population diversity is lost, restart but keep best.
        # Check std deviation of fitness
        pop_std = np.std(fitness)
        if pop_std < 1e-8:
            # Soft Restart
            # Only if we have reasonable time left (e.g. > 5%)
            if (time.time() - start_time) < (max_time * 0.95):
                # Preserve best
                best_idx = np.argmin(fitness)
                saved_best = pop[best_idx].copy()
                saved_best_val = fitness[best_idx]
                
                # Re-initialize population around bounds
                pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
                
                # Put best back
                pop[0] = saved_best
                fitness[:] = float('inf')
                fitness[0] = saved_best_val
                
                # Reset Archive and Memory
                archive = np.empty((0, dim))
                M_CR[:] = 0.5
                M_F[:] = 0.5
                
                # Re-evaluate new pop (skipping 0)
                for k in range(1, current_pop_size):
                    if (time.time() - start_time) >= max_time:
                        return global_best_val
                    val = func(pop[k])
                    fitness[k] = val
                    if val < global_best_val:
                        global_best_val = val

    return global_best_val
