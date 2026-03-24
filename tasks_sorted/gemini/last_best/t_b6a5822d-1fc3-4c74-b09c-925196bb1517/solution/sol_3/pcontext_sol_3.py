#The following Python code implements the **L-SHADE** (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution) algorithm.
#
#### Improvements Explained:
#1.  **Linear Population Size Reduction (LPSR)**: Unlike the previous implementation that used a fixed or restarted population, this algorithm dynamically reduces the population size linearly with respect to the elapsed time. This allows for maximum exploration in the beginning and focuses on exploitation (fine-tuning) as the time limit approaches, significantly improving convergence speed and accuracy.
#2.  **External Archive**: An archive of inferior solutions is maintained to preserve population diversity. The mutation strategy draws secondary difference vectors from the union of the current population and this archive, preventing premature convergence to local optima.
#3.  **Robust Bound Handling**: Input parameters are strictly clipped to bounds to ensure valid evaluations.
#4.  **Time-Aware Budgeting**: The algorithm calculates "progress" based on `max_time`, seamlessly adapting the population reduction schedule to the provided time constraint without needing a fixed number of generations.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History based Adaptive Differential Evolution).
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Subtract a small buffer to ensure we return the result before the hard limit
    limit_time = timedelta(seconds=max_time - 0.05)

    def is_time_up():
        return (datetime.now() - start_time) >= limit_time

    # --- Initialization ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    diff_b = ub - lb

    # Population Size Parameters (L-SHADE)
    # Start with a larger population for exploration, reduce to min_pop for exploitation
    # Cap max_pop to ensure reasonable iteration speed for higher dimensions
    pop_max = int(np.clip(20 * dim, 50, 300))
    pop_min = 4
    
    current_pop_size = pop_max
    
    # Initialize Population
    pop = lb + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_pos = None

    # Evaluate Initial Population
    for i in range(current_pop_size):
        if is_time_up():
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_pos = pop[i].copy()

    # External Archive: Stores distinct solutions to maintain diversity
    archive = []
    
    # SHADE Memory Parameters
    H = 6  # Memory size
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    mem_k = 0  # Memory index pointer

    # --- Main Loop ---
    while True:
        if is_time_up():
            return best_fitness

        # Calculate Progress (0.0 to 1.0) based on Time
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0

        # 1. Linear Population Size Reduction
        # Linearly decrease population size from pop_max to pop_min
        target_size = int(round(pop_max - (pop_max - pop_min) * progress))
        if target_size < pop_min: target_size = pop_min

        if current_pop_size > target_size:
            # Sort by fitness (ascending, since minimizing)
            sorted_idx = np.argsort(fitness)
            
            # Keep the top 'target_size' individuals
            keep_idx = sorted_idx[:target_size]
            pop = pop[keep_idx]
            fitness = fitness[keep_idx]
            
            # Update current size
            current_pop_size = target_size
            
            # Resize Archive: Archive size cannot exceed Population size
            if len(archive) > current_pop_size:
                # Remove random elements to shrink archive
                num_to_del = len(archive) - current_pop_size
                for _ in range(num_to_del):
                    archive.pop(np.random.randint(0, len(archive)))

        # 2. Adaptive Parameter Generation
        # Select memory index for each individual
        r_idx = np.random.randint(0, H, current_pop_size)
        m_cr_selected = mem_cr[r_idx]
        m_f_selected = mem_f[r_idx]

        # Generate CR (Normal Distribution, clipped [0, 1])
        cr = np.random.normal(m_cr_selected, 0.1)
        cr = np.clip(cr, 0, 1)

        # Generate F (Cauchy Distribution)
        # Cauchy: loc + scale * tan(pi * (rand - 0.5))
        f = m_f_selected + 0.1 * np.tan(np.pi * (np.random.rand(current_pop_size) - 0.5))
        
        # Handle F constraints
        # If F <= 0, regenerate until > 0. If F > 1, clamp to 1.
        bad_f = f <= 0
        while np.any(bad_f):
            count = np.sum(bad_f)
            f[bad_f] = m_f_selected[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(count) - 0.5))
            bad_f = f <= 0
        f = np.minimum(f, 1.0)

        # 3. Mutation and Crossover (current-to-pbest/1/bin)
        sorted_indices = np.argsort(fitness)
        
        # Construct Union Population (Population + Archive) for mutation
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop

        trial_pop = np.zeros_like(pop)

        # Generate Trials
        for i in range(current_pop_size):
            # Select x_pbest from top p% (random p in [2/N, 0.2])
            p_val = np.random.uniform(2.0/current_pop_size, 0.2)
            top_cut = int(max(2, current_pop_size * p_val))
            pbest_idx = sorted_indices[np.random.randint(0, top_cut)]
            x_pbest = pop[pbest_idx]

            # Select x_r1 from Population (distinct from i)
            r1 = np.random.randint(0, current_pop_size)
            while r1 == i:
                r1 = np.random.randint(0, current_pop_size)
            x_r1 = pop[r1]

            # Select x_r2 from Union (distinct from i and r1)
            union_size = len(union_pop)
            r2 = np.random.randint(0, union_size)
            # Simple check to avoid trivial self-selection, though rare in large sets
            while r2 == i or (r2 < current_pop_size and r2 == r1):
                r2 = np.random.randint(0, union_size)
            x_r2 = union_pop[r2]

            # Mutation Vector
            mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (x_r1 - x_r2)

            # Binomial Crossover
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < cr[i]
            mask[j_rand] = True # Ensure at least one dimension is changed
            
            trial = np.where(mask, mutant, pop[i])

            # Bound Constraints (Clipping)
            trial = np.clip(trial, lb, ub)
            trial_pop[i] = trial

        # 4. Selection and Updates
        succ_mask = np.zeros(current_pop_size, dtype=bool)
        diff_fitness = np.zeros(current_pop_size)

        for i in range(current_pop_size):
            if is_time_up(): return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Successful update
                if f_trial < fitness[i]:
                    succ_mask[i] = True
                    diff_fitness[i] = fitness[i] - f_trial
                    
                    # Add parent to archive before replacement
                    if len(archive) < current_pop_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace a random individual in archive
                        if len(archive) > 0:
                            rem_idx = np.random.randint(0, len(archive))
                            archive[rem_idx] = pop[i].copy()
                
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_pos = pop[i].copy()

        # 5. Update History Memory
        if np.any(succ_mask):
            denom = np.sum(diff_fitness[succ_mask])
            if denom > 0:
                # Weighted Lehmer Mean based on improvement magnitude
                weights = diff_fitness[succ_mask] / denom
                
                # Update CR
                succ_cr = cr[succ_mask]
                mean_cr = np.sum(weights * succ_cr)
                
                # Update F
                succ_f = f[succ_mask]
                num_f = np.sum(weights * (succ_f ** 2))
                den_f = np.sum(weights * succ_f)
                mean_f = num_f / den_f if den_f > 0 else 0.5
                
                # Store in Memory
                mem_cr[mem_k] = np.clip(mean_cr, 0, 1)
                mem_f[mem_k] = np.clip(mean_f, 0, 1)
                mem_k = (mem_k + 1) % H

    return best_fitness
