#Here is a fully operational implementation using **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution)**.
#
#### Algorithm Explanation
#1.  **L-SHADE Architecture**: This is an enhancement of the SHADE algorithm (which adapts parameters $F$ and $CR$ based on history). L-SHADE is a consistent winner in evolutionary computation competitions (CEC) for single-objective optimization.
#2.  **Linear Population Size Reduction (LPSR)**: Unlike the previous algorithms that used a fixed population size or hard restarts, this algorithm starts with a large population to explore the global space. As time progresses, it linearly reduces the population size. This forces the algorithm to shift from **Exploration** (searching everywhere) to **Exploitation** (refining the best solution) exactly as the time limit approaches.
#3.  **Time-Based Schedule**: Standard L-SHADE uses "Max Evaluations" to schedule the population reduction. Since we are given "Max Time", this implementation calculates the reduction schedule dynamically based on the `elapsed_time / max_time` ratio.
#4.  **External Archive**: It maintains a history of inferior solutions that were recently replaced. This allows the differential mutation operator to learn from "bad" directions as well as good ones, maintaining diversity without extra evaluations.
#
#### Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History Adaptive Differential Evolution).
    """
    start_time = datetime.now()
    # Safety buffer: stop slightly before max_time to ensure return
    time_limit = timedelta(seconds=max_time * 0.98)
    
    # --- Helper Functions ---
    def get_elapsed_ratio():
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / (max_time * 0.98), 1.0)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initial Population Size (N_init)
    # A larger initial population (e.g., 18*dim) improves global search capability.
    pop_size_init = int(round(18 * dim))
    # Minimum Population Size (N_min) - usually 4 is the minimum for mutation strategies
    pop_size_min = 4
    
    current_pop_size = pop_size_init
    
    # Initialize Population
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    # Evaluate Initial Population
    best_fitness = float('inf')
    best_solution = None
    
    for i in range(current_pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_fitness if best_solution is not None else float('inf')
        
        try:
            val = func(pop[i])
        except:
            val = float('inf')
            
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_solution = pop[i].copy()

    # --- L-SHADE Memory Initialization ---
    # H: Memory size for historical success parameters
    H = 6
    mem_M_F = np.full(H, 0.5) # Mean F memory
    mem_M_CR = np.full(H, 0.5) # Mean CR memory
    k_mem = 0 # Memory index pointer
    
    # External Archive (stores replaced individuals to maintain diversity)
    archive = []
    archive_size_rate = 2.6
    
    # --- Main Loop ---
    while True:
        # Check Time
        if datetime.now() - start_time >= time_limit:
            return best_fitness

        # 1. Population Size Reduction (LPSR)
        # Calculate target population size based on time elapsed
        time_ratio = get_elapsed_ratio()
        
        plan_pop_size = int(round(((pop_size_min - pop_size_init) * time_ratio) + pop_size_init))
        plan_pop_size = max(pop_size_min, plan_pop_size)

        # Reduce population if needed
        if current_pop_size > plan_pop_size:
            # Sort by fitness (descending badness) and remove worst
            # We want to keep the indices with lowest fitness
            sort_indexes = np.argsort(fitness)
            keep_indexes = sort_indexes[:plan_pop_size]
            
            pop = pop[keep_indexes]
            fitness = fitness[keep_indexes]
            current_pop_size = plan_pop_size
            
            # Resize archive if it exceeds new limit
            max_archive_size = int(current_pop_size * archive_size_rate)
            if len(archive) > max_archive_size:
                # Randomly remove elements to fit
                del_count = len(archive) - max_archive_size
                # Simple truncation or random removal
                archive = archive[:max_archive_size]

        # 2. Parameter Adaptation (F and CR generation)
        # Pick random memory index for each individual
        r_idxs = np.random.randint(0, H, current_pop_size)
        m_f = mem_M_F[r_idxs]
        m_cr = mem_M_CR[r_idxs]

        # Generate CR: Normal distribution, clipped [0, 1]
        # If memory is near -1 (terminal), fix to 0
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F: Cauchy distribution
        # F must be > 0. If > 1, clip to 1. 
        # Detailed logic: if F <= 0, regenerate.
        f = m_f + 0.1 * np.random.standard_cauchy(current_pop_size)
        
        # Repair F
        retry_limit = 0
        while retry_limit < 10:
            bad_f_idx = f <= 0
            if not np.any(bad_f_idx):
                break
            # Regenerate bad ones
            f[bad_f_idx] = m_f[bad_f_idx] + 0.1 * np.random.standard_cauchy(np.sum(bad_f_idx))
            retry_limit += 1
        
        f[f <= 0] = 0.5 # Fallback
        f = np.minimum(f, 1.0)

        # 3. Mutation: current-to-pbest/1
        # p_best rate depends on population size, typically top 5%-20%
        p_val = max(2, int(current_pop_size * 0.11)) 
        sorted_indices = np.argsort(fitness)
        top_indices = sorted_indices[:p_val]
        
        # Vectors:
        # x_i: current pop
        # x_pbest: randomly selected from top p%
        # x_r1: random from pop, != i
        # x_r2: random from Union(pop, archive), != i, != r1
        
        pbest_idxs = np.random.choice(top_indices, current_pop_size)
        x_pbest = pop[pbest_idxs]
        
        # R1 selection
        idxs = np.arange(current_pop_size)
        r1 = np.random.randint(0, current_pop_size, current_pop_size)
        # Fix collisions r1 == i
        r1 = np.where(r1 == idxs, (r1 + 1) % current_pop_size, r1)
        x_r1 = pop[r1]
        
        # R2 selection (Union of Pop and Archive)
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        union_size = len(union_pop)
        r2 = np.random.randint(0, union_size, current_pop_size)
        
        # Fix collisions r2 == i or r2 == r1
        # Note: r2 index could be >= current_pop_size, so only check if < current_pop_size
        r2_conflict = (r2 == idxs) | (r2 == r1)
        if np.any(r2_conflict):
            # Simple shift strategy
            r2[r2_conflict] = (r2[r2_conflict] + 1) % union_size
            
        x_r2 = union_pop[r2]
        
        # Compute Mutant V = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
        f_col = f[:, np.newaxis]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # Bound Constraint (Clipping)
        mutant = np.clip(mutant, min_b, max_b)
        
        # 4. Crossover (Binomial)
        # Random dimension to ensure at least one parameter changes
        j_rand = np.random.randint(0, dim, current_pop_size)
        j_mask = np.zeros((current_pop_size, dim), dtype=bool)
        j_mask[np.arange(current_pop_size), j_rand] = True
        
        cross_mask = np.random.rand(current_pop_size, dim) < cr[:, np.newaxis]
        trial_mask = cross_mask | j_mask
        
        trial_pop = np.where(trial_mask, mutant, pop)
        
        # 5. Selection and Evaluation
        succ_f = []
        succ_cr = []
        diff_fitness = []
        
        max_archive_size = int(current_pop_size * archive_size_rate)

        for i in range(current_pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_fitness
            
            try:
                f_trial = func(trial_pop[i])
            except:
                f_trial = float('inf')
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                # Improvement or equal
                if f_trial < fitness[i]:
                    # Record success parameters
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fitness.append(fitness[i] - f_trial)
                    
                    # Update Archive with replaced parent
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive_size:
                        # Randomly remove
                        rm_idx = np.random.randint(0, len(archive))
                        archive.pop(rm_idx)
                
                # Update Population
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_solution = trial_pop[i].copy()

        # 6. Memory Update (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            succ_f = np.array(succ_f)
            succ_cr = np.array(succ_cr)
            diff_fitness = np.array(diff_fitness)
            
            total_diff = np.sum(diff_fitness)
            if total_diff > 0:
                weights = diff_fitness / total_diff
                
                # Update Mean F (Lehmer Mean)
                sum_sq_f = np.sum(weights * (succ_f ** 2))
                sum_f = np.sum(weights * succ_f)
                if sum_f > 0:
                    mean_f_new = sum_sq_f / sum_f
                    mem_M_F[k_mem] = 0.5 * mem_M_F[k_mem] + 0.5 * mean_f_new
                    
                # Update Mean CR (Weighted Mean)
                mean_cr_new = np.sum(weights * succ_cr)
                # If mean_cr is degenerate (-1 in some implementations), handle or clamp
                if mean_cr_new < 0: mean_cr_new = 0
                mem_M_CR[k_mem] = 0.5 * mem_M_CR[k_mem] + 0.5 * mean_cr_new
                
                k_mem = (k_mem + 1) % H

    return best_fitness
