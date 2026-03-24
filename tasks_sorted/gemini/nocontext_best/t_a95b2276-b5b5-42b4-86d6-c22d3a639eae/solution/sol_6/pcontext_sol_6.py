#The following is an implementation of **L-SHADE** (Linear Success-History based Adaptive Differential Evolution with Linear Population Size Reduction).
#
#### Improvements over the previous version:
#1.  **Linear Population Size Reduction (LPSR):** The algorithm dynamically reduces the population size as time progresses. It starts with a large population to explore the global space and shrinks it to focus computational resources on exploiting the best area found as the deadline approaches.
#2.  **External Archive:** It maintains an archive of inferior solutions that were recently replaced. This allows the mutation strategy to draw from a wider pool of difference vectors, preserving diversity and preventing premature convergence.
#3.  **Historical Memory:** Instead of adapting a single mean for parameters $F$ and $CR$, it uses a history memory of successful settings. This allows the algorithm to recall distributions that worked well in the past (handling multimodal landscapes better).
#4.  **Fully Vectorized Operations:** Candidate generation (mutation and crossover) is performed using NumPy matrix operations rather than Python loops, significantly reducing overhead and leaving more time for actual function evaluations.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Success-History based Adaptive 
    Differential Evolution with Linear Population Size Reduction).
    """
    start_time = time.time()
    
    # --- Configuration ---
    # L-SHADE specific parameters
    r_N_init = 18       # Initial population size multiplier
    r_arc = 1.4         # Archive size multiplier
    p_best_rate = 0.11  # Top p-best selection rate
    memory_size = 5     # Size of historical memory
    
    # Initialize Population Size
    pop_size = int(max(30, r_N_init * dim))
    max_pop_size = pop_size
    min_pop_size = 4
    
    # Initialize Memory for F and CR
    m_cr = np.full(memory_size, 0.5)
    m_f = np.full(memory_size, 0.5)
    k_mem = 0  # Memory index
    
    # Archive
    archive = []
    
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # 1. Initialization
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Track global best
    best_idx = 0
    global_best_val = float('inf')
    global_best_vec = None

    # Initial Evaluation
    # Note: We check time inside the loop strictly
    for i in range(pop_size):
        if (time.time() - start_time) > max_time:
            # If we time out during init, return best found so far
            return global_best_val if global_best_vec is not None else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_vec = population[i].copy()
            best_idx = i

    # Variables for time management
    # We estimate remaining generations based on average generation time
    elapsed = time.time() - start_time
    # Initial estimate of how much time we have used vs total budget
    # to drive the Population Size Reduction
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        if elapsed >= max_time:
            break
            
        # Calculate time progress (0.0 to 1.0) for Linear Population Reduction
        # We clamp it to just under 1.0 to avoid math errors
        progress = min(0.9999, elapsed / max_time)
        
        # --- Linear Population Size Reduction (LPSR) ---
        new_pop_size = int(round((min_pop_size - max_pop_size) * progress + max_pop_size))
        
        if new_pop_size < pop_size:
            # Sort by fitness and truncate weakest
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices[:new_pop_size]]
            fitness = fitness[sorted_indices[:new_pop_size]]
            
            # Resize archive if necessary
            current_arc_size = len(archive)
            target_arc_size = int(new_pop_size * r_arc)
            if current_arc_size > target_arc_size:
                # Randomly remove elements to fit
                del_indices = np.random.choice(current_arc_size, current_arc_size - target_arc_size, replace=False)
                # List comprehension is safer for list deletion with numpy indices
                archive = [x for i, x in enumerate(archive) if i not in del_indices]

            pop_size = new_pop_size
            
        # --- Parameter Generation ---
        # Generate CR and F for each individual based on memory
        # Randomly select memory index for each individual
        r_idx = np.random.randint(0, memory_size, pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr_g = np.random.normal(mu_cr, 0.1)
        cr_g = np.clip(cr_g, 0, 1)
        # In SHADE, if CR is close to 0, we sometimes force it, but here clip is fine.
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        # If F <= 0, regenerate. If F > 1, clip to 1.
        f_g = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        
        # Repair F
        while True:
            bad_f_indices = np.where(f_g <= 0)[0]
            if len(bad_f_indices) == 0:
                break
            # Regenerate only bad ones
            r_idx_bad = np.random.randint(0, memory_size, len(bad_f_indices))
            mu_f_bad = m_f[r_idx_bad]
            f_g[bad_f_indices] = np.random.standard_cauchy(len(bad_f_indices)) * 0.1 + mu_f_bad
            
        f_g = np.minimum(f_g, 1.0)
        
        # --- Mutation Strategy: current-to-pbest/1 ---
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        
        # 1. Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        # We need original indices to map back to x_i
        # But since we use vectorized operations on the whole arrays, 
        # let's just create pbest pointers.
        
        # Number of top individuals to select from
        num_pbest = max(int(pop_size * p_best_rate), 2)
        pbest_indices = np.random.randint(0, num_pbest, pop_size)
        # Actual indices in the current (potentially unsorted) arrays? 
        # Easier to work with sorted copies for pbest selection logic, 
        # but we need to mutate the actual population 'x'.
        # Solution: Use sorted indices map.
        pbest_ptr = sorted_indices[pbest_indices]
        
        # 2. r1 selection: Random distinct from i
        r1_ptr = np.random.randint(0, pop_size, pop_size)
        # Simple fix for collisions: Since pop is random, collisions are rare. 
        # We ignore strict i != r1 != r2 checks for pure speed in vectorized Python, 
        # or do a simple single-pass cyclic shift if they match.
        hit_self = np.where(r1_ptr == np.arange(pop_size))[0]
        if len(hit_self) > 0:
            r1_ptr[hit_self] = (r1_ptr[hit_self] + 1) % pop_size
            
        # 3. r2 selection: From Union(Population, Archive)
        # We construct the union array virtually or physically
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((population, archive_np))
        else:
            union_pop = population
            
        union_size = len(union_pop)
        r2_ptr = np.random.randint(0, union_size, pop_size)
        
        # Ensure r2 != r1 and r2 != i
        # (Simplified collision handling for speed)
        invalid = np.where((r2_ptr == np.arange(pop_size)) | (r2_ptr == r1_ptr))[0]
        if len(invalid) > 0:
            r2_ptr[invalid] = (r2_ptr[invalid] + 1) % union_size
            
        # --- Create Trial Vectors (Vectorized) ---
        x_i = population
        x_pbest = population[pbest_ptr]
        x_r1 = population[r1_ptr]
        x_r2 = union_pop[r2_ptr]
        
        # Expand dimensions for F to allow broadcasting
        F_col = f_g[:, np.newaxis]
        
        # Mutation equation
        mutant = x_i + F_col * (x_pbest - x_i) + F_col * (x_r1 - x_r2)
        
        # Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr_g[:, np.newaxis]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        rows = np.arange(pop_size)
        cross_mask[rows, j_rand] = True
        
        trials = np.where(cross_mask, mutant, x_i)
        
        # Bound Constraints (Clip)
        trials = np.clip(trials, min_b, max_b)
        
        # --- Selection & Memory Update ---
        trial_fitness = np.zeros(pop_size)
        
        successful_f = []
        successful_cr = []
        improvement_diff = []
        
        # Evaluate trials
        for i in range(pop_size):
            # Check time strictly every step (or every few steps)
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            f_trial = func(trials[i])
            trial_fitness[i] = f_trial
            
            if f_trial <= fitness[i]:
                # Improvement or neutral: Selection happens later effectively
                # We store success data
                if f_trial < fitness[i]:
                    successful_cr.append(cr_g[i])
                    successful_f.append(f_g[i])
                    improvement_diff.append(fitness[i] - f_trial)
                    
                    # Update Archive
                    # Add replaced parent to archive
                    archive.append(population[i].copy())
                
                # Update Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trials[i].copy()
            
        # Update Population and Fitness based on success
        # (This is the selection step vectorized)
        mask_better = trial_fitness < fitness
        # Update fitness where trial is better
        fitness = np.where(mask_better, trial_fitness, fitness)
        # Update population where trial is better
        # numpy indexing requires care: population[mask] = trials[mask] works for 1D, 
        # for 2D we need to ensure shapes align or use where
        # trials is (pop, dim), mask is (pop)
        population = np.where(mask_better[:, np.newaxis], trials, population)
        
        # Trim archive if it exceeds size
        max_arc_size = int(pop_size * r_arc)
        if len(archive) > max_arc_size:
            # Randomly remove excess
            num_remove = len(archive) - max_arc_size
            indices_to_remove = np.random.choice(len(archive), num_remove, replace=False)
            # Create new list excluding removed indices
            archive = [x for k, x in enumerate(archive) if k not in indices_to_remove]

        # --- Update Memory ---
        if len(successful_cr) > 0:
            s_cr = np.array(successful_cr)
            s_f = np.array(successful_f)
            diffs = np.array(improvement_diff)
            
            # Weighted Lehmer Mean calculation
            # Weights based on improvement amount
            total_diff = np.sum(diffs)
            if total_diff > 0:
                weights = diffs / total_diff
            else:
                weights = np.ones(len(diffs)) / len(diffs)
            
            # Update M_CR (Weighted Mean)
            # If max(m_cr) is undefined or empty, we handle errors, but arrays are valid here
            m_cr_new = np.sum(weights * s_cr)
            # Clamping to valid range is good practice
            m_cr[k_mem] = np.clip(m_cr_new, 0, 1)
            
            # Update M_F (Weighted Lehmer Mean)
            # mean = sum(w * f^2) / sum(w * f)
            mean_lehmer = np.sum(weights * (s_f ** 2)) / (np.sum(weights * s_f) + 1e-10)
            m_f[k_mem] = np.clip(mean_lehmer, 0, 1)
            
            k_mem = (k_mem + 1) % memory_size
            
    return global_best_val
