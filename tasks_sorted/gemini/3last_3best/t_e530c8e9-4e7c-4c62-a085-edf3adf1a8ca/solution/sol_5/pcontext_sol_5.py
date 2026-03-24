#The output values and codes for the best generated algorithms are as follows:
#1. output value is: 21.279399677312753
#
# 1. algorithm code is:
#The following is a fully operational, self-contained implementation of an improved optimization algorithm.
#
#**Algorithm Explanation:**
#This implementation uses an advanced variant of **Differential Evolution (DE)** with the **"current-to-best/1/bin"** strategy. 
#1.  **Strategy**: Unlike standard `rand/1` (which explores blindly), `current-to-best` directs candidates towards the best solution found so far while maintaining diversity using random difference vectors. This significantly improves convergence speed in time-constrained environments.
#2.  **Adaptive Parameters**: The mutation factor $F$ is "dithered" (randomized) for each candidate to prevent getting stuck in local optima.
#3.  **Restart Mechanism**: If the population converges early (variance becomes negligible), the algorithm triggers a "soft restart." It preserves the best solution found but re-initializes the rest of the population to explore new areas of the search space, maximizing the utility of the available time.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution with 'current-to-best' strategy
    and automatic restarts to handle stagnation.
    """
    start_time = time.time()
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population size: Adapted to dimension but bounded for performance
    # N = 15 * dim is generally robust, clipped to [20, 100] to manage runtime overhead
    pop_size = int(np.clip(dim * 15, 20, 100))
    
    # Initialize population randomly within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    global_best_val = float('inf')
    global_best_idx = -1
    
    # Evaluate initial population
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return global_best_val
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_idx = i

    # --- Hyperparameters ---
    # CR (Crossover Rate): 0.9 promotes convergence by preserving good structures
    # F (Mutation Factor): Randomized between 0.5 and 1.0 (dithering) per vector
    CR = 0.9 
    
    # --- Main Loop ---
    while True:
        # Check time
        if (time.time() - start_time) >= max_time:
            return global_best_val

        # Get the vector of the current best individual
        best_vec = pop[global_best_idx]
        
        # Iterate through population (Asynchronous update)
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            # Select r1, r2 distinct from i
            # Efficient selection logic
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
                
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
            
            # Current vector and random difference vectors
            x_i = pop[i]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Mutation Strategy: current-to-best/1
            # V = X_i + F * (X_best - X_i) + F * (X_r1 - X_r2)
            # This balances exploitation (move to best) and exploration (random difference)
            F = np.random.uniform(0.5, 1.0)
            mutant = x_i + F * (best_vec - x_i) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(dim) < CR
            # Ensure at least one parameter is updated
            cross_mask[np.random.randint(dim)] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # Boundary Constrain (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # Evaluation
            f_trial = func(trial)
            
            # Selection (Greedy)
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best immediately (Asynchronous)
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_idx = i
                    best_vec = trial

        # --- Restart Mechanism ---
        # If population diversity (fitness std dev) is too low, we are stuck.
        # Restart population but keep the best solution.
        if np.std(fitness) < 1e-6:
            # Only restart if we have enough time left (>10% of max_time)
            if (time.time() - start_time) < (max_time * 0.9):
                # Save best
                saved_best = pop[global_best_idx].copy()
                
                # Re-initialize population
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                pop[0] = saved_best # Preserve best at index 0
                
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = global_best_val
                global_best_idx = 0
                
                # Evaluate new random individuals (skip index 0)
                for k in range(1, pop_size):
                    if (time.time() - start_time) >= max_time:
                        return global_best_val
                    
                    val = func(pop[k])
                    fitness[k] = val
                    
                    if val < global_best_val:
                        global_best_val = val
                        global_best_idx = k
    
    return global_best_val
#
#2. output value is: 22.185373409918583
#
# 2. algorithm code is:
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
#
#3. output value is: 31.990037200662613
#
# 3. algorithm code is:
#Here is a fully operational, self-contained implementation of an improved optimization algorithm.
#
#**Algorithm Explanation:**
#This implementation employs a **Self-Adaptive Differential Evolution (jDE)** algorithm enhanced with a **"current-to-pbest"** mutation strategy.
#1.  **Current-to-pbest Mutation**: Instead of moving towards the single best solution (which risks premature convergence), individuals move towards a random solution selected from the top $p\\%$ (p-best) of the population. This strikes a superior balance between exploitation and exploration.
#2.  **Self-Adaptation (jDE)**: The control parameters $F$ (mutation factor) and $CR$ (crossover rate) are not fixed. Each individual carries its own $F$ and $CR$ values, which evolve along with the solution. Successful parameter values are propagated to the next generation, automatically tuning the algorithm to the specific function landscape.
#3.  **Stagnation Detection & Restart**: The algorithm monitors population diversity. If the population converges (fitness variance drops near zero), it triggers a restart mechanism that preserves the global best solution while re-initializing the rest of the population to search new areas, ensuring the entire time budget is utilized effectively.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE)
    with 'current-to-pbest' mutation strategy and automatic restarts.
    """
    start_time = time.time()
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size
    # Adaptive size based on dimension, clamped to [30, 100] to balance
    # exploration capability with computational speed within the time limit.
    pop_size = int(np.clip(dim * 10, 30, 100))
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize jDE Control Parameters (one per individual)
    # F (Mutation): Initialized around 0.5
    # CR (Crossover): Initialized around 0.9
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Strategy Parameter: Top percentage for p-best selection (e.g., 15%)
    p_best_rate = 0.15 
    
    global_best_val = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return global_best_val
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            
    # Sort population by fitness (Indices 0..N are Best..Worst)
    # Sorting is required for efficient p-best selection
    sort_idx = np.argsort(fitness)
    pop = pop[sort_idx]
    fitness = fitness[sort_idx]
    F = F[sort_idx]
    CR = CR[sort_idx]
    
    # --- Main Optimization Loop ---
    while True:
        # Strict time check
        if (time.time() - start_time) >= max_time:
            return global_best_val
            
        # Prepare arrays for the next generation
        next_pop = np.copy(pop)
        next_fitness = np.copy(fitness)
        next_F = np.copy(F)
        next_CR = np.copy(CR)
        
        # Calculate size of the 'p-best' pool
        p_num = max(1, int(pop_size * p_best_rate))
        
        # Iterate over each individual
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            # 1. jDE Parameter Adaptation
            # Update F with probability 0.1
            if np.random.rand() < 0.1:
                # New F in range [0.1, 1.0]
                new_F_val = 0.1 + 0.9 * np.random.rand()
            else:
                new_F_val = F[i]
            
            # Update CR with probability 0.1
            if np.random.rand() < 0.1:
                # New CR in range [0.0, 1.0]
                new_CR_val = np.random.rand()
            else:
                new_CR_val = CR[i]
            
            # 2. Mutation: current-to-pbest/1
            # Vector = Current + F * (P_best - Current) + F * (r1 - r2)
            
            # Select x_pbest randomly from the top p% individuals
            p_idx = np.random.randint(0, p_num)
            x_pbest = pop[p_idx]
            
            # Select r1, r2 distinct from current index i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
                
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
                
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            x_i = pop[i]
            
            # Calculate mutant vector
            # Vectorized operations for speed
            diff_pbest = x_pbest - x_i
            diff_r = x_r1 - x_r2
            mutant = x_i + new_F_val * diff_pbest + new_F_val * diff_r
            
            # 3. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < new_CR_val
            # Guarantee at least one dimension is changed
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # 4. Bound Constraint Handling (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection (Greedy)
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                # Improvement: Accept new vector and new parameters
                next_pop[i] = trial
                next_fitness[i] = f_trial
                next_F[i] = new_F_val
                next_CR[i] = new_CR_val
                
                if f_trial < global_best_val:
                    global_best_val = f_trial
            else:
                # No improvement: Keep old vector and old parameters (implicit in next_* init)
                pass
        
        # Advance generation
        pop = next_pop
        fitness = next_fitness
        F = next_F
        CR = next_CR
        
        # Sort population for next iteration's p-best selection
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        F = F[sort_idx]
        CR = CR[sort_idx]
        
        # --- Restart Mechanism ---
        # If population diversity is extremely low, we are likely stuck in a local optimum.
        # We restart the population but keep the best solution found so far.
        fitness_range = fitness[-1] - fitness[0]
        
        if fitness_range < 1e-7:
            # Only restart if we have at least 10% of time remaining
            if (time.time() - start_time) < (max_time * 0.9):
                # Keep the best individual (at index 0 due to sort)
                # Re-initialize indices 1 to end
                n_reset = pop_size - 1
                pop[1:] = min_b + np.random.rand(n_reset, dim) * diff_b
                
                # Reset Control Parameters for new individuals
                F[1:] = 0.5
                CR[1:] = 0.9
                
                # Evaluate new individuals
                for k in range(1, pop_size):
                    if (time.time() - start_time) >= max_time:
                        return global_best_val
                    
                    val = func(pop[k])
                    fitness[k] = val
                    
                    if val < global_best_val:
                        global_best_val = val
                
                # Re-sort to maintain invariant for next loop
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx]
                fitness = fitness[sort_idx]
                F = F[sort_idx]
                CR = CR[sort_idx]

    return global_best_val
#
#Improvement:
##The following is a fully operational, self-contained implementation of an advanced optimization algorithm.
##
##### Algorithm Explanation
##This algorithm is a robust implementation of **SHADE (Success-History Adaptive Differential Evolution) with Restart Mechanism**.
##
##It integrates the most effective components from the previous top performers:
##1.  **Adaptive Parameters (SHADE)**: Unlike static DE (Code 1) or simple jDE (Code 3), this maintains a historical memory of successful $F$ and $CR$ values, adapting them using a weighted Lehmer mean to fit the optimization landscape dynamically.
##2.  **Restart Mechanism**: Addressing the weakness of the LPSR approach (Code 2) in functions with deceptive local optima, this version monitors population stagnation. If the population converges, it triggers a full restart (preserving the best solution) to explore new basins of attraction.
##3.  **Midpoint-Target Bound Handling**: Instead of simple clipping (which biases search to boundaries), it uses a midpoint/bounce-back strategy, significantly improving performance on bound-constrained problems.
##4.  **Vectorized Implementation**: The core evolution loop is fully vectorized for maximum throughput in Python.
##
##### Python Implementation
##
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History Adaptive Differential Evolution)
    combined with a Restart Mechanism to escape local optima.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Compromise between exploration and speed
    # We clip it to ensure efficient execution within time limits
    pop_size = int(np.clip(15 * dim, 30, 80))
    
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    best_overall_val = float('inf')
    
    # Helper to handle boundaries using midpoint-target strategy
    # This preserves population distribution better than simple clipping
    def apply_bound_constraints(trial_vecs, parent_vecs):
        # Lower violations
        viol_l = trial_vecs < min_b
        if np.any(viol_l):
            # trial = (min + parent) / 2
            # Broadcast min_b to shape of trial
            target = (min_b + parent_vecs) * 0.5
            trial_vecs[viol_l] = target[viol_l]
            
        # Upper violations
        viol_u = trial_vecs > max_b
        if np.any(viol_u):
            target = (max_b + parent_vecs) * 0.5
            trial_vecs[viol_u] = target[viol_u]
            
        return trial_vecs

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check time before starting a new run
        if (time.time() - start_time) >= max_time:
            return best_overall_val

        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_overall_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < best_overall_val:
                best_overall_val = val
        
        # SHADE Memory Init
        memory_size = 5
        M_CR = np.full(memory_size, 0.5)
        M_F = np.full(memory_size, 0.5)
        k_mem = 0
        archive = np.empty((0, dim))
        
        # --- Evolution Loop ---
        while True:
            # Time Check
            if (time.time() - start_time) >= max_time:
                return best_overall_val
            
            # Sort Population (required for current-to-pbest)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Restart Trigger: Stagnation (Variance check)
            # If population has collapsed to a single point, we must restart
            if np.std(fitness) < 1e-9:
                break # Break inner loop -> Trigger Restart
                
            # 1. Parameter Generation
            # Randomly select from memory
            r_indices = np.random.randint(0, memory_size, pop_size)
            m_cr = M_CR[r_indices]
            m_f = M_F[r_indices]
            
            # Generate CR
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # Generate F (Cauchy)
            F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Handle F constraints
            # F > 1 -> 1
            F = np.minimum(F, 1.0)
            # F <= 0 -> Regenerate
            # Fast vectorized regeneration
            while True:
                bad_mask = F <= 0
                if not np.any(bad_mask):
                    break
                F[bad_mask] = m_f[bad_mask] + 0.1 * np.random.standard_cauchy(np.sum(bad_mask))
                F = np.minimum(F, 1.0)

            # 2. Mutation: current-to-pbest/1
            # Select p-best (random p in [2/pop, 0.2])
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            n_best = int(max(2, p_val * pop_size))
            
            # pbest indices
            idx_pbest = np.random.randint(0, n_best, pop_size)
            x_pbest = pop[idx_pbest]
            
            # r1 indices (distinct from i)
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            # Fix collisions r1 == i
            coll_r1 = (idx_r1 == np.arange(pop_size))
            idx_r1[coll_r1] = (idx_r1[coll_r1] + 1) % pop_size
            x_r1 = pop[idx_r1]
            
            # r2 indices (distinct from i, r1; from Union)
            if len(archive) > 0:
                union_pop = np.vstack((pop, archive))
            else:
                union_pop = pop
            
            idx_r2 = np.random.randint(0, len(union_pop), pop_size)
            # (Simplified collision handling for r2 usually sufficient for DE)
            
            # 3. Compute Mutant Vectors
            F_col = F[:, None]
            # Union access
            x_r2 = union_pop[idx_r2]
            
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            cross_rand = np.random.rand(pop_size, dim)
            cross_mask = cross_rand < CR[:, None]
            # Ensure 1 param
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 5. Bound Constraints (Midpoint)
            trial = apply_bound_constraints(trial, pop)
            
            # 6. Evaluation
            trial_fitness = np.zeros(pop_size)
            success_mask = np.zeros(pop_size, dtype=bool)
            fit_diff = np.zeros(pop_size)
            
            for k in range(pop_size):
                if (time.time() - start_time) >= max_time:
                    return best_overall_val
                
                f_t = func(trial[k])
                trial_fitness[k] = f_t
                
                if f_t < fitness[k]:
                    success_mask[k] = True
                    fit_diff[k] = fitness[k] - f_t
                    if f_t < best_overall_val:
                        best_overall_val = f_t
            
            # 7. Update Archive & Pop
            n_succ = np.sum(success_mask)
            if n_succ > 0:
                # Add parents to archive
                new_arch = pop[success_mask]
                if len(archive) == 0:
                    archive = new_arch
                else:
                    archive = np.vstack((archive, new_arch))
                
                # Trim archive
                if len(archive) > pop_size:
                    n_del = len(archive) - pop_size
                    del_idx = np.random.choice(len(archive), n_del, replace=False)
                    archive = np.delete(archive, del_idx, axis=0)
                    
                # Update Pop
                pop[success_mask] = trial[success_mask]
                fitness[success_mask] = trial_fitness[success_mask]
                
                # 8. Update Memory (Weighted Lehmer Mean)
                succ_F = F[success_mask]
                succ_CR = CR[success_mask]
                succ_diff = fit_diff[success_mask]
                
                # Weights
                weights = succ_diff / np.sum(succ_diff)
                
                # Update M_CR (Weighted Mean)
                if np.max(succ_CR) > 0:
                    mean_cr = np.sum(weights * succ_CR)
                    M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
                
                # Update M_F (Lehmer Mean)
                num_f = np.sum(weights * (succ_F ** 2))
                den_f = np.sum(weights * succ_F)
                if den_f > 0:
                    mean_f = num_f / den_f
                    M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                    
                k_mem = (k_mem + 1) % memory_size
    
    return best_overall_val
