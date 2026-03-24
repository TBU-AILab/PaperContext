#The following Python code implements **L-SHADE (Linear Success-History based Adaptive Differential Evolution)** with time-based population reduction.
#
#**Improvements over the previous algorithm:**
#1.  **Global Search Capability:** Unlike CMA-ES, which is a powerful *local* optimizer that can struggle with highly multimodal landscapes (getting stuck in local basins like the result ~123 suggests), Differential Evolution (DE) is inherently better at global exploration and escaping local optima.
#2.  **Adaptive Parameters:** This algorithm removes the need to manually tune constants ($F$ and $Cr$). It uses a historical memory to "learn" the best mutation and crossover rates for the specific function landscape as the search progresses.
#3.  **Linear Population Reduction (LPSR):** The algorithm starts with a large population to explore the search space broadly. As time passes, it linearly reduces the population size. This forces the algorithm to shift from **Exploration** (searching everywhere) to **Exploitation** (refining the best solutions) exactly as the time limit approaches.
#4.  **Vectorized Operations:** The implementation relies purely on NumPy vectorization for mutation, crossover, and boundary handling, minimizing Python loop overhead and maximizing the number of function evaluations within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Algorithm: L-SHADE (Linear Success-History Adaptive Differential Evolution)
    Adapted for Time-Based Resource Allocation.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper Functions ---
    def get_remaining_seconds():
        return (time_limit - (datetime.now() - start_time)).total_seconds()

    def clip_bounds(vec, lb, ub):
        # Intelligent boundary handling: 
        # If a value exceeds bounds, set it to the bound or midpoint depending on severity
        return np.clip(vec, lb, ub)

    # --- Initialization ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    
    # L-SHADE Parameters
    # Initial population size: standard heuristic is 18 * dim, but capped for efficiency
    # so we don't spend all time in the first generation for high dimensions.
    pop_size_init = min(18 * dim, 500) 
    pop_size_min = 4
    
    # External Archive for maintaining diversity
    # Stores inferior solutions to guide mutation (current-to-pbest/1 w/ archive)
    archive_size = int(pop_size_init * 2.0) 
    archive = []
    
    # Historical Memory for adaptive parameters F (Scale) and Cr (Crossover)
    memory_size = 5
    m_cr = np.full(memory_size, 0.5)
    m_f = np.full(memory_size, 0.5)
    k_mem = 0 # Memory index pointer

    # Initialize Population
    # We use random uniform initialization
    pop = np.random.uniform(lb, ub, (pop_size_init, dim))
    
    # Evaluate Initial Population
    # We must loop because func expects a single 1D array, not a matrix
    fitness = np.zeros(pop_size_init)
    best_idx = 0
    best_val = float('inf')
    
    for i in range(pop_size_init):
        if get_remaining_seconds() <= 0: return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i

    # Current Population Size variable (will decrease over time)
    pop_size = pop_size_init
    
    # --- Main Loop ---
    while True:
        # Time Check
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed >= max_time:
            return best_val
        
        # Calculate Progress (0.0 to 1.0)
        progress = elapsed / max_time
        
        # 1. Linear Population Size Reduction (LPSR) strategy
        # Calculate target population size based on time progress
        plan_pop_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
        plan_pop_size = max(pop_size_min, plan_pop_size)
        
        if pop_size > plan_pop_size:
            # Remove worst individuals to match target size
            sort_indexes = np.argsort(fitness)
            # Keep the best 'plan_pop_size'
            keep_indexes = sort_indexes[:plan_pop_size]
            
            pop = pop[keep_indexes]
            fitness = fitness[keep_indexes]
            pop_size = plan_pop_size
            
            # Reduce archive size dynamically relative to current pop size
            curr_archive_cap = int(pop_size * 2.0)
            if len(archive) > curr_archive_cap:
                # Randomly remove excess from archive
                del_count = len(archive) - curr_archive_cap
                del_indices = np.random.choice(len(archive), del_count, replace=False)
                # List comprehension is faster/safer for list deletion than numpy array manipulation for variable size
                archive = [arr for i, arr in enumerate(archive) if i not in del_indices]

        # 2. Generate Adaptive Parameters (F and Cr)
        # Select random memory index for each individual
        r_idxs = np.random.randint(0, memory_size, pop_size)
        
        # Generate Cr (Normal Distribution around memory)
        mu_cr = m_cr[r_idxs]
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1) # Valid crossover probability
        
        # Generate F (Cauchy Distribution around memory)
        mu_f = m_f[r_idxs]
        # Numpy doesn't have a direct cauchy(loc, scale), use standard_cauchy * scale + loc
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        f = np.clip(f, 0, 1) # Valid scale factor
        
        # If F is effectively zero, regenerate to avoid stagnation
        f[f <= 0] = 0.5 

        # 3. Mutation: current-to-pbest/1
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # Sort population to find pbest (top p percent)
        # p is random in [2/pop_size, 0.2]
        sorted_indices = np.argsort(fitness)
        p_best_rate = np.random.uniform(2.0/pop_size, 0.2)
        p_best_count = max(2, int(p_best_rate * pop_size))
        top_p_indices = sorted_indices[:p_best_count]
        
        # Select pbest for each individual
        pbest_indices = np.random.choice(top_p_indices, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (must be different from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Simple fix for collision: if r1 == i, shift by 1 (wrapping)
        collision = (r1_indices == np.arange(pop_size))
        r1_indices[collision] = (r1_indices[collision] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Union of Pop and Archive)
        # Archive is a list of arrays, convert to array for indexing if not empty
        if len(archive) > 0:
            arr_archive = np.array(archive)
            pop_all = np.vstack((pop, arr_archive))
        else:
            pop_all = pop
            
        r2_indices = np.random.randint(0, len(pop_all), pop_size)
        # Collision handling for r2 is less critical due to pool size, but good practice
        collision_r2 = (r2_indices == np.arange(pop_size)) # Only check vs self (i)
        r2_indices[collision_r2] = (r2_indices[collision_r2] + 1) % len(pop_all)
        x_r2 = pop_all[r2_indices]
        
        # Calculate Mutant Vector V
        # Reshape F for broadcasting: (pop_size, ) -> (pop_size, 1)
        F_col = f[:, None]
        
        # Vectorized mutation equation
        # v = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
        v = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        # Generate mask where rand < Cr
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        
        # Ensure at least one dimension is taken from mutant (j_rand)
        j_rand = np.random.randint(0, dim, pop_size)
        # Use advanced indexing to set specific dimensions to True
        mask[np.arange(pop_size), j_rand] = True
        
        # Create Trial Vectors U
        u = np.where(mask, v, pop)
        
        # Bound Constraints
        u = clip_bounds(u, lb, ub)
        
        # 5. Selection and Memory Update
        new_fitness = np.zeros(pop_size)
        
        # Lists to store successful parameters
        good_cr = []
        good_f = []
        diff_fitness = []
        
        for i in range(pop_size):
            if get_remaining_seconds() <= 0: return best_val
            
            # Evaluate trial
            f_trial = func(u[i])
            new_fitness[i] = f_trial
            
            # Update Global Best immediately
            if f_trial < best_val:
                best_val = f_trial
            
            # Selection Step
            # If trial is better or equal
            if f_trial <= fitness[i]:
                # Improvement
                df = fitness[i] - f_trial
                
                # Add old solution to archive
                archive.append(pop[i].copy())
                
                # Store successful params
                good_cr.append(cr[i])
                good_f.append(f[i])
                diff_fitness.append(df)
                
                # Update population immediately (or deferred, deferred is easier here)
                pop[i] = u[i]
                fitness[i] = f_trial
        
        # Maintain Archive Size limits (random removal if too big)
        curr_archive_cap = int(pop_size * 2.0)
        while len(archive) > curr_archive_cap:
            idx_to_remove = np.random.randint(0, len(archive))
            archive.pop(idx_to_remove)
            
        # 6. Update Historical Memory
        # Weighted Lehmer Mean based on fitness improvement amount
        if len(good_cr) > 0:
            diff_fitness = np.array(diff_fitness)
            good_cr = np.array(good_cr)
            good_f = np.array(good_f)
            
            # Weights
            total_diff = np.sum(diff_fitness)
            if total_diff > 0:
                weights = diff_fitness / total_diff
                
                # Update M_CR (Weighted Mean)
                # If max is 0 (all equal), weights are uniform, handled implies total_diff > 0
                m_cr_new = np.sum(weights * good_cr)
                
                # Update M_F (Lehmer Mean: sum(w*f^2) / sum(w*f))
                sum_wf = np.sum(weights * good_f)
                if sum_wf > 1e-10:
                    m_f_new = np.sum(weights * (good_f**2)) / sum_wf
                else:
                    m_f_new = 0.5
                
                # Update Memory slots (Round Robin)
                m_cr[k_mem] = m_cr_new
                m_f[k_mem] = m_f_new
                k_mem = (k_mem + 1) % memory_size
        
        # --- Check Convergence / Soft Restart ---
        # If all population converged to a single point (variance ~ 0), 
        # but time remains, we should perturb to find other basins.
        if pop_size >= 5:
            std_dev = np.std(fitness)
            rng_fit = np.max(fitness) - np.min(fitness)
            
            # If population is stagnant and we have > 10% time left
            if rng_fit < 1e-8 and progress < 0.9:
                # Keep best
                idx_best = np.argmin(fitness)
                x_best = pop[idx_best].copy()
                f_best_local = fitness[idx_best]
                
                # Re-initialize rest of population with high mutation around best
                # or completely random to explore new areas
                perturb_scale = (ub - lb) * 0.1 # 10% of domain perturbation
                
                for i in range(pop_size):
                    if i == idx_best: continue
                    
                    # Half random, Half perturbed
                    if np.random.rand() < 0.5:
                        pop[i] = np.random.uniform(lb, ub)
                    else:
                        pop[i] = x_best + np.random.uniform(-1, 1, dim) * perturb_scale
                        pop[i] = clip_bounds(pop[i], lb, ub)
                    
                    # We must evaluate these new points
                    if get_remaining_seconds() <= 0: return best_val
                    fitness[i] = func(pop[i])
                
                # Reset Archive to flush stagnation
                archive = []

    return best_val
