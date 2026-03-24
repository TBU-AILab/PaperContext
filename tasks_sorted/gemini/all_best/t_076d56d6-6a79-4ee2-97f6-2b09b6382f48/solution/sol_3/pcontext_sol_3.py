#The output value of the best generated algorithm is: 4.589970908775712
#
#The best generated algorithm code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the function `func` within `max_time` using Restart-SHADE with IPOP.
    
    Algorithm Description:
    This algorithm implements **SHADE (Success-History based Adaptive Differential Evolution)** 
    combined with a **Restart mechanism** and **IPOP (Increasing Population Size)** strategy.
    
    Key Features:
    1.  **Adaptive Parameters**: Maintains a history (memory) of successful `F` (scaling factor) 
        and `CR` (crossover rate) values. New parameters are sampled based on this history, 
        allowing the algorithm to self-tune to the landscape (e.g., learning to be aggressive 
        or conservative).
    2.  **External Archive**: Preserves diversity by storing recently replaced inferior solutions, 
        which are used in the mutation operator `current-to-pbest/1` to prevent premature convergence.
    3.  **Restarts with IPOP**: Detects stagnation (when population variance vanishes) and restarts 
        the search with a larger population (scaled by 1.5x). This allows quick convergence on 
        easy problems while retaining the capacity to explore complex, multimodal landscapes given enough time.
    4.  **Elitism**: The global best solution is injected into every new restart population to ensure 
        monotonic improvement.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff = max_b - min_b
    
    # --- Global Best Tracking ---
    global_best_val = float('inf')
    global_best_vec = None
    
    # --- Restart Configuration ---
    # Start with a moderate population size to allow fast iterations.
    # Base size 20 or 10*dim handles low and high dimensions reasonably.
    base_pop_size = max(20, int(10 * dim))
    restart_count = 0
    
    # --- SHADE Memory Configuration ---
    memory_size = 5
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check time at start of restart
        if datetime.now() - start_time >= time_limit:
            return global_best_val
            
        # IPOP: Scale population size exponentially with restarts
        pop_size = int(base_pop_size * (1.5 ** restart_count))
        
        # Initialize Population (Uniform random)
        # Shape: (pop_size, dim)
        population = min_b + np.random.rand(pop_size, dim) * diff
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize SHADE Memory
        # M_CR and M_F initialized to 0.5, will adapt during evolution
        m_cr = np.full(memory_size, 0.5)
        m_f = np.full(memory_size, 0.5)
        k_mem = 0
        
        # Initialize Archive
        # Stores decent solutions to maintain diversity in mutation
        archive = []
        
        # --- Elitism (Injection) ---
        # Inject the best solution found so far into the new population
        # This ensures we never lose the best known spot.
        start_idx = 0
        if global_best_vec is not None:
            population[0] = global_best_vec
            fitness[0] = global_best_val
            start_idx = 1
            
        # --- Initial Evaluation ---
        for i in range(start_idx, pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = population[i].copy()
                
        # --- Generation Loop ---
        while True:
            # Strict Time Check
            if datetime.now() - start_time >= time_limit:
                return global_best_val
                
            # --- Convergence / Stagnation Detection ---
            # If the population fitness range is negligible, we are stuck in a basin.
            min_fit = np.min(fitness)
            max_fit = np.max(fitness)
            
            if (max_fit - min_fit) < 1e-8:
                break # Break inner loop -> Trigger Restart
                
            # --- Parameter Generation (Vectorized) ---
            # Select random memory index for each individual
            r_idx = np.random.randint(0, memory_size, pop_size)
            
            # Generate CR: Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr[r_idx], 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F: Cauchy(M_F, 0.1)
            # Cauchy random number generator: loc + scale * tan(pi * (U - 0.5))
            u = np.random.rand(pop_size)
            f = m_f[r_idx] + 0.1 * np.tan(np.pi * (u - 0.5))
            
            # Check F constraints: if F <= 0, retry; if F > 1, clip to 1.
            # Retry logic for F <= 0 (must be positive)
            retry_mask = f <= 0
            while np.any(retry_mask):
                u_retry = np.random.rand(np.sum(retry_mask))
                f[retry_mask] = m_f[r_idx][retry_mask] + 0.1 * np.tan(np.pi * (u_retry - 0.5))
                retry_mask = f <= 0
            f = np.clip(f, 0.0, 1.0) # Clip upper bound
            
            # --- Mutation: current-to-pbest/1 with Archive ---
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            
            # 1. Select p-best indices (top 11%)
            sorted_indices = np.argsort(fitness)
            num_pbest = max(2, int(0.11 * pop_size))
            top_indices = sorted_indices[:num_pbest]
            pbest_indices = np.random.choice(top_indices, pop_size)
            
            # 2. Select r1 (from Population)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            
            # 3. Select r2 (from Population U Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                # Concatenate current pop and archive for r2 selection
                candidates_r2 = np.vstack((population, archive_np))
            else:
                candidates_r2 = population
                
            r2_indices = np.random.randint(0, len(candidates_r2), pop_size)
            
            # Gather vectors for mutation equation
            x = population
            x_pbest = population[pbest_indices]
            x_r1 = population[r1_indices]
            x_r2 = candidates_r2[r2_indices]
            
            # Compute Mutant V (Vectorized)
            f_col = f[:, np.newaxis] # Reshape for broadcasting
            mutant = x + f_col * (x_pbest - x) + f_col * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            # Generate crossover mask
            cross_mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            
            # Ensure at least one dimension is taken from mutant (prevent copy of parent)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, x)
            
            # --- Bound Handling ---
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection and Memory Update Preparation ---
            succ_f = []
            succ_cr = []
            diff_fitness = []
            
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return global_best_val
                
                # Evaluation
                f_trial = func(trial_pop[i])
                f_old = fitness[i]
                
                if f_trial < f_old:
                    # Successful trial
                    fitness_imp = f_old - f_trial
                    
                    # Store successful parameters
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fitness.append(fitness_imp)
                    
                    # Add parent to archive before replacement
                    archive.append(population[i].copy())
                    
                    # Update Population
                    population[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    # Update Global Best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_vec = trial_pop[i].copy()
                        
            # --- Archive Maintenance ---
            # Randomly remove solutions if archive grows too large (limit to pop_size)
            while len(archive) > pop_size:
                idx_rem = np.random.randint(0, len(archive))
                archive.pop(idx_rem)
            
            # --- SHADE Memory Update (Weighted Lehmer Mean) ---
            if len(succ_f) > 0:
                sf = np.array(succ_f)
                scr = np.array(succ_cr)
                w_diff = np.array(diff_fitness)
                
                # Normalize weights based on fitness improvement
                w = w_diff / np.sum(w_diff)
                
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(w * scr)
                
                # Weighted Lehmer Mean for F (sum(w*f^2)/sum(w*f))
                mean_f = np.sum(w * (sf ** 2)) / np.sum(w * sf)
                
                # Update Memory (History)
                m_cr[k_mem] = np.clip(mean_cr, 0.0, 1.0)
                m_f[k_mem] = np.clip(mean_f, 0.1, 1.0)
                
                # Advance memory pointer
                k_mem = (k_mem + 1) % memory_size
        
        # Increment restart counter (Trigger population increase)
        restart_count += 1

    return global_best_val
