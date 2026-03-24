#The output value of the best generated algorithm is: 4.589970908775712
#
#The best generated algorithm code:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the function `func` using Restart-based SHADE with IPOP.
    
    Algorithm Description:
    1.  **SHADE (Success-History based Adaptive Differential Evolution)**: Adapts F and CR 
        parameters using a historical memory to learn the best strategy for the specific 
        function landscape.
    2.  **Warm-Start Parameters**: Memory is initialized with F=0.5 and CR=0.9, which were 
        empirically found to be robust starting points in previous iterations.
    3.  **External Archive**: Maintains diversity by storing inferior solutions replaced 
        during selection, used in the `current-to-pbest/1` mutation strategy.
    4.  **IPOP (Increasing Population Size)**: Restarts the optimization with exponentially 
        growing population size upon convergence (stagnation) to escape local optima.
    5.  **Vectorized Operations**: Maximizes efficiency of mutation/crossover steps to 
        allocate more time for function evaluations.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- Global Best Tracking ---
    best_val = float('inf')
    best_vec = None
    
    # --- IPOP Configuration ---
    # Robust initial population size based on dimension
    base_pop = 20 + 5 * dim 
    restart_count = 0
    
    # --- SHADE Configuration ---
    memory_size = 5

    # --- Main Restart Loop ---
    while True:
        # Check time before committing to a new restart
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        # IPOP: Scale population size exponentially (1.5x per restart)
        pop_size = int(base_pop * (1.5 ** restart_count))
        
        # --- Initialization ---
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize SHADE Memory
        # Warm start: CR=0.9 (aggressive crossover), F=0.5 (balanced mutation)
        m_cr = np.full(memory_size, 0.9)
        m_f = np.full(memory_size, 0.5)
        k_mem = 0
        
        # Initialize Archive (Capacity = pop_size)
        archive = np.empty((pop_size, dim))
        arc_count = 0
        
        # Elitism: Inject global best into the new population
        start_idx = 0
        if best_vec is not None:
            pop[0] = best_vec.copy()
            fitness[0] = best_val
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_vec = pop[i].copy()
                
        # --- Evolution Loop ---
        while True:
            # Strict time check
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            # --- Convergence Detection ---
            # If population fitness variance is negligible, trigger restart
            if np.max(fitness) - np.min(fitness) < 1e-8:
                break
                
            # --- 1. Parameter Generation (Vectorized) ---
            # Randomly select memory indices for each individual
            r_idx = np.random.randint(0, memory_size, pop_size)
            mf_r = m_f[r_idx]
            mcr_r = m_cr[r_idx]
            
            # Generate CR: Normal(M_CR, 0.1), clipped [0, 1]
            cr = np.random.normal(mcr_r, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F: Cauchy(M_F, 0.1)
            # Cauchy = loc + scale * tan(pi * (uniform - 0.5))
            u = np.random.rand(pop_size)
            f = mf_r + 0.1 * np.tan(np.pi * (u - 0.5))
            
            # Handle F constraints: F > 0 (retry), F <= 1 (clip)
            retry_mask = f <= 0
            while np.any(retry_mask):
                cnt = np.sum(retry_mask)
                # Retry generating F for invalid indices
                f[retry_mask] = mf_r[retry_mask] + 0.1 * np.tan(np.pi * (np.random.rand(cnt) - 0.5))
                retry_mask = f <= 0
            f = np.clip(f, 0.0, 1.0)
            
            # --- 2. Mutation: current-to-pbest/1 ---
            # Sort population by fitness
            sorted_idx = np.argsort(fitness)
            
            # Select p-best (Top 10%, minimum 2)
            num_top = max(2, int(0.1 * pop_size))
            top_indices = sorted_idx[:num_top]
            pbest_idx = np.random.choice(top_indices, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1 (distinct from i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            col = r1_idx == np.arange(pop_size)
            r1_idx[col] = (r1_idx[col] + 1) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 (distinct from r1 and i, from Population U Archive)
            union_size = pop_size + arc_count
            r2_idx = np.random.randint(0, union_size, pop_size)
            
            # Approximate collision avoidance for speed
            col2 = (r2_idx == r1_idx) | (r2_idx == np.arange(pop_size))
            r2_idx[col2] = (r2_idx[col2] + 1) % union_size
            
            # Construct x_r2 vector from Pop or Archive
            x_r2 = np.empty((pop_size, dim))
            mask_pop = r2_idx < pop_size
            
            # Fill from Population
            x_r2[mask_pop] = pop[r2_idx[mask_pop]]
            
            # Fill from Archive
            mask_arc = ~mask_pop
            if np.any(mask_arc):
                idx_in_arc = r2_idx[mask_arc] - pop_size
                x_r2[mask_arc] = archive[idx_in_arc]
                
            # Compute Mutant Vector V
            # Reshape F for broadcasting: (pop_size, 1) * (pop_size, dim)
            f_col = f.reshape(-1, 1)
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- 3. Crossover (Binomial) ---
            cross_mask = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # --- 4. Bound Handling (Clipping) ---
            trial = np.clip(trial, min_b, max_b)
            
            # --- 5. Selection and Memory Update ---
            succ_f = []
            succ_cr = []
            succ_diff = []
            
            # Evaluate trial vectors
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_val
                    
                val_trial = func(trial[i])
                
                if val_trial < fitness[i]:
                    # Successful Update
                    diff = fitness[i] - val_trial
                    
                    # Update Archive: Add parent to archive
                    if arc_count < pop_size:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Random replacement if full
                        rand_k = np.random.randint(0, pop_size)
                        archive[rand_k] = pop[i].copy()
                    
                    # Store parameter success data
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    succ_diff.append(diff)
                    
                    # Update Population
                    pop[i] = trial[i]
                    fitness[i] = val_trial
                    
                    # Update Global Best
                    if val_trial < best_val:
                        best_val = val_trial
                        best_vec = trial[i].copy()
                        
            # --- 6. SHADE Memory Adaptation ---
            if len(succ_f) > 0:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                s_diff = np.array(succ_diff)
                
                # Calculate weights based on fitness improvement
                total_imp = np.sum(s_diff)
                if total_imp > 0:
                    w = s_diff / total_imp
                    
                    # Weighted Arithmetic Mean for CR
                    mean_cr = np.sum(w * s_cr)
                    
                    # Weighted Lehmer Mean for F ( sum(w*f^2) / sum(w*f) )
                    num = np.sum(w * (s_f ** 2))
                    den = np.sum(w * s_f)
                    mean_f = num / den if den > 0 else 0.5
                    
                    # Update Memory at pointer k
                    m_cr[k_mem] = np.clip(mean_cr, 0.0, 1.0)
                    m_f[k_mem] = np.clip(mean_f, 0.0, 1.0)
                    
                    # Advance memory pointer
                    k_mem = (k_mem + 1) % memory_size
                    
        # Loop broken (Convergence) -> Prepare for next restart
        restart_count += 1
        
    return best_val
