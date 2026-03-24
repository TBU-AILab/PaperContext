#The following algorithm implements **L-SHADE** (Linear Population Size Reduction Success-History Adaptive Differential Evolution). It is widely considered one of the state-of-the-art variants of Differential Evolution for black-box optimization.
#
#**Key Improvements over the previous JADE implementation:**
#1.  **Linear Population Size Reduction (LPSR):** The population size linearly decreases from a large initial value (for exploration) to a small minimal value (for exploitation) as time progresses. This drastically improves convergence accuracy in fixed-budget scenarios.
#2.  **Success-History Adaptation:** Instead of simple adaptive parameters, it uses a memory (history) of successful $F$ and $CR$ values to sample new parameters. This allows the algorithm to learn multimodal distributions of optimal control parameters.
#3.  **External Archive:** It maintains an archive of recently superseded solutions. This preserves diversity by allowing the "difference vectors" in mutation to utilize information from promising but slightly inferior regions.
#4.  **Robust Vectorization:** Fully vectorized operations for mutation, crossover, and memory updates to maximize the number of function evaluations within `max_time`.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE (Linear Population Size Reduction 
    Success-History Adaptive Differential Evolution).
    """
    start_time = time.time()
    
    # --- Configuration & Hyperparameters ---
    # Bounds processing
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Population Size Parameters (LPSR)
    # Start with a large population for exploration, reduce to min_pop_size
    # CEC benchmarks often suggest N_init around 18 * dim
    init_pop_size = int(np.clip(18 * dim, 30, 200))
    min_pop_size = 4
    
    # Archive Parameters
    arc_rate = 2.0  # Archive capacity relative to current population size
    archive = []    # Stores superseded solutions
    
    # Memory Parameters (History Adaptation)
    H = 5           # History memory size
    mem_M_cr = np.full(H, 0.5) # Memory for Crossover Probability
    mem_M_f = np.full(H, 0.5)  # Memory for Mutation Factor
    mem_k = 0       # Current memory index
    
    # --- Initialization ---
    pop_size = init_pop_size
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fit = float('inf')
    # best_sol = None # (Optional: kept if we needed to return the vector)
    
    # Initial Evaluation
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_fit if best_fit != float('inf') else float('inf')
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_fit:
            best_fit = val
            
    # Sort population by fitness (best at index 0)
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Main Optimization Loop ---
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            return best_fit
            
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on time progress
        progress = elapsed / max_time
        new_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        if new_size < min_pop_size: new_size = min_pop_size
        
        # If reduction is needed
        if new_size < pop_size:
            pop_size = new_size
            # Truncate population (since it's sorted, we remove the worst)
            pop = pop[:pop_size]
            fitness = fitness[:pop_size]
            
            # Resize Archive to maintain arc_rate proportionality
            target_arc_size = max(0, int(pop_size * arc_rate))
            while len(archive) > target_arc_size:
                # Randomly remove elements
                archive.pop(np.random.randint(0, len(archive)))
                
        # 2. Parameter Generation (Vectorized)
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_M_cr[r_idx]
        m_f = mem_M_f[r_idx]
        
        # Generate CR ~ Normal(M_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(M_f, 0.1)
        # Handle F <= 0 (regenerate) and F > 1 (clip to 1)
        f = np.zeros(pop_size)
        # Generate sequentially for correctness with retry logic (fast enough for N~100)
        for i in range(pop_size):
            while True:
                val = m_f[i] + 0.1 * np.random.standard_cauchy()
                if val > 0:
                    f[i] = min(val, 1.0)
                    break
                    
        # 3. Mutation: current-to-pbest/1
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # Select p-best indices (top p% of population)
        # p is essentially random or fixed; L-SHADE often uses greedy p.
        # We use p = 0.11 (top 11%)
        p_val = 0.11
        top_p = max(2, int(pop_size * p_val))
        pbest_indices = np.random.randint(0, top_p, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (from Pop, != i)
        r1 = np.random.randint(0, pop_size, pop_size)
        # Fix self-collisions
        hit_self = (r1 == np.arange(pop_size))
        while np.any(hit_self):
            r1[hit_self] = np.random.randint(0, pop_size, np.sum(hit_self))
            hit_self = (r1 == np.arange(pop_size))
        x_r1 = pop[r1]
        
        # Select r2 (from Pop Union Archive, != i, != r1)
        if len(archive) > 0:
            a_arr = np.array(archive)
            pop_union_arc = np.vstack((pop, a_arr))
        else:
            pop_union_arc = pop
            
        n_union = len(pop_union_arc)
        r2 = np.random.randint(0, n_union, pop_size)
        
        # Fix collisions for r2
        hit = (r2 == np.arange(pop_size)) | (r2 == r1)
        while np.any(hit):
            r2[hit] = np.random.randint(0, n_union, np.sum(hit))
            hit = (r2 == np.arange(pop_size)) | (r2 == r1)
        x_r2 = pop_union_arc[r2]
        
        # Compute Mutant Vectors
        f_col = f[:, None]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask, mutant, pop)
        
        # 5. Bound Constraints (Clipping)
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Evaluation & Selection
        success_indices = []
        diffs = []
        
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_fit
                
            f_trial = func(trial[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                # If strictly better, store improvements for memory update
                if f_trial < fitness[i]:
                    # Archive maintenance: Add parent X to archive
                    if len(archive) < int(pop_size * arc_rate):
                        archive.append(pop[i].copy())
                    else:
                        # Replace random individual in full archive
                        if len(archive) > 0:
                            archive[np.random.randint(0, len(archive))] = pop[i].copy()
                            
                    diffs.append(fitness[i] - f_trial)
                    success_indices.append(i)
                
                # Update Population
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    
        # 7. Update History Memory
        if len(success_indices) > 0:
            s_idx = np.array(success_indices)
            s_f = f[s_idx]
            s_cr = cr[s_idx]
            s_df = np.array(diffs)
            
            # Calculate Weights based on fitness improvement
            if np.sum(s_df) > 0:
                w = s_df / np.sum(s_df)
            else:
                w = np.ones(len(s_df)) / len(s_df)
            
            # Weighted Lehmer Mean for F
            mean_f = np.sum(w * (s_f**2)) / np.sum(w * s_f)
            
            # Weighted Arithmetic Mean for CR
            mean_cr = np.sum(w * s_cr)
            
            # Update Memory at index k
            mem_M_f[mem_k] = mean_f
            mem_M_cr[mem_k] = mean_cr
            mem_k = (mem_k + 1) % H
            
        # 8. Sort Population for next generation (required for p-best selection)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
    return best_fit
