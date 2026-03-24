#Here is an improved algorithm. It implements **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution) with Restart**.
#
#### Rationale for Improvement
#The previous best result was achieved using JADE. To improve further, this solution incorporates features from **SHADE** and **L-SHADE**, which are state-of-the-art enhancements over JADE, and adds a **Restart Mechanism**.
#1.  **History-Based Parameter Adaptation (SHADE)**: Instead of a single global mean for parameters $F$ and $Cr$, this algorithm uses a memory bank. New parameters are sampled based on successful parameters from the recent history, allowing the algorithm to learn multimodal parameter distributions required for different stages of optimization.
#2.  **Linear Population Size Reduction (LPSR)**: The population size starts large to encourage exploration and linearly decreases over time. This forces the algorithm to shift from exploration to exploitation as the deadline approaches, maximizing efficiency within the fixed `max_time`.
#3.  **Restart Mechanism**: If the population converges (stagnates) before the time limit, the algorithm restarts the search (keeping the best solution found) to explore other potential basins of attraction.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE (Linear Population Size Reduction - 
    Success-History based Adaptive Differential Evolution) with Restart.
    """
    # --- Time Management ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Hyperparameters ---
    # Initial Population: High to explore
    p_init = int(max(30, 25 * dim))
    p_min = 4 # Minimum population size
    
    # SHADE Memory
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # --- Initialization ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Current Population
    pop_size = p_init
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    archive = []
    
    # Initial Evaluation
    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # --- Main Loop ---
    while datetime.now() < end_time:
        
        # 1. Linear Population Size Reduction (LPSR) based on Time
        # Calculate progress ratio (0.0 to 1.0)
        elapsed = (datetime.now() - start_time).total_seconds()
        ratio = elapsed / max_time
        if ratio > 1.0: ratio = 1.0
        
        # Determine target size
        target_size = int(round(p_init + (p_min - p_init) * ratio))
        target_size = max(p_min, target_size)
        
        # Reduce if necessary
        if pop_size > target_size:
            # Sort by fitness (ascending)
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:target_size]]
            fitness = fitness[sorted_idx[:target_size]]
            pop_size = target_size
            
            # Shrink archive to match current pop_size (common heuristic in L-SHADE)
            if len(archive) > pop_size:
                # Randomly keep pop_size elements
                np.random.shuffle(archive)
                archive = archive[:pop_size]

        # 2. Restart Mechanism
        # If population has collapsed (converged) too early, restart.
        # Check standard deviation of fitness
        if pop_size > p_min:
            std_fit = np.std(fitness)
            fit_range = np.max(fitness) - np.min(fitness)
            
            if std_fit < 1e-8 or fit_range < 1e-8:
                # Restart: Keep best, re-init rest
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                pop[0] = best_vec # Elitism
                
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_val
                
                # Reset Memory
                mem_cr.fill(0.5)
                mem_f.fill(0.5)
                archive = []
                
                # Evaluate new population
                for i in range(1, pop_size):
                    if datetime.now() >= end_time:
                        return best_val
                    val = func(pop[i])
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_vec = pop[i].copy()
                continue # Skip evolution for this iteration

        # 3. Parameter Generation (SHADE)
        # Randomly select memory index for each individual
        r_idxs = np.random.randint(0, H, pop_size)
        
        # CR ~ Normal(M_cr, 0.1)
        cr = np.random.normal(mem_cr[r_idxs], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(M_f, 0.1)
        # numpy doesn't have a direct Cauchy with location/scale, construct it:
        f = mem_f[r_idxs] + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Constrain F
        # If F > 1, clamp to 1. If F <= 0, regenerate.
        retry_mask = f <= 0
        while np.any(retry_mask):
            f[retry_mask] = mem_f[r_idxs[retry_mask]] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
            retry_mask = f <= 0
        f[f > 1] = 1.0

        # 4. Evolution
        # Prepare containers for next generation
        new_pop = np.copy(pop)
        new_fitness = np.copy(fitness)
        
        succ_cr = []
        succ_f = []
        diff_f = []
        
        # Sort for p-best selection
        sorted_indices = np.argsort(fitness)
        
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_val
                
            x_i = pop[i]
            
            # Strategy: current-to-pbest/1
            # Choose p from [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_cnt = int(max(2, p_val * pop_size))
            p_best_idx = sorted_indices[np.random.randint(0, top_cnt)]
            x_pbest = pop[p_best_idx]
            
            # r1 distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = pop[r1]
            
            # r2 distinct from i, r1 (from Union of Pop and Archive)
            n_arch = len(archive)
            r2_idx = np.random.randint(0, pop_size + n_arch)
            
            if r2_idx < pop_size:
                r2 = r2_idx
                while r2 == i or r2 == r1:
                    r2_idx = np.random.randint(0, pop_size + n_arch)
                    if r2_idx >= pop_size: break
                    r2 = r2_idx
                
                if r2_idx < pop_size:
                    x_r2 = pop[r2_idx]
                else:
                    x_r2 = archive[r2_idx - pop_size]
            else:
                x_r2 = archive[r2_idx - pop_size]
            
            # Mutation
            mutant = x_i + f[i] * (x_pbest - x_i) + f[i] * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            mask_cr = np.random.rand(dim) < cr[i]
            j_rand = np.random.randint(dim)
            mask_cr[j_rand] = True
            
            trial = np.where(mask_cr, mutant, x_i)
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                
                if f_trial < fitness[i]:
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diff_f.append(fitness[i] - f_trial)
                    archive.append(x_i.copy())
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
        
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size (<= pop_size)
        while len(archive) > pop_size:
            del archive[np.random.randint(0, len(archive))]
            
        # 5. Update Memory (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            s_f = np.array(succ_f)
            s_cr = np.array(succ_cr)
            w = np.array(diff_f)
            
            # Normalize weights
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
                
                mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                mean_cr = np.sum(w * s_cr)
                
                mem_f[k_mem] = mean_f
                mem_cr[k_mem] = mean_cr
                
                k_mem = (k_mem + 1) % H
    
    return best_val
