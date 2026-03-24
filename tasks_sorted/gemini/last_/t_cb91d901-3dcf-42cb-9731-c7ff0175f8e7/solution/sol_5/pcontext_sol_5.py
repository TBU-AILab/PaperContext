#Here is a fully operational, self-contained Python algorithm implementing **L-SHADE** (Linear Success-History based Adaptive Differential Evolution).
#
#### Improvement Logic
#The previous algorithm (IPOP-CMA-ES) is powerful for local convergence and rotated landscapes but can be computationally expensive ($O(N^3)$ per generation) and sometimes converges prematurely on highly multimodal functions.
#
#**L-SHADE** is a state-of-the-art global optimizer (winner of multiple CEC competitions) that improves upon this by:
#1.  **Adaptive Control Parameters ($F$ and $Cr$):** It learns the optimal mutation scale and crossover rate based on successful past updates, removing the need for manual tuning.
#2.  **Linear Population Size Reduction (LPSR):** It starts with a large population to explore the global space and linearly reduces the population size over time. This forces the algorithm to shift naturally from **Exploration** (finding the valley) to **Exploitation** (refining the minimum).
#3.  **External Archive:** It maintains diversity by utilizing inferior solutions (recently replaced parents) in the mutation strategy, preventing stagnation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Implements L-SHADE (Linear Success-History based Adaptive Differential Evolution).
    
    Features:
    - Adapts mutation factor (F) and crossover rate (Cr) using a historical memory.
    - Uses 'current-to-pbest/1' mutation strategy using an external archive.
    - Linearly reduces population size from N_init to N_min to balance exploration/exploitation.
    """
    
    # --- Configuration ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # L-SHADE Constants
    N_min = 4  # Minimum population size
    # Initial population size: 18 * dim is a standard literature value for SHADE, 
    # but we cap it for very high dimensions to ensure speed.
    N_init = int(round(max(10, min(18 * dim, 300)))) 
    
    pop_size = N_init
    max_pop_size = N_init
    
    # History Memory Parameters
    H = 6 # Memory size
    M_cr = np.full(H, 0.5) # Memory for Crossover
    M_f = np.full(H, 0.5)  # Memory for Mutation Factor
    k_mem = 0 # Memory index pointer
    
    # Archiving
    archive = []
    arc_rate = 2.6
    max_arc_size = int(round(N_init * arc_rate))
    
    # --- Initialization ---
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    
    # Random initial population
    pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.zeros(pop_size)
    
    # Evaluate initial population
    best_fitness = float('inf')
    best_sol = None
    
    # Safety check for immediate timeout
    if (datetime.now() - start_time) >= limit:
        # If no time, just run one random guess
        return func(pop[0])

    for i in range(pop_size):
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
        # Time check inside init loop for very slow functions
        if (datetime.now() - start_time) >= limit:
            return best_fitness

    # --- Main Loop ---
    while True:
        # Check Time
        elapsed = datetime.now() - start_time
        if elapsed >= limit:
            break
        
        # Calculate Progress (0.0 to 1.0) for Population Reduction
        # We rely on time ratio since max_evals is unknown.
        progress = min(1.0, elapsed.total_seconds() / max_time)
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target population size based on remaining time
        next_pop_size = int(round(((N_min - N_init) * progress) + N_init))
        next_pop_size = max(N_min, next_pop_size)
        
        if pop_size > next_pop_size:
            # Sort by fitness (descending badness) to remove worst
            sort_idx = np.argsort(fitness)
            n_remove = pop_size - next_pop_size
            
            # Remove worst individuals
            # Keep the top 'next_pop_size'
            keep_idx = sort_idx[:next_pop_size]
            pop = pop[keep_idx]
            fitness = fitness[keep_idx]
            pop_size = next_pop_size
            
            # Resize Archive if necessary
            curr_arc_size = int(round(pop_size * arc_rate))
            if len(archive) > curr_arc_size:
                # Randomly delete excess archive members
                del_indices = np.random.choice(len(archive), len(archive) - curr_arc_size, replace=False)
                archive = [arr for i, arr in enumerate(archive) if i not in del_indices]
                max_arc_size = curr_arc_size

        # 2. Parameter Generation
        # Generate F and Cr for each individual based on memory
        # Pick random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_cr_selected = M_cr[r_idx]
        m_f_selected = M_f[r_idx]
        
        # Cauchy distribution for F: location=M_f, scale=0.1
        # If F > 1, cap at 1. If F <= 0, regenerate.
        crs = np.random.normal(m_cr_selected, 0.1)
        crs = np.clip(crs, 0, 1) # Normal dist clipped [0,1]
        
        # Cauchy generation manually: loc + scale * tan(pi * (rand - 0.5))
        fs = np.zeros(pop_size)
        for i in range(pop_size):
            while True:
                f_val = m_f_selected[i] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                if f_val <= 0: continue
                if f_val > 1: f_val = 1.0
                fs[i] = f_val
                break

        # 3. Mutation (current-to-pbest/1) & Crossover
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        
        # p-best parameter (top p% of population, minimum 2 individuals)
        p_ratio = max(2.0 / pop_size, 0.2 * (1 - progress)) # Adaptive p seems to help, or fixed 0.11
        p_count = max(2, int(round(p_ratio * pop_size)))
        
        trial_pop = np.zeros_like(pop)
        
        # Prepare Archive + Population for the second difference vector
        # (Union of current population and archive)
        if len(archive) > 0:
            pop_all = np.vstack((pop, np.array(archive)))
        else:
            pop_all = pop
            
        for i in range(pop_size):
            # Select p-best: random from top p_count
            pbest_idx = sorted_indices[np.random.randint(0, p_count)]
            x_pbest = pop[pbest_idx]
            
            # Select r1: random from pop, != i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = pop[r1]
            
            # Select r2: random from pop U archive, != i, != r1
            r2 = np.random.randint(0, len(pop_all))
            while r2 == i or (r2 < pop_size and r2 == r1):
                r2 = np.random.randint(0, len(pop_all))
            x_r2 = pop_all[r2]
            
            # Mutation Vector
            mutant = pop[i] + fs[i] * (x_pbest - pop[i]) + fs[i] * (x_r1 - x_r2)
            
            # Binomial Crossover
            # Pick a random dimension to ensure at least one changes
            j_rand = np.random.randint(0, dim)
            mask = (np.random.rand(dim) < crs[i])
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, pop[i])
            
            # Bound Handling: Resample if out of bounds or Clip? 
            # SHADE usually clips or uses bounce-back. Clipping is safer/faster.
            trial = np.clip(trial, lb, ub)
            trial_pop[i] = trial

        # 4. Evaluation & Selection
        trial_fitness = np.zeros(pop_size)
        success_mask = np.zeros(pop_size, dtype=bool)
        diff_fitness = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Check time strictly inside the evaluation loop
            if (datetime.now() - start_time) >= limit:
                return best_fitness
            
            f_trial = func(trial_pop[i])
            trial_fitness[i] = f_trial
            
            # Greedy Selection
            if f_trial < fitness[i]:
                success_mask[i] = True
                diff_fitness[i] = fitness[i] - f_trial
                
                # Add parent to archive before replacement
                archive.append(pop[i].copy())
                
                # Update population
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
        
        # Maintain Archive Size
        if len(archive) > max_arc_size:
            # Randomly remove elements
            num_to_remove = len(archive) - max_arc_size
            # Simply shuffle and slice for efficiency
            import random
            random.shuffle(archive)
            archive = archive[:max_arc_size]

        # 5. Update Memory (History)
        # Only if there were successful individuals
        if np.any(success_mask):
            succ_f = fs[success_mask]
            succ_cr = crs[success_mask]
            succ_diff = diff_fitness[success_mask]
            
            # Weights based on fitness improvement
            weights = succ_diff / np.sum(succ_diff)
            
            # Weighted Lehmer Mean for F
            # mean_wl = sum(w * f^2) / sum(w * f)
            mean_f = np.sum(weights * (succ_f ** 2)) / np.sum(weights * succ_f)
            
            # Weighted Arithmetic Mean for Cr
            mean_cr = np.sum(weights * succ_cr)
            
            # Update Memory (moving average could be used, here simple ring buffer)
            M_f[k_mem] = mean_f
            M_cr[k_mem] = mean_cr
            
            k_mem = (k_mem + 1) % H
            
    return best_fitness
