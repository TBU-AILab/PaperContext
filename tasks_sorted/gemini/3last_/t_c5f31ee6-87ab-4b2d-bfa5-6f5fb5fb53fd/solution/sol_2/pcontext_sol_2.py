#The following algorithm is an implementation of **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution)**, simplified for this context.
#
#### Improvements over previous algorithms:
#1.  **Linear Population Size Reduction (LPSR):** The algorithm starts with a large population to explore the search space globally and linearly reduces the population size as time progresses. This shifts the focus from exploration to exploitation, dedicating computational power to refining the best solutions in the final stages.
#2.  **Adaptive Parameters (History-based):** Instead of fixed $F$ (Mutation) and $CR$ (Crossover) values, it maintains a memory of successful parameter configurations and adapts them. This removes the need for manual tuning and adapts to the specific function landscape.
#3.  **Current-to-pbest Strategy:** It uses a greedy mutation strategy `DE/current-to-pbest/1`, guiding individuals toward the top $p\%$ of the population rather than just the single best (which prevents premature convergence) or random ones (which is too slow).
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History Adaptive Differential Evolution).
    
    This algorithm adapts F and CR parameters based on historical success 
    and linearly reduces population size to balance exploration and exploitation.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial population size: High for exploration
    p_init = int(max(20, 18 * dim))
    # Minimum population size: Low for final exploitation
    p_min = 4
    
    # Memory for adaptive parameters
    memory_size = 5
    memory_idx = 0
    # Initialize memory with 0.5
    m_cr = np.full(memory_size, 0.5)
    m_f = np.full(memory_size, 0.5)
    
    # Archive is omitted for code compactness, utilizing population diversity instead.
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    pop_size = p_init
    # Generate initial population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_idx = -1
    
    # Evaluate initial population
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val if best_idx != -1 else func(pop[i])
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Loop ---
    while True:
        # Check time
        elapsed = datetime.now() - start_time
        if elapsed >= time_limit:
            return best_val
            
        # 1. Linear Population Size Reduction
        # Calculate progress (0.0 to 1.0)
        progress = elapsed.total_seconds() / max_time
        new_pop_size = int(round((p_min - p_init) * progress + p_init))
        new_pop_size = max(p_min, new_pop_size)
        
        # If population needs reduction
        if pop_size > new_pop_size:
            # Sort by fitness (ascending) and keep the best 'new_pop_size'
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices[:new_pop_size]]
            fitness = fitness[sort_indices[:new_pop_size]]
            pop_size = new_pop_size
            # Update best index after sort
            best_idx = 0 # Since it's sorted, best is at 0
            best_val = fitness[0]

        # 2. Parameter Generation
        # Select random memory index for each individual
        r_idx = np.random.randint(0, memory_size, pop_size)
        mu_cr = m_cr[r_idx]
        mu_f = m_f[r_idx]
        
        # Generate CR: Normal distribution around memory, clipped [0, 1]
        # If CR < 0 -> 0, If CR > 1 -> 1. CR ~ N(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F: Cauchy distribution. If F > 1 -> 1. If F <= 0 -> Regenerate
        # F ~ C(mu_f, 0.1)
        f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Retry logic for F <= 0 (vectorized replacement)
        while np.any(f <= 0):
            mask = f <= 0
            f[mask] = mu_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
        f = np.clip(f, 0, 1) # Standard clipping for upper bound

        # 3. Mutation: DE/current-to-pbest/1
        # Sort population to identify top p-best individuals
        sorted_indices = np.argsort(fitness)
        
        # p-best size: typically top 11% (p=0.11), min 2 individuals
        p_best_rate = 0.11
        num_p_best = max(2, int(pop_size * p_best_rate))
        top_p_indices = sorted_indices[:num_p_best]
        
        # Select p_best for each individual
        p_best_choice = np.random.choice(top_p_indices, pop_size)
        x_pbest = pop[p_best_choice]
        
        # Select r1 and r2
        # r1 != i, r2 != r1 != i
        # fast random selection (collisions are rare/acceptable in high dim/speed tradeoff)
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Enforce distinctness manually for robustness
        for k in range(pop_size):
            while r1[k] == k: 
                r1[k] = np.random.randint(0, pop_size)
            while r2[k] == k or r2[k] == r1[k]: 
                r2[k] = np.random.randint(0, pop_size)
                
        x_r1 = pop[r1]
        x_r2 = pop[r2]
        
        # Mutation Vector V = Current + F*(pbest - Current) + F*(r1 - r2)
        # Reshape f for broadcasting: (pop_size, 1)
        f_col = f[:, None]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover: Binomial
        # Create mask
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 5. Boundary Handling
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Selection and Memory Update
        improved_f = []
        improved_cr = []
        improvement_magnitudes = []
        
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            trial_val = func(trial[i])
            
            if trial_val <= fitness[i]:
                # Improvement or equal found
                if trial_val < fitness[i]:
                    # Only record strictly better for memory updates
                    improved_f.append(f[i])
                    improved_cr.append(cr[i])
                    improvement_magnitudes.append(fitness[i] - trial_val)
                
                fitness[i] = trial_val
                pop[i] = trial[i]
                
                if trial_val < best_val:
                    best_val = trial_val
                    best_idx = i
        
        # 7. Update Memory (Weighted Lehmer Mean)
        if len(improved_f) > 0:
            weights = np.array(improvement_magnitudes)
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                
                # Lehmer Mean for F
                f_np = np.array(improved_f)
                mean_f = np.sum(weights * (f_np ** 2)) / np.sum(weights * f_np)
                
                # Weighted Arithmetic Mean for CR
                cr_np = np.array(improved_cr)
                mean_cr = np.sum(weights * cr_np)
                
                # Update memory at index
                m_f[memory_idx] = mean_f
                m_cr[memory_idx] = mean_cr
                
                # Increment memory index
                memory_idx = (memory_idx + 1) % memory_size
            
    return best_val
