#The following algorithm implements an improved version of **L-SHADE** (Linear Population Size Reduction Success-History Adaptive Differential Evolution), incorporating elements from the **jSO** (journal-SHADE) variant which is known for high performance in constrained optimization.
#
#**Key Improvements over previous versions:**
#1.  **Midpoint Bound Constraint Handling**: Instead of simply clipping values to the bounds (which can reduce diversity near the edges), this algorithm sets the violating variable to the midpoint between the parent value and the bound. This "bounce-back" behavior preserves population variance while ensuring feasibility.
#2.  **Dynamic $p$-best Strategy**: The parameter $p$, which controls the greediness of the mutation strategy (`current-to-pbest`), adapts linearly from $0.20$ (exploration) down to $0.05$ (exploitation) over the course of the optimization.
#3.  **Linear Population Size Reduction (LPSR)**: The population size reduces linearly from a robust initial size (based on dimension) down to a minimal size (4), forcing convergence exactly as the time limit approaches.
#4.  **Robust Time Management**: The algorithm performs strict time checks inside the evaluation loop to handle expensive functions or very short time limits gracefully.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using an improved L-SHADE algorithm with jSO features.
    
    Key mechanisms:
    - Linear Population Size Reduction (LPSR).
    - Dynamic 'p' parameter for mutation (Exploration -> Exploitation).
    - Midpoint-target bound constraint handling (Better than clipping).
    - Adaptive F and CR parameters with Success History Memory.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial Population Size: 
    # 18*dim is a standard robust heuristic for L-SHADE-RSP/jSO.
    # Capped between 30 and 250 to ensure performance on various hardware/time limits.
    pop_size_init = int(round(18 * dim))
    pop_size_init = max(30, min(pop_size_init, 250))
    min_pop_size = 4
    
    # Memory Parameters (History Size H=6)
    H = 6
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive parameters
    archive = []
    arc_rate = 2.0 # Archive size relative to current population
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    diff_b = ub - lb
    
    # Initialize Population
    population = lb + np.random.rand(pop_size_init, dim) * diff_b
    fitness = np.full(pop_size_init, float('inf'))
    
    best_val = float('inf')
    
    # Evaluate Initial Population
    for i in range(pop_size_init):
        # Strict time check
        if (datetime.now() - start_time) >= limit:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            
    # Sort population by fitness (required for p-best selection)
    sorted_idxs = np.argsort(fitness)
    population = population[sorted_idxs]
    fitness = fitness[sorted_idxs]
    
    # --- Main Optimization Loop ---
    while True:
        # Global Time Check
        now = datetime.now()
        if (now - start_time) >= limit:
            break
            
        # Calculate Progress (0.0 to 1.0)
        elapsed = (now - start_time).total_seconds()
        progress = min(1.0, elapsed / max_time)
        
        # 1. Linear Population Size Reduction (LPSR)
        target_size = int(round(pop_size_init + (min_pop_size - pop_size_init) * progress))
        target_size = max(min_pop_size, target_size)
        curr_size = len(population)
        
        if curr_size > target_size:
            # Truncate population (worst individuals are at the end due to sort)
            population = population[:target_size]
            fitness = fitness[:target_size]
            curr_size = target_size
            
            # Reduce Archive if necessary
            max_arc_size = int(curr_size * arc_rate)
            while len(archive) > max_arc_size:
                archive.pop(np.random.randint(len(archive)))
                
        # 2. Dynamic p-value
        # Scales linearly from 0.2 down to 0.05 to shift from exploration to exploitation
        p_val = 0.2 - 0.15 * progress
        p_val = max(0.05, p_val)
        
        # 3. Parameter Adaptation (F and CR)
        r_idxs = np.random.randint(0, H, curr_size)
        m_f = mem_f[r_idxs]
        m_cr = mem_cr[r_idxs]
        
        # Generate CR: Normal(mean=M_cr, std=0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F: Cauchy(loc=M_f, scale=0.1)
        f = m_f + 0.1 * np.random.standard_cauchy(curr_size)
        
        # Repair F: if F <= 0, regenerate. if F > 1, clip to 1.
        retry_mask = f <= 0
        while np.any(retry_mask):
            f[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
            retry_mask = f <= 0
        f = np.clip(f, 0, 1)
        
        # 4. Mutation Strategy: current-to-pbest/1
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # Select X_pbest from top p%
        p_num = max(2, int(p_val * curr_size))
        pbest_idxs = np.random.randint(0, p_num, curr_size)
        x_pbest = population[pbest_idxs]
        
        # Select X_r1 (random from population, distinct from i)
        r1_idxs = np.random.randint(0, curr_size, curr_size)
        for i in range(curr_size):
            while r1_idxs[i] == i:
                r1_idxs[i] = np.random.randint(0, curr_size)
        x_r1 = population[r1_idxs]
        
        # Select X_r2 (random from Union(Population, Archive), distinct from i and r1)
        if len(archive) > 0:
            union_pop = np.vstack((population, np.array(archive)))
        else:
            union_pop = population
        union_size = len(union_pop)
        
        r2_idxs = np.random.randint(0, union_size, curr_size)
        for i in range(curr_size):
            while (r2_idxs[i] == i) or (r2_idxs[i] == r1_idxs[i]):
                r2_idxs[i] = np.random.randint(0, union_size)
        x_r2 = union_pop[r2_idxs]
        
        # Compute Mutant Vectors
        f_v = f[:, np.newaxis]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        rand_v = np.random.rand(curr_size, dim)
        mask = rand_v < cr[:, np.newaxis]
        j_rand = np.random.randint(0, dim, curr_size)
        mask[np.arange(curr_size), j_rand] = True
        
        trial = np.where(mask, mutant, population)
        
        # 6. Bound Constraint Handling: Midpoint Target
        # If a variable violates bounds, set it to the average of the bound and the parent.
        # This is statistically superior to clipping for Differential Evolution.
        lower_mask = trial < lb
        trial[lower_mask] = (lb[lower_mask] + population[lower_mask]) / 2.0
        
        upper_mask = trial > ub
        trial[upper_mask] = (ub[upper_mask] + population[upper_mask]) / 2.0
        
        # 7. Evaluation & Selection
        success_f = []
        success_cr = []
        diff_f = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(curr_size):
            if (datetime.now() - start_time) >= limit:
                return best_val
            
            t_val = func(trial[i])
            
            # Greedy selection
            if t_val <= fitness[i]:
                new_pop[i] = trial[i]
                new_fit[i] = t_val
                
                # If strictly better, record success for memory update
                if t_val < fitness[i]:
                    archive.append(population[i].copy())
                    diff_f.append(fitness[i] - t_val)
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                
                if t_val < best_val:
                    best_val = t_val
        
        population = new_pop
        fitness = new_fit
        
        # 8. Archive Maintenance (Keep size bounded)
        max_arc_size = int(curr_size * arc_rate)
        while len(archive) > max_arc_size:
            archive.pop(np.random.randint(len(archive)))
            
        # 9. Memory Update (Weighted Lehmer Mean)
        if len(diff_f) > 0:
            w = np.array(diff_f)
            w_sum = np.sum(w)
            if w_sum > 0:
                w /= w_sum
                
                sf = np.array(success_f)
                scr = np.array(success_cr)
                
                # Lehmer Mean for F
                mean_f = np.sum(w * (sf ** 2)) / (np.sum(w * sf) + 1e-15)
                # Weighted Mean for CR
                mean_cr = np.sum(w * scr)
                
                mem_f[k_mem] = mean_f
                mem_cr[k_mem] = mean_cr
                k_mem = (k_mem + 1) % H
                
        # 10. Resort Population for next generation
        sorted_idxs = np.argsort(fitness)
        population = population[sorted_idxs]
        fitness = fitness[sorted_idxs]

    return best_val
