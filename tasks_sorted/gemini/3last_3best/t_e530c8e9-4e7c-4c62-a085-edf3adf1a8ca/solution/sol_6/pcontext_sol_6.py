#The following is a fully operational, self-contained implementation of an improved optimization algorithm.
#
#**Algorithm Explanation:**
#This algorithm is an **Asynchronous Self-Adaptive Differential Evolution with Archive (jDE-Async)**. It is designed to outperform standard DE and complex variants like L-SHADE in time-constrained scenarios by maximizing the number of effective evaluations while maintaining robust convergence properties.
#
#1.  **Asynchronous Updates**: Unlike standard generational algorithms (where the population is updated all at once at the end), this algorithm updates individuals immediately after a successful trial. This allows beneficial genetic information to propagate through the population instantly, significantly accelerating convergence speed.
#2.  **External Archive**: An archive stores high-quality solutions recently displaced from the population. This preserves diversity and prevents the algorithm from cycling or converging prematurely, a common issue in greedy strategies.
#3.  **Strategy**: It uses `current-to-pbest/1/bin` with the difference vector utilizing the archive (`x_r2` from Union(Pop, Archive)). This balances exploitation (moving towards top $p\%$ best) and exploration.
#4.  **jDE Parameter Adaptation**: Instead of complex history-based adaptation (which adds overhead), it uses the lightweight jDE mechanism where $F$ and $CR$ control parameters are encoded into each individual and evolve with them.
#5.  **Restart Mechanism**: A stagnation detector monitors population variance. If the search stalls, it triggers a soft restart, preserving the global best but refreshing the rest of the population to explore new basins of attraction within the remaining time.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Asynchronous Self-Adaptive Differential Evolution 
    with Archive (jDE-Async) and Restart Mechanism.
    """
    start_time = time.time()
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Adaptive Population Size
    # Moderate size allows for more generations/updates within the time limit.
    # Clipped to a reasonable range to ensure sufficient diversity without excessive overhead.
    pop_size = int(np.clip(dim * 20, 50, 150))
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize jDE Control Parameters (one per individual)
    # F: Mutation factor, CR: Crossover rate
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Archive to store replaced individuals (preserves diversity)
    archive = []
    
    # Global Best Tracking
    best_idx = -1
    best_val = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i
            
    # Configuration for p-best selection
    # Select from the top 10% of individuals
    p_ratio = 0.1
    
    # --- Main Optimization Loop ---
    while True:
        # Time Check
        if (time.time() - start_time) >= max_time:
            return best_val
            
        # Sort population by fitness to identify 'p-best' individuals
        # Although updates are asynchronous, we sort once per 'generation' loop for efficiency.
        sorted_indices = np.argsort(fitness)
        
        # --- Stagnation Detection & Restart ---
        # If population diversity (std dev) is extremely low, we are stuck.
        if np.std(fitness) < 1e-6:
            # Only restart if we have meaningful time left (>10% of max_time)
            if (time.time() - start_time) < (max_time * 0.9):
                # Preserve the single best solution found so far
                saved_best_vec = pop[best_idx].copy()
                saved_best_val = best_val
                
                # Re-initialize the rest of the population randomly
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                pop[0] = saved_best_vec  # Place best at index 0
                
                # Reset parameters to encourage exploration
                F[:] = 0.5
                CR[:] = 0.9
                
                # Reset Archive
                archive = []
                
                # Re-evaluate new population (skipping index 0)
                fitness[:] = float('inf')
                fitness[0] = saved_best_val
                best_idx = 0
                
                for k in range(1, pop_size):
                    if (time.time() - start_time) >= max_time:
                        return best_val
                    val = func(pop[k])
                    fitness[k] = val
                    if val < best_val:
                        best_val = val
                        best_idx = k
                
                # Re-sort indices after restart
                sorted_indices = np.argsort(fitness)
        
        # --- Asynchronous Evolution Loop ---
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
            
            idx = i # Current individual index
            
            # 1. Parameter Adaptation (jDE Logic)
            # With small probability, regenerate F and CR
            new_F = F[idx]
            if np.random.rand() < 0.1:
                new_F = 0.1 + 0.9 * np.random.rand() # Range [0.1, 1.0]
            
            new_CR = CR[idx]
            if np.random.rand() < 0.1:
                new_CR = np.random.rand() # Range [0.0, 1.0]
            
            # 2. Mutation Strategy: current-to-pbest/1
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Select X_pbest: Randomly from top p%
            p_count = max(2, int(pop_size * p_ratio))
            p_rand_idx = np.random.randint(0, p_count)
            idx_pbest = sorted_indices[p_rand_idx]
            x_pbest = pop[idx_pbest]
            
            x_curr = pop[idx]
            
            # Select r1: Random from population, distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == idx:
                r1 = np.random.randint(0, pop_size)
            x_r1 = pop[r1]
            
            # Select r2: Random from Union(Population, Archive), distinct from i and r1
            union_size = pop_size + len(archive)
            r2 = np.random.randint(0, union_size)
            while r2 == idx or r2 == r1:
                r2 = np.random.randint(0, union_size)
            
            if r2 < pop_size:
                x_r2 = pop[r2]
            else:
                x_r2 = archive[r2 - pop_size]
                
            # Generate Mutant Vector
            mutant = x_curr + new_F * (x_pbest - x_curr) + new_F * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < new_CR
            # Guarantee at least one parameter changes
            j_rand = np.random.randint(0, dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, x_curr)
            
            # 4. Boundary Constraint Handling (Midpoint Target)
            # If a variable is out of bounds, set it to the average of the bound and the parent.
            # This is less biased than clipping to the bound.
            
            # Lower bound check
            viol_l = trial < min_b
            if np.any(viol_l):
                trial[viol_l] = (min_b[viol_l] + x_curr[viol_l]) * 0.5
            
            # Upper bound check
            viol_u = trial > max_b
            if np.any(viol_u):
                trial[viol_u] = (max_b[viol_u] + x_curr[viol_u]) * 0.5
                
            # 5. Evaluation
            f_trial = func(trial)
            
            # 6. Selection (Greedy)
            if f_trial < fitness[idx]:
                # Improvement found
                
                # Add the old individual to the Archive
                if len(archive) < pop_size:
                    archive.append(x_curr.copy())
                else:
                    # If full, replace a random archive member
                    rand_loc = np.random.randint(0, len(archive))
                    archive[rand_loc] = x_curr.copy()
                
                # Update Population immediately (Asynchronous)
                pop[idx] = trial
                fitness[idx] = f_trial
                
                # Adopt the new parameters
                F[idx] = new_F
                CR[idx] = new_CR
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = idx
            else:
                # No improvement: Discard trial and temporary parameters
                pass

    return best_val
