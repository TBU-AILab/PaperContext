#Here is a fully operational, self-contained Python implementation of an advanced optimization algorithm.
#
#This solution implements a **Self-Adaptive Differential Evolution (SHADE-inspired)** algorithm hybridised with **Local Search** and **Restarts**.
#
#**Key Features:**
#1.  **Adaptive Parameters**: It automatically adapts the Mutation Factor (`F`) and Crossover Rate (`CR`) using a history-based memory, removing the need for manual tuning and allowing the algorithm to adjust to the fitness landscape dynamically.
#2.  **Strategy `current-to-pbest/1`**: Utilizes a greedy mutation strategy that guides search towards the best solutions found so far while maintaining diversity.
#3.  **Local Polishing**: When convergence is detected (low population variance), a coordinate descent local search is triggered to refine the best solution to high precision.
#4.  **Restart Mechanism**: Escapes local optima by resetting the population (keeping the best elite) when stagnation occurs.
#5.  **Strict Time Management**: Time is checked frequently to ensure the best possible result is returned within `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a black-box function 'func' using a SHADE-inspired Differential Evolution
    with Local Search polishing and Restarts.
    """
    start_time = datetime.now()
    # Use 95% of the available time to ensure safe return
    limit = timedelta(seconds=max_time * 0.95)

    # --- 1. Algorithm Configuration ---
    # Population size: Adaptive based on dimension.
    # A larger population is safer for global search, but slower.
    # We clip it to be between 30 and 100 to balance speed and diversity.
    pop_size = int(np.clip(10 * dim, 30, 100))
    
    # SHADE Memory Parameters
    H_mem = 5  # History size for parameter adaptation
    M_CR = np.full(H_mem, 0.5)  # Memory for Crossover Rate
    M_F = np.full(H_mem, 0.5)   # Memory for Mutation Factor
    k_mem = 0  # Memory index pointer

    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Helper for Time Checking
    def is_time_up():
        return (datetime.now() - start_time) >= limit

    # --- 2. Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = np.zeros(dim)

    # Evaluate Initial Population
    for i in range(pop_size):
        if is_time_up(): return best_fitness
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()

    # --- 3. Main Optimization Loop ---
    while not is_time_up():
        
        # --- A. Parameter Generation (Adaptive) ---
        # Select random memory indices
        r_idx = np.random.randint(0, H_mem, pop_size)
        m_cr = M_CR[r_idx]
        m_f = M_F[r_idx]

        # Generate CR ~ Normal(m_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)

        # Generate F ~ Cauchy(m_f, 0.1), clipped [0.1, 1.0]
        # Cauchy gen: location + scale * tan(pi * (rand - 0.5))
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        f = np.clip(f, 0.1, 1.0) # Ensure minimal mutation strength

        # --- B. Mutation (current-to-pbest/1) ---
        # Sort population to identify top p-best individuals
        sorted_idx = np.argsort(fitness)
        
        # Randomly choose p between 2/pop_size and 0.2
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        top_count = int(max(2, p_val * pop_size))
        top_indices = sorted_idx[:top_count]
        
        # Select pbest for each individual
        pbest_indices = np.random.choice(top_indices, pop_size)
        
        # Select distinct random indices r1, r2
        idxs = np.arange(pop_size)
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Quick collision handling (simple shift)
        mask = (r1 == idxs)
        r1[mask] = (r1[mask] + 1) % pop_size
        mask = (r2 == idxs) | (r2 == r1)
        r2[mask] = (r2[mask] + 2) % pop_size
        
        # Compute Mutant Vectors
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        x = pop
        x_pbest = pop[pbest_indices]
        x_r1 = pop[r1]
        x_r2 = pop[r2]
        
        f_col = f[:, np.newaxis]
        mutant = x + f_col * (x_pbest - x) + f_col * (x_r1 - x_r2)

        # --- C. Crossover (Binomial) ---
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, np.newaxis]
        # Force at least one dimension from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[idxs, j_rand] = True
        
        trial = np.where(cross_mask, mutant, x)
        # Bound constraints (Clip)
        trial = np.clip(trial, min_b, max_b)

        # --- D. Selection & Memory Update Prep ---
        winning_f = []
        winning_cr = []
        diff_vals = []
        
        for i in range(pop_size):
            if is_time_up(): return best_fitness
            
            trial_val = func(trial[i])
            
            if trial_val <= fitness[i]:
                improvement = fitness[i] - trial_val
                
                # Record successful parameters
                if improvement > 0:
                    winning_f.append(f[i])
                    winning_cr.append(cr[i])
                    diff_vals.append(improvement)
                
                fitness[i] = trial_val
                pop[i] = trial[i]
                
                if trial_val < best_fitness:
                    best_fitness = trial_val
                    best_sol = trial[i].copy()

        # --- E. Update Adaptive Memory ---
        if len(diff_vals) > 0:
            diffs = np.array(diff_vals)
            # Weighted Lehmer Mean weights
            weights = diffs / np.sum(diffs)
            
            if np.sum(weights) > 0:
                s_f = np.array(winning_f)
                s_cr = np.array(winning_cr)
                
                mean_cr = np.sum(weights * s_cr)
                mean_f = np.sum(weights * s_f**2) / np.sum(weights * s_f)
                
                M_CR[k_mem] = mean_cr
                M_F[k_mem] = mean_f
                k_mem = (k_mem + 1) % H_mem

        # --- F. Stagnation Check, Local Search & Restart ---
        # If population diversity is very low, we are likely stuck in a local optimum
        std_dev = np.std(fitness)
        if std_dev < 1e-8:
            
            # 1. Local Polish (Coordinate Descent) on the best solution
            ls_improved = True
            step_size = 0.05
            
            # Loop to refine solution until step size is negligible or time runs out
            while ls_improved and step_size > 1e-7:
                ls_improved = False
                if is_time_up(): break
                
                for d in range(dim):
                    if is_time_up(): break
                    
                    original_val = best_sol[d]
                    step = step_size * diff_b[d]
                    
                    # Try positive step
                    best_sol[d] = np.clip(original_val + step, min_b[d], max_b[d])
                    val = func(best_sol)
                    if val < best_fitness:
                        best_fitness = val
                        ls_improved = True
                        continue
                    
                    # Try negative step
                    best_sol[d] = np.clip(original_val - step, min_b[d], max_b[d])
                    val = func(best_sol)
                    if val < best_fitness:
                        best_fitness = val
                        ls_improved = True
                        continue
                    
                    # Revert
                    best_sol[d] = original_val
                
                # Reduce step size if no improvement in full pass
                if not ls_improved:
                    step_size *= 0.5
                    ls_improved = True # Force one more pass with smaller step
            
            # 2. Soft Restart
            # Keep the polished best solution, randomize the rest
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol # Elitism
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fitness
            
            # Re-evaluate new random population
            for i in range(1, pop_size):
                if is_time_up(): return best_fitness
                fitness[i] = func(pop[i])
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_sol = pop[i].copy()

    return best_fitness
