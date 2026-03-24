#Here is the improved, fully operational, and self-contained algorithm.
#
#This implementation enhances the **Differential Evolution (DE)** approach by:
#1.  Using a **"Best/1/bin" mutation strategy** for faster convergence compared to random strategies.
#2.  Integrating a **Local Search (Pattern Search/Coordinate Descent)** phase at the end of each restart cycle. This "polishes" the best solution found, significantly improving precision (exploiting the local basin).
#3.  Maintaining **Latin Hypercube Sampling (LHS)** and **Restarts** to ensure global exploration and prevent getting stuck in local optima.
#4.  Utilizing **Vectorized Operations** for efficiency.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using a Hybrid Algorithm:
    1. Global Search: Differential Evolution (DE/best/1/bin) with Restarts & LHS.
    2. Local Search: Coordinate Descent (Pattern Search) for final polishing.
    """
    
    # --- Time Management ---
    start_time = time.time()
    # Reserve a small buffer (0.05s) to ensure safe return before timeout
    time_limit = max_time - 0.05
    
    def is_time_up():
        return (time.time() - start_time) >= time_limit

    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    domain_width = ub - lb
    
    # Global best tracker
    global_best_val = float('inf')
    global_best_pos = None
    
    # --- Configuration ---
    # Population size: adaptive but clamped for speed (20-60 is usually optimal for speed/quality)
    pop_size = int(np.clip(10 * dim, 20, 60))
    
    # --- Main Loop (Restarts) ---
    while True:
        if is_time_up(): return global_best_val
        
        # 1. Initialization: Latin Hypercube Sampling (LHS)
        # Stratifies samples to cover the domain evenly
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            r = np.random.rand(pop_size)
            pop[:, d] = lb[d] + (perm + r) / pop_size * domain_width[d]
        
        # Evaluate Initial Population
        fitness = np.full(pop_size, float('inf'))
        current_best_idx = 0
        current_best_val = float('inf')
        
        for i in range(pop_size):
            if is_time_up(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            
            # Track best in current population
            if val < current_best_val:
                current_best_val = val
                current_best_idx = i
            
            # Track global best
            if val < global_best_val:
                global_best_val = val
                global_best_pos = pop[i].copy()

        # 2. Differential Evolution Phase (DE/best/1/bin)
        # Exploitative strategy to converge quickly to a basin
        stagnation_counter = 0
        prev_best_val = current_best_val
        
        while True:
            if is_time_up(): return global_best_val
            
            # Adaptive parameters (Randomized per generation)
            # F (Mutation factor): 0.5 to 0.9
            F = 0.5 + 0.4 * np.random.rand()
            # CR (Crossover probability): 0.8 to 1.0 (High CR for fast convergence)
            CR = 0.8 + 0.2 * np.random.rand()
            
            # Generate Mutation Indices
            # Strategy: V = Best + F * (r1 - r2)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Vectorized Mutation
            best_vec = pop[current_best_idx]
            mutants = best_vec + F * (pop[r1] - pop[r2])
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(pop_size, dim) < CR
            # Ensure at least one dimension is mutated
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trials = np.where(cross_mask, mutants, pop)
            trials = np.clip(trials, lb, ub)
            
            # Selection
            improved = False
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                f_trial = func(trials[i])
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_pos = trials[i].copy()
                
                # Greedy Selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trials[i]
                    improved = True
                    
                    if f_trial < current_best_val:
                        current_best_val = f_trial
                        current_best_idx = i
            
            # Convergence Detection
            if current_best_val < prev_best_val:
                prev_best_val = current_best_val
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Stop if population variance is low or stagnated
            if np.std(fitness) < 1e-6 or stagnation_counter > 15:
                break
        
        # 3. Local Search Phase (Coordinate Descent / Pattern Search)
        # Polishes the best solution found so far to high precision
        if global_best_pos is not None:
            # Create a working copy of the best position
            ls_pos = global_best_pos.copy()
            ls_val = global_best_val
            
            # Initial step size: 5% of the domain width
            step_sizes = domain_width * 0.05
            
            # Iterate while step size is significant
            while np.max(step_sizes) > 1e-8:
                if is_time_up(): return global_best_val
                
                improved_ls = False
                # Randomize dimension order to avoid bias
                dims_order = np.random.permutation(dim)
                
                for d in dims_order:
                    if is_time_up(): return global_best_val
                    
                    origin = ls_pos[d]
                    step = step_sizes[d]
                    
                    # Try positive step
                    ls_pos[d] = np.clip(origin + step, lb[d], ub[d])
                    val = func(ls_pos)
                    
                    if val < ls_val:
                        ls_val = val
                        if val < global_best_val:
                            global_best_val = val
                            global_best_pos = ls_pos.copy()
                        improved_ls = True
                        continue
                    
                    # Try negative step
                    ls_pos[d] = np.clip(origin - step, lb[d], ub[d])
                    val = func(ls_pos)
                    
                    if val < ls_val:
                        ls_val = val
                        if val < global_best_val:
                            global_best_val = val
                            global_best_pos = ls_pos.copy()
                        improved_ls = True
                    else:
                        # Revert if no improvement
                        ls_pos[d] = origin
                
                # Shrink step size if no improvement found in any dimension
                if not improved_ls:
                    step_sizes *= 0.5
            
    return global_best_val
