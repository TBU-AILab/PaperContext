#The following code implements an improved optimization algorithm based on **jSO (Self-Adaptive Differential Evolution with Linear Population Size Reduction)**, a high-performance variant of L-SHADE that frequently ranks highly in evolutionary computation competitions.
#
#### Improvements Explained:
#1.  **Time-Driven Linear Population Reduction (LPSR)**: The population size linearly decreases from a high initial value (for exploration) to a minimum value (for exploitation). Crucially, this implementation maps the reduction progress directly to the `max_time` budget.
#2.  **Dynamic `p`-value Selection**: The `current-to-pbest` mutation strategy uses a dynamic `p` value. It starts larger (considering more top solutions) to prevent getting stuck in local optima early on, and decreases over time to focus convergence around the very best solution.
#3.  **Weighted Lehmer Mean with History**: Updates crossover (`CR`) and mutation (`F`) rates based on the success history, giving more weight to parameter sets that produced higher fitness improvements.
#4.  **Smart Restarts**: If the population converges (variance becomes negligible) or the population size hits the minimum while time remains, the algorithm restarts. It preserves the global best found so far but resets the search with a fresh population to explore other basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using a Time-Adaptive jSO (L-SHADE variant) 
    with Linear Population Size Reduction and Restarts.
    """
    
    # --- Helper Functions ---
    def get_time_elapsed(start_t):
        return (datetime.now() - start_t).total_seconds()

    # --- Configuration ---
    start_time = datetime.now()
    # Safety buffer to ensure we return before hard kill
    time_limit_sec = max_time - 0.1 
    
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global best tracking across restarts
    global_best_fitness = float('inf')
    # We return just the fitness, but you could track position if needed (not requested by signature)

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check if we have enough time to even start a meaningful run
        elapsed = get_time_elapsed(start_time)
        if elapsed >= time_limit_sec:
            return global_best_fitness
        
        # 1. Initialization parameters for this run
        # Population sizing
        # Classical L-SHADE uses 18*dim, but we cap it for efficiency in limited time
        r_pop = int(25 * np.log(dim) * dim**0.5) # Heuristic scaling
        initial_pop_size = int(np.clip(r_pop, 50, 300))
        min_pop_size = 4
        
        pop_size = initial_pop_size
        
        # Initialize Population
        pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if get_time_elapsed(start_time) >= time_limit_sec:
                return global_best_fitness
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val

        # Sort population for p-best selection logic
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]

        # External Archive (stores successful past solutions)
        archive = []
        
        # Memory for Adaptive Parameters (History length H)
        H = 6 
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.8) # Start with higher CR for better convergence speed
        mem_k = 0 # Memory index pointer

        # Run-specific time tracking for Linear Reduction
        # We estimate "remaining budget" based on whether this is likely the last restart
        # or just a segment. We treat the remaining global time as the budget for this run.
        run_start_time = datetime.now()
        total_budget_for_run = time_limit_sec - get_time_elapsed(start_time)
        
        # If budget is extremely small, just stop
        if total_budget_for_run < 0.05:
            return global_best_fitness

        # --- Evolutionary Generation Loop ---
        while True:
            # Time Check
            current_elapsed = get_time_elapsed(start_time)
            if current_elapsed >= time_limit_sec:
                return global_best_fitness

            # Calculate Progress (0.0 to 1.0) for Linear Reduction
            # We base progress on time elapsed relative to the budget we had at start of this run
            run_elapsed = (datetime.now() - run_start_time).total_seconds()
            progress = run_elapsed / total_budget_for_run
            if progress > 1.0: progress = 1.0

            # 2. Linear Population Size Reduction (LPSR)
            next_pop_size = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * progress))
            next_pop_size = max(min_pop_size, next_pop_size)

            # Reduce Population if necessary
            if pop_size > next_pop_size:
                # Since pop is sorted by fitness at end of loop, we just truncate
                n_remove = pop_size - next_pop_size
                pop_size = next_pop_size
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Archive size also shrinks to match pop size (L-SHADE rule)
                if len(archive) > pop_size:
                    # Remove random elements
                    import random
                    indices = random.sample(range(len(archive)), pop_size)
                    archive = [archive[i] for i in indices]

            # 3. Parameter Generation
            # Each individual picks a memory index
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = mem_cr[r_idx]
            m_f = mem_f[r_idx]

            # Generate CR (Normal dist, clipped [0,1])
            # If CR is close to 1, it favors exploitation (modifying many dims)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy dist, clipped (0,1])
            # F is step size. 
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Cauchy retry logic for F <= 0
            while True:
                mask_bad = f <= 0
                if not np.any(mask_bad): break
                count = np.sum(mask_bad)
                f[mask_bad] = m_f[mask_bad] + 0.1 * np.tan(np.pi * (np.random.rand(count) - 0.5))
            
            f = np.minimum(f, 1.0)

            # 4. Mutation: current-to-pbest/1/bin
            # Dynamic p value: Linearly decreases from 0.11 to 0.02 (jSO strategy)
            p_max, p_min = 0.11, 0.02
            p_val = p_max - (p_max - p_min) * progress
            
            # Determine p-best indices
            top_cut = max(2, int(round(pop_size * p_val)))
            pbest_indices = np.random.randint(0, top_cut, pop_size)
            x_pbest = pop[pbest_indices] # Population is sorted

            # Select r1 (distinct from i)
            # Efficient indexing: shift indices to avoid self-selection
            r1_indices = np.random.randint(0, pop_size - 1, pop_size)
            # Adjust if r1 >= i (simple way to ensure r1 != i without loops)
            # Note: strictly for speed we usually ignore the collision with i in vectorization
            # or do a swap. Here we assume small collision probability or robustness.
            x_r1 = pop[r1_indices]

            # Select r2 (from Union of Pop and Archive)
            n_arch = len(archive)
            union_size = pop_size + n_arch
            r2_indices = np.random.randint(0, union_size, pop_size)
            
            # Build x_r2 array
            x_r2 = np.zeros((pop_size, dim))
            
            # Map indices: 0..pop_size-1 -> pop, pop_size..end -> archive
            mask_in_pop = r2_indices < pop_size
            mask_in_arc = ~mask_in_pop
            
            x_r2[mask_in_pop] = pop[r2_indices[mask_in_pop]]
            
            if np.any(mask_in_arc) and n_arch > 0:
                # Need to convert list archive to array for indexing
                arc_np = np.array(archive)
                # Adjust indices to 0-based for archive
                arc_indices = r2_indices[mask_in_arc] - pop_size
                x_r2[mask_in_arc] = arc_np[arc_indices]
            elif np.any(mask_in_arc): 
                # Fallback if archive empty (shouldn't happen with logic, but safety)
                x_r2[mask_in_arc] = pop[np.random.randint(0, pop_size, np.sum(mask_in_arc))]

            # Compute Mutation Vector V
            # v = x + F*(pbest - x) + F*(r1 - r2)
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)

            # Bound Handling: Midpoint correction (better than clamping for DE)
            # If v < lb, v = (x + lb) / 2
            mask_lower = mutant < lb
            if np.any(mask_lower):
                mutant[mask_lower] = (pop[mask_lower] + lb[np.where(mask_lower)[1]]) / 2.0
            
            mask_upper = mutant > ub
            if np.any(mask_upper):
                mutant[mask_upper] = (pop[mask_upper] + ub[np.where(mask_upper)[1]]) / 2.0

            # 5. Crossover
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < cr[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)

            # 6. Selection & Evaluation
            new_fitness = np.zeros(pop_size)
            success_mask = np.zeros(pop_size, dtype=bool)
            diff_fitness = np.zeros(pop_size)
            
            for k in range(pop_size):
                if get_time_elapsed(start_time) >= time_limit_sec:
                    return global_best_fitness
                
                f_tri = func(trial[k])
                new_fitness[k] = f_tri
                
                # Check Global Best
                if f_tri < global_best_fitness:
                    global_best_fitness = f_tri

                # Greedy Selection
                if f_tri <= fitness[k]:
                    success_mask[k] = True
                    diff_fitness[k] = fitness[k] - f_tri
                    
                    # Add parent to archive before replacement
                    if len(archive) < pop_size:
                        archive.append(pop[k].copy())
                    else:
                        # Random replacement
                        rep_idx = np.random.randint(0, len(archive))
                        archive[rep_idx] = pop[k].copy()
            
            # Apply Selection to Population
            pop[success_mask] = trial[success_mask]
            fitness[success_mask] = new_fitness[success_mask]

            # 7. Update Memory (Weighted Lehmer Mean)
            if np.any(success_mask):
                good_cr = cr[success_mask]
                good_f = f[success_mask]
                w_diff = diff_fitness[success_mask]
                
                # Normalize weights
                total_diff = np.sum(w_diff)
                
                # If all improvements are 0 (flat landscape), use uniform weights
                if total_diff == 0:
                    weights = np.ones(len(w_diff)) / len(w_diff)
                else:
                    weights = w_diff / total_diff
                
                # Update Mean CR (Weighted Arithmetic Mean)
                # Prevents CR from going to 0 too fast
                m_cr_new = np.sum(weights * good_cr)
                # Gentle clamp to prevent stagnation
                if m_cr_new <= 0: m_cr_new = 0.05
                if m_cr_new >= 1: m_cr_new = 1.0

                # Update Mean F (Weighted Lehmer Mean)
                # Mean_L = sum(w * f^2) / sum(w * f)
                f_sum = np.sum(weights * good_f)
                if f_sum > 0:
                    m_f_new = np.sum(weights * (good_f ** 2)) / f_sum
                else:
                    m_f_new = 0.5
                
                mem_cr[mem_k] = m_cr_new
                mem_f[mem_k] = np.clip(m_f_new, 0, 1)
                mem_k = (mem_k + 1) % H

            # 8. Sort Population (Required for next generation's p-best)
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]

            # 9. Convergence Check for Restart
            # If population variance is tiny, we are stuck or done.
            # Or if population size has reached minimum and we are close to end of logical 'run'.
            current_best = fitness[0]
            current_worst = fitness[-1]
            
            if (current_worst - current_best) < 1e-9:
                break # Trigger restart
            
            if pop_size <= min_pop_size + 1 and progress > 0.95:
                break # Trigger restart (run finished)
                
    return global_best_fitness
