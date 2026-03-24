#Here is a fully operational, self-contained Python algorithm.
#
#This algorithm implements **L-SHADE (Success-History Adaptive Differential Evolution)** augmented with **Linear Population Size Reduction (LPSR)**, a **Stagnation-based Restart Mechanism**, and an integrated **MTS-LS1 (Multiple Trajectory Search)** local search.
#
#**Why this improves results:**
#1.  **L-SHADE w/ LPSR**: This is a state-of-the-art global optimizer (winner of multiple CEC competitions). It adapts mutation/crossover rates historically and reduces population size linearly to shift from exploration to exploitation.
#2.  **MTS-LS1 Local Search**: Unlike standard gradient descent (which fails on non-differentiable functions) or coordinate descent, MTS-LS1 adjusts search ranges per dimension dynamically. It is triggered whenever a new global best is found to immediately "drain" the basin of attraction.
#3.  **Aggressive Restarts**: If the population variance collapses (stagnation) or the population becomes too small while time remains, the algorithm saves the best solution and restarts the rest of the population to find new potential minima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with LPSR, Adaptive Restarts, 
    and MTS-LS1 Local Search.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    def get_time_ratio():
        """Returns elapsed time ratio (0.0 to 1.0)."""
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    def check_timelimit():
        """Returns True if time is up."""
        return datetime.now() - start_time >= time_limit

    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b

    # Global Best Tracking
    best_fitness = float('inf')
    best_pos = None

    # Wrapper for function evaluation to handle updates and time
    def evaluate(x):
        nonlocal best_fitness, best_pos
        val = func(x)
        if val < best_fitness:
            best_fitness = val
            best_pos = x.copy()
        return val

    # --- Parameters ---
    # Population Size: Standard L-SHADE starts with 18*dim, capped for performance
    initial_pop_size = int(np.clip(18 * dim, 40, 200)) 
    min_pop_size = 4
    
    # Archive parameters
    archive_size_rate = 2.0
    
    # Memory for adaptive parameters (History length H=5)
    mem_size = 5
    memory_sf = np.full(mem_size, 0.5)
    memory_scr = np.full(mem_size, 0.5)
    memory_pos = 0

    # MTS-LS1 Search Range initialization
    search_range = diff_b * 0.4

    # --- Initial Population ---
    pop_size = initial_pop_size
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if check_timelimit(): return best_fitness
        fitness[i] = evaluate(pop[i])

    archive = []

    # --- Main Loop ---
    while not check_timelimit():
        
        # 1. Population Size Reduction (LPSR)
        # Calculate target size based on time ratio
        progress = get_time_ratio()
        target_size = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * progress))
        
        if pop_size > target_size:
            n_reduce = pop_size - target_size
            # Remove worst individuals
            sorting_idx = np.argsort(fitness)
            # Keep best (start of sorted), remove worst (end of sorted)
            survivor_indices = sorting_idx[:-n_reduce]
            
            pop = pop[survivor_indices]
            fitness = fitness[survivor_indices]
            pop_size = target_size
            
            # Reduce archive size if necessary
            curr_arc_size = len(archive)
            target_arc_size = int(pop_size * archive_size_rate)
            if curr_arc_size > target_arc_size:
                del_indices = np.random.choice(curr_arc_size, curr_arc_size - target_arc_size, replace=False)
                # List comprehension is safer for deletion than numpy delete on list
                new_archive = [archive[i] for i in range(curr_arc_size) if i not in del_indices]
                archive = new_archive

        # 2. Restart Mechanism
        # Trigger if population variance is extremely low (stagnation) or size is minimal early
        pop_std = np.std(fitness)
        if (pop_std < 1e-8) or (pop_size <= min_pop_size and progress < 0.85):
            # Reset Population
            pop_size = initial_pop_size
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Inject Global Best
            if best_pos is not None:
                pop[0] = best_pos.copy()
            
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fitness
            
            # Evaluate new pop (skip index 0)
            for i in range(1, pop_size):
                if check_timelimit(): return best_fitness
                fitness[i] = evaluate(pop[i])
            
            # Reset Adaptive Memory
            memory_sf = np.full(mem_size, 0.5)
            memory_scr = np.full(mem_size, 0.5)
            archive = []
            search_range = diff_b * 0.4 # Reset local search step
            continue

        # 3. Generate Parameter Adaptation
        # r_idx: index from memory to use
        r_idx = np.random.randint(0, mem_size, pop_size)
        
        # Generate CR (Normal dist)
        cr = np.random.normal(memory_scr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy dist)
        f = np.random.standard_cauchy(pop_size) * 0.1 + memory_sf[r_idx]
        # Check constraints for F
        while np.any(f <= 0):
            neg_mask = f <= 0
            f[neg_mask] = np.random.standard_cauchy(np.sum(neg_mask)) * 0.1 + memory_sf[r_idx][neg_mask]
        f = np.clip(f, 0, 1)

        # 4. Mutation Strategy: current-to-pbest/1
        # Sort population for p-best selection
        sorted_indices = np.argsort(fitness)
        pop_sorted = pop[sorted_indices]
        
        # p is a random value between 2/pop_size and 0.2 (top 20%)
        # This adds robustness compared to fixed 0.11
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        num_pbest = max(2, int(p_val * pop_size))
        
        pbest_indices = np.random.randint(0, num_pbest, pop_size)
        x_pbest = pop_sorted[pbest_indices]
        
        # Archive usage
        if len(archive) > 0:
            archive_np = np.array(archive)
            pool = np.vstack((pop, archive_np))
        else:
            pool = pop
            
        # r1, r2 selection
        idxs = np.arange(pop_size)
        r1 = np.random.randint(0, pop_size, pop_size)
        # Ensure r1 != i
        conflict = (r1 == idxs)
        r1[conflict] = (r1[conflict] + 1) % pop_size
        x_r1 = pop[r1]
        
        r2 = np.random.randint(0, len(pool), pop_size)
        # Ideally r2 != r1 and r2 != i, simplified here for speed
        x_r2 = pool[r2]
        
        # Compute Mutant
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        f_col = f[:, None]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        # Force at least one dim from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[idxs, j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 6. Evaluation and Selection
        success_sf = []
        success_scr = []
        diff_f = []
        
        # Track if we improved global best this generation
        improved_global = False
        
        for i in range(pop_size):
            if check_timelimit(): return best_fitness
            
            # Evaluate trial
            f_trial = func(trial_pop[i])
            
            if f_trial < fitness[i]:
                # Improvement
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_pos = trial_pop[i].copy()
                    improved_global = True
                
                # Archive handling
                archive.append(pop[i].copy())
                
                # Record successful params
                success_sf.append(f[i])
                success_scr.append(cr[i])
                diff_f.append(fitness[i] - f_trial)
                
                # Update population
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
        
        # Maintain Archive Limit
        while len(archive) > int(pop_size * archive_size_rate):
            del archive[np.random.randint(0, len(archive))]
            
        # 7. Update Memories
        if len(success_sf) > 0:
            w_sf = np.array(success_sf)
            w_scr = np.array(success_scr)
            w_df = np.array(diff_f)
            
            # Weighted Lehmer Mean
            total_imp = np.sum(w_df)
            if total_imp > 0:
                weights = w_df / total_imp
                
                # Mean SF
                m_sf = np.sum(weights * (w_sf ** 2)) / np.sum(weights * w_sf)
                
                # Mean SCR
                m_scr = np.sum(weights * w_scr)
                
                memory_sf[memory_pos] = 0.5 * memory_sf[memory_pos] + 0.5 * m_sf
                memory_pos = (memory_pos + 1) % mem_size # Update position only for SF (L-SHADE specific)
                memory_scr[memory_pos] = 0.5 * memory_scr[memory_pos] + 0.5 * m_scr
            
        # 8. Integrated Local Search (MTS-LS1)
        # Triggered if global best improved OR we are in the final 5% of time
        if improved_global or progress > 0.95:
            if best_pos is not None:
                # Perform a simplified MTS-LS1 iteration on the best position
                # This explores dimensions independently and adjusts step sizes
                
                # Make a temporary working copy to avoid polluting best_pos prematurely
                current_best = best_pos.copy()
                current_f = best_fitness
                
                improved_ls = False
                
                # Shuffle dimensions to avoid bias
                dims = np.random.permutation(dim)
                
                for d in dims:
                    if check_timelimit(): break
                    
                    original_val = current_best[d]
                    
                    # Try Negative Step
                    current_best[d] = np.clip(original_val - search_range[d], min_b[d], max_b[d])
                    val = func(current_best)
                    
                    if val < current_f:
                        current_f = val
                        best_fitness = val
                        best_pos = current_best.copy() # Update global immediately
                        search_range[d] *= 1.5 # Expand search range
                        improved_ls = True
                    else:
                        # Try Positive Step (0.5 SR from original is implicit in MTS logic, 
                        # but here we do explicit bidirectional check for clarity/robustness)
                        # Actually MTS-LS1 logic: if neg fails, try pos from original
                        current_best[d] = np.clip(original_val + 0.5 * search_range[d], min_b[d], max_b[d])
                        val = func(current_best)
                        
                        if val < current_f:
                            current_f = val
                            best_fitness = val
                            best_pos = current_best.copy()
                            search_range[d] *= 1.5
                            improved_ls = True
                        else:
                            # Neither direction worked, restore and shrink
                            current_best[d] = original_val
                            search_range[d] *= 0.5
                
                # Clamp search range to avoid underflow
                search_range = np.clip(search_range, 1e-15, diff_b)

    return best_fitness
