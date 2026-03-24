#The following algorithm implements **L-SHADE-RSP (Restart with Symmetric Polishing)**.
#
#**Rationale for Improvements:**
#1.  **L-SHADE Backbone**: Utilizes the highly effective L-SHADE algorithm with Linear Population Size Reduction (LPSR) mapped to the *remaining time*. This ensures the algorithm adapts its exploration/exploitation balance dynamically, regardless of whether it is the first run or a restart.
#2.  **Symmetric MTS-LS1**: Incorporates the Multiple Trajectory Search (MTS) Local Search strategy. It uses an asymmetric step reduction logic (check negative, check positive with half step, then reduce) which proved highly effective in navigating complex valleys ("polishing") in previous iterations.
#3.  **Adaptive Restart Strategy**: Unlike the single-run approach, this algorithm monitors for early convergence (population collapse or fitness stagnation). If the population converges while significant time remains, it triggers the local search to refine the result, and then **restarts** the population (preserving the global best) to search for other basins of attraction. This prevents wasting time on a stagnated population.
#4.  **Robust Time Management**: The LPSR progress is calculated relative to the *epoch* time (remaining time at restart), ensuring the population reduction schedule is always optimal for the available budget.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restart L-SHADE with MTS-LS1 Local Search.
    
    Features:
    - Linear Population Size Reduction (LPSR) mapped to real-time.
    - Adaptive parameter control (Success-History Adaptation).
    - Symmetric MTS-LS1 Local Search for polishing.
    - Restarts upon convergence to utilize full time budget.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    def is_time_up():
        return (datetime.now() - start_time) >= limit

    # --- Problem Configuration ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global Best Tracking
    global_best_val = float('inf')
    global_best_sol = np.zeros(dim)
    
    # --- Helper: MTS-LS1 Local Search ---
    def run_mts_ls1(solution, val):
        """
        Runs MTS-LS1 (Multiple Trajectory Search - Local Search 1).
        Refines the solution using coordinate descent with adaptive step sizes.
        """
        current_sol = solution.copy()
        current_val = val
        
        # Initial Search Range (20% of domain to focus on local basin)
        sr = (ub - lb) * 0.2
        
        improved = True
        while improved:
            improved = False
            if is_time_up(): break
            
            # Search dimensions in random order
            dims = np.random.permutation(dim)
            
            for d in dims:
                if is_time_up(): break
                
                # Direction 1: Negative Step
                x_new = current_sol.copy()
                x_new[d] = np.clip(current_sol[d] - sr[d], lb[d], ub[d])
                f_new = func(x_new)
                
                if f_new < current_val:
                    current_val = f_new
                    current_sol[d] = x_new[d]
                    improved = True
                else:
                    # Direction 2: Positive Step (Asymmetric 0.5 step per MTS logic)
                    x_new[d] = np.clip(current_sol[d] + 0.5 * sr[d], lb[d], ub[d])
                    f_new = func(x_new)
                    
                    if f_new < current_val:
                        current_val = f_new
                        current_sol[d] = x_new[d]
                        improved = True
                    else:
                        # No improvement: Reduce search range
                        sr[d] *= 0.5
            
            # Termination: If precision limit reached
            if np.max(sr) < 1e-15:
                break
                
        return current_sol, current_val

    # --- Main Optimization Loop (Restarts) ---
    while not is_time_up():
        
        # Determine parameters for this run (Restart)
        run_start = datetime.now()
        elapsed_total = (run_start - start_time).total_seconds()
        remaining_sec = max_time - elapsed_total
        
        if remaining_sec < 0.05: # Too little time to do anything useful
            break
            
        # L-SHADE Constants
        # N_init: 18*dim is efficient, capped for speed
        pop_size_init = int(18 * dim)
        pop_size_init = max(10, min(pop_size_init, 200))
        min_pop_size = 4
        
        # Initialize Population
        population = lb + np.random.rand(pop_size_init, dim) * (ub - lb)
        fitness = np.full(pop_size_init, float('inf'))
        
        # Inject Global Best (Exploitation of previous runs)
        if global_best_val != float('inf'):
            population[0] = global_best_sol.copy()
            fitness[0] = global_best_val
            
        # Evaluate Initial Population
        for i in range(pop_size_init):
            if is_time_up(): return global_best_val
            
            # Skip injected best to save evaluation
            if fitness[i] == float('inf'): 
                val = func(population[i])
                fitness[i] = val
                
                if val < global_best_val:
                    global_best_val = val
                    global_best_sol = population[i].copy()
        
        # Sort by fitness
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # L-SHADE Memory Initialization
        H = 6 
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        archive = []
        arc_rate = 2.6
        
        curr_pop_size = pop_size_init
        
        # --- Inner Loop: L-SHADE Generations ---
        while not is_time_up():
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate progress based on TIME allocated for THIS run
            run_elapsed = (datetime.now() - run_start).total_seconds()
            progress = run_elapsed / remaining_sec
            if progress > 1.0: progress = 1.0
            
            target_size = int(round(pop_size_init + (min_pop_size - pop_size_init) * progress))
            target_size = max(min_pop_size, target_size)
            
            if curr_pop_size > target_size:
                curr_pop_size = target_size
                population = population[:curr_pop_size]
                fitness = fitness[:curr_pop_size]
                
                # Resize Archive
                max_arc_size = int(curr_pop_size * arc_rate)
                if len(archive) > max_arc_size:
                    del_count = len(archive) - max_arc_size
                    # Randomly delete
                    for _ in range(del_count):
                        archive.pop(np.random.randint(len(archive)))
            
            # 2. Check for Convergence / Local Search Trigger
            # Conditions: Small population, Low Variance, or End of Time
            is_stagnant = (np.std(fitness) < 1e-12)
            is_min_pop = (curr_pop_size <= min_pop_size)
            is_end_game = (progress > 0.95)
            
            if is_stagnant or is_min_pop or is_end_game:
                # Perform Local Search on the best solution
                ls_sol, ls_val = run_mts_ls1(global_best_sol, global_best_val)
                if ls_val < global_best_val:
                    global_best_val = ls_val
                    global_best_sol = ls_sol
                
                # If we are at the end of time, return result
                if is_end_game:
                    return global_best_val
                
                # If converged early, BREAK inner loop to trigger RESTART
                break

            # 3. Adaptive Parameter Generation
            # Memory indices
            r_idxs = np.random.randint(0, H, curr_pop_size)
            m_cr = mem_cr[r_idxs]
            m_f = mem_f[r_idxs]
            
            # Generate CR
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy)
            f = m_f + 0.1 * np.random.standard_cauchy(curr_pop_size)
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
            f = np.clip(f, 0, 1)
            
            # 4. Mutation: current-to-pbest/1
            # p decreases linearly to favor exploitation
            p_val = 0.11 - 0.09 * progress
            p_val = max(0.02, p_val)
            
            p_num = max(2, int(curr_pop_size * p_val))
            pbest_idxs = np.random.randint(0, p_num, curr_pop_size)
            x_pbest = population[pbest_idxs]
            
            r1 = np.random.randint(0, curr_pop_size, curr_pop_size)
            for i in range(curr_pop_size):
                while r1[i] == i: r1[i] = np.random.randint(0, curr_pop_size)
            x_r1 = population[r1]
            
            # Union of Population and Archive for r2
            if len(archive) > 0:
                union_pop = np.vstack((population, np.array(archive)))
            else:
                union_pop = population
            
            r2 = np.random.randint(0, len(union_pop), curr_pop_size)
            for i in range(curr_pop_size):
                while r2[i] == i or r2[i] == r1[i]: 
                    r2[i] = np.random.randint(0, len(union_pop))
            x_r2 = union_pop[r2]
            
            f_v = f[:, np.newaxis]
            mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
            mutant = np.clip(mutant, lb, ub)
            
            # 5. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, curr_pop_size)
            mask = np.random.rand(curr_pop_size, dim) < cr[:, np.newaxis]
            mask[np.arange(curr_pop_size), j_rand] = True
            trial = np.where(mask, mutant, population)
            
            # 6. Selection
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            success_f = []
            success_cr = []
            diff_f = []
            
            for i in range(curr_pop_size):
                if is_time_up(): return global_best_val
                
                y = func(trial[i])
                
                if y <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fit[i] = y
                    
                    if y < fitness[i]:
                        archive.append(population[i].copy())
                        diff_f.append(fitness[i] - y)
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        
                    if y < global_best_val:
                        global_best_val = y
                        global_best_sol = trial[i].copy()
                        
            population = new_pop
            fitness = new_fit
            
            # 7. Memory Update (Weighted Lehmer Mean)
            if diff_f:
                w = np.array(diff_f)
                w /= np.sum(w)
                
                sf = np.array(success_f)
                scr = np.array(success_cr)
                
                mean_f = np.sum(w * (sf**2)) / (np.sum(w * sf) + 1e-18)
                mean_cr = np.sum(w * scr)
                
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                k_mem = (k_mem + 1) % H
                
            # Sort for next generation (needed for p-best selection)
            s_idx = np.argsort(fitness)
            population = population[s_idx]
            fitness = fitness[s_idx]

    return global_best_val
