#Here is a fully operational, self-contained Python algorithm.
#
#This algorithm implements **L-SHADE-EpSin-MTS**, a hybrid evolutionary algorithm that combines:
#1.  **L-SHADE**: An adaptive Differential Evolution algorithm with Linear Population Size Reduction (LPSR) and historical memory for parameters ($F$ and $Cr$).
#2.  **MTS-LS1**: A robust Local Search mechanism (from Multiple Trajectory Search) used to "drain the basin" of the best solution when stagnation is detected.
#3.  **Adaptive Restart**: Instead of running once, the algorithm detects convergence (stagnation) and restarts the population while keeping the elite solution. This allows it to escape local optima in multimodal landscapes.
#4.  **Episodic Progress**: The population reduction is tied to global time, but restarts allow "fresh" exploration phases that become increasingly aggressive (exploitative) as time runs out.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using a Restart-based L-SHADE with MTS-LS1 Local Search.
    
    Mechanism:
    1. Global Time Management: Tracks overall progress to adjust population size linearly.
    2. L-SHADE Core: Adapts F and CR parameters using success history.
    3. Stagnation Detection: If fitness variance drops or no improvement for N gens, triggers LS.
    4. MTS-LS1: Performs a trajectory-based local search to refine the best solution.
    5. Soft Restart: Keeps global best, re-initializes rest of population to find better basins.
    """
    
    # --- Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Bound processing
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_fitness = float('inf')
    global_best_pos = None

    # --- Helper Functions ---
    def check_timeout():
        """Checks if the allocated time has expired."""
        return datetime.now() - start_time >= time_limit

    def evaluate(x):
        """Evaluates x, updates global best, handles boundary clipping."""
        nonlocal global_best_fitness, global_best_pos
        
        # Clip to bounds
        x_c = np.clip(x, min_b, max_b)
        val = func(x_c)
        
        if val < global_best_fitness:
            global_best_fitness = val
            global_best_pos = x_c.copy()
            
        return val, x_c

    def perform_mts_ls1(center_x, center_f, search_range):
        """
        MTS-LS1 (Multiple Trajectory Search - Local Search 1).
        Robust derivative-free local search to refine the solution.
        """
        best_x = center_x.copy()
        best_f = center_f
        
        # Limit the local search steps to prevent hogging time
        max_ls_iter = 50 
        ls_iter = 0
        
        # Randomize dimension order
        dims = np.random.permutation(dim)
        
        while ls_iter < max_ls_iter:
            if check_timeout(): break
            ls_iter += 1
            improved_in_iter = False
            
            for i in dims:
                if check_timeout(): break
                
                original_val = best_x[i]
                sr = search_range[i]
                
                # Try Negative Step
                best_x[i] = np.clip(original_val - sr, min_b[i], max_b[i])
                val, _ = evaluate(best_x)
                
                if val < best_f:
                    best_f = val
                    improved_in_iter = True
                else:
                    # Try Positive Step (0.5 * SR)
                    best_x[i] = np.clip(original_val + 0.5 * sr, min_b[i], max_b[i])
                    val, _ = evaluate(best_x)
                    
                    if val < best_f:
                        best_f = val
                        improved_in_iter = True
                    else:
                        # Restore and shrink search range
                        best_x[i] = original_val
                        search_range[i] *= 0.5
            
            if not improved_in_iter:
                # If a full pass over dimensions yielded no improvement, stop LS
                break
                
        return best_x, best_f

    # --- Main Optimization Loop (Multi-start) ---
    while not check_timeout():
        
        # 1. Initialize L-SHADE Population
        # Linear Population Size Reduction (LPSR) parameters
        # Initial size: 18*D (standard), clipped to reasonable limits
        initial_pop_size = int(np.clip(18 * dim, 30, 200))
        min_pop_size = 4
        
        pop_size = initial_pop_size
        
        # Generate random population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Inject Global Best (Elitism for Restart)
        start_idx = 0
        if global_best_pos is not None:
            pop[0] = global_best_pos.copy()
            fitness[0], _ = evaluate(pop[0])
            start_idx = 1
            
        # Evaluate initial population
        for i in range(start_idx, pop_size):
            if check_timeout(): return global_best_fitness
            fitness[i], pop[i] = evaluate(pop[i])
            
        # SHADE Adaptive Memory
        H = 6 # Memory size
        mem_F = np.full(H, 0.5)
        mem_CR = np.full(H, 0.5)
        k_mem = 0
        
        # External Archive
        archive = []
        arc_rate = 2.0 # Archive size relative to pop_size
        
        # MTS Search Range (Initialized to 40% of domain)
        mts_search_range = diff_b * 0.4
        
        # Stagnation Counters
        stagnation_count = 0
        last_best_fit = np.min(fitness)
        
        # --- Epoch Loop ---
        while not check_timeout():
            
            # A. Calculate Progress (0.0 to 1.0) for LPSR
            elapsed_sec = (datetime.now() - start_time).total_seconds()
            progress = elapsed_sec / max(max_time, 1e-3)
            
            # B. Linear Population Size Reduction
            # Calculate target size based on global time progress
            target_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                n_remove = pop_size - target_size
                # Sort and remove worst
                sorted_indices = np.argsort(fitness)
                pop = pop[sorted_indices[:-n_remove]]
                fitness = fitness[sorted_indices[:-n_remove]]
                pop_size = target_size
                
                # Resize archive
                arc_limit = int(pop_size * arc_rate)
                if len(archive) > arc_limit:
                    archive = archive[:arc_limit]
            
            # C. Stagnation Check
            curr_best = np.min(fitness)
            if np.abs(curr_best - last_best_fit) < 1e-8:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = curr_best
            
            # Trigger Restart if stagnant or converged
            # Conditions: Variance too low OR No improvement for 30 gens OR Pop too small
            if np.std(fitness) < 1e-9 or stagnation_count > 30 or pop_size <= min_pop_size:
                # Perform deep local search on the best candidate before restarting
                best_idx = np.argmin(fitness)
                perform_mts_ls1(pop[best_idx], fitness[best_idx], mts_search_range)
                break # Break inner loop -> Trigger Restart
                
            # D. Parameter Adaptation (SHADE)
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_F[r_idx]
            m_cr = mem_CR[r_idx]
            
            # Generate CR (Normal)
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0, 1)
            
            # Generate F (Cauchy)
            F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Retry if F <= 0
            while np.any(F <= 0):
                neg = F <= 0
                F[neg] = m_f[neg] + 0.1 * np.random.standard_cauchy(np.sum(neg))
            F = np.minimum(F, 1.0)
            
            # E. Mutation (current-to-pbest/1)
            # p varies from 0.2 (exploration) to 0.1 (exploitation) based on progress
            p_val = max(2.0/pop_size, 0.2 - 0.1 * progress)
            n_pbest = int(max(1, p_val * pop_size))
            
            sorted_idx = np.argsort(fitness)
            pbest_indices = sorted_idx[:n_pbest]
            # Randomly select one pbest for each individual
            pbest_selection = pbest_indices[np.random.randint(0, n_pbest, pop_size)]
            x_pbest = pop[pbest_selection]
            
            # Select r1
            r1 = np.random.randint(0, pop_size, pop_size)
            conflict = r1 == np.arange(pop_size)
            r1[conflict] = (r1[conflict] + 1) % pop_size
            x_r1 = pop[r1]
            
            # Select r2 (Union of Pop and Archive)
            if len(archive) > 0:
                pool = np.vstack((pop, np.array(archive)))
            else:
                pool = pop
            
            r2 = np.random.randint(0, len(pool), pop_size)
            # Simple conflict resolution
            conflict2 = (r2 == r1) | (r2 == np.arange(pop_size))
            r2[conflict2] = (r2[conflict2] + 1) % len(pool)
            x_r2 = pool[r2]
            
            # Compute Mutant
            diff1 = x_pbest - pop
            diff2 = x_r1 - x_r2
            mutant = pop + F[:, None] * diff1 + F[:, None] * diff2
            
            # F. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            mask = np.random.rand(pop_size, dim) <= CR[:, None]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # G. Selection
            success_F = []
            success_CR = []
            diff_fitness = []
            
            new_pop = pop.copy()
            new_fitness_arr = fitness.copy()
            
            for i in range(pop_size):
                if check_timeout(): return global_best_fitness
                
                f_trial, t_x = evaluate(trial[i])
                
                if f_trial < fitness[i]:
                    new_pop[i] = t_x
                    new_fitness_arr[i] = f_trial
                    
                    archive.append(pop[i].copy())
                    success_F.append(F[i])
                    success_CR.append(CR[i])
                    diff_fitness.append(fitness[i] - f_trial)
            
            pop = new_pop
            fitness = new_fitness_arr
            
            # Maintain Archive Size
            arc_limit = int(pop_size * arc_rate)
            while len(archive) > arc_limit:
                archive.pop(np.random.randint(0, len(archive)))
                
            # H. Update Memory (Weighted Lehmer Mean)
            if len(diff_fitness) > 0:
                weights = np.array(diff_fitness)
                weights /= np.sum(weights)
                
                sF = np.array(success_F)
                sCR = np.array(success_CR)
                
                mean_scr = np.sum(weights * sCR)
                mean_sf = np.sum(weights * (sF**2)) / (np.sum(weights * sF) + 1e-15)
                
                mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * mean_scr
                mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * mean_sf
                k_mem = (k_mem + 1) % H

    return global_best_fitness
