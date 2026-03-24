#An improved algorithm based on **Restart-L-SHADE** with **MTS-LS1 Local Search**.
#
#**Rationale:**
#1.  **L-SHADE Core**: The base optimizer is L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution), a state-of-the-art algorithm for continuous optimization. It features adaptive control of mutation ($F$) and crossover ($CR$) parameters and linearly reduces population size to transition from global exploration to exploitation.
#2.  **Restart Mechanism**: Standard evolutionary algorithms can stagnate in local optima. This algorithm monitors population convergence (variance). When converged, it triggers a **Restart**, re-initializing the population to explore different basins of attraction while preserving the global best solution.
#3.  **MTS-LS1 Local Search**: Before restarting (or when the population is very small), a coordinate-descent-based local search (MTS-LS1) is applied to the best-so-far solution. This "polishing" step significantly refines the precision of the result, often by several orders of magnitude.
#4.  **Robust Time Management**: The algorithm checks the time limit strictly within every evaluation loop to ensure it returns the best result immediately when time is up.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restart-L-SHADE with MTS-LS1 Local Search.
    
    1. L-SHADE: Main evolutionary loop with adaptive parameters and population reduction.
    2. MTS-LS1: Local search applied to the best solution upon convergence.
    3. Restart: If the population converges, restart search to find better basins.
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
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # --- Algorithm Parameters ---
    # L-SHADE Constants
    pop_size_init_base = int(max(30, 20 * dim))
    min_pop_size = 4
    
    # Main Loop (Restarts)
    while not is_time_up():
        
        # --- Initialization (New Restart) ---
        # Reset population size
        curr_pop_size = pop_size_init_base
        
        # Initialize Population (Uniform Random)
        population = lb + np.random.rand(curr_pop_size, dim) * (ub - lb)
        fitness = np.full(curr_pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(curr_pop_size):
            if is_time_up(): return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = population[i].copy()
                
        # Sort by fitness
        sorted_idxs = np.argsort(fitness)
        population = population[sorted_idxs]
        fitness = fitness[sorted_idxs]
        
        # Reset Memory for L-SHADE
        H = 5 # History size
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # Local Search Step Size (MTS-LS1) initialization
        # Start with half the domain size
        sr = (ub - lb) * 0.5
        
        # Generation Counter
        gen = 0
        max_gens_estimate = 2000 # Heuristic for linear reduction scaling
        
        # --- L-SHADE Optimization Loop ---
        while not is_time_up():
            gen += 1
            
            # 1. Linear Population Size Reduction (LPSR)
            # Reduce population based on progress estimate
            progress = min(1.0, gen / max_gens_estimate)
            target_size = int(round(pop_size_init_base * (1.0 - progress) + min_pop_size * progress))
            target_size = max(min_pop_size, target_size)
            
            if curr_pop_size > target_size:
                # Truncate the worst (since population is sorted)
                curr_pop_size = target_size
                population = population[:curr_pop_size]
                fitness = fitness[:curr_pop_size]
                
                # Resize Archive
                if len(archive) > curr_pop_size:
                    # Randomly remove elements
                    del_count = len(archive) - curr_pop_size
                    for _ in range(del_count):
                        archive.pop(np.random.randint(len(archive)))

            # 2. Adaptive Parameter Generation
            r_idxs = np.random.randint(0, H, curr_pop_size)
            m_cr = mem_cr[r_idxs]
            m_f = mem_f[r_idxs]
            
            # Generate CR (Normal Distribution)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy Distribution)
            f = m_f + 0.1 * np.random.standard_cauchy(curr_pop_size)
            # Repair F
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
            f = np.clip(f, 0, 1)
            
            # 3. Mutation: current-to-pbest/1
            p = 0.11 # Top 11%
            p_num = max(2, int(p * curr_pop_size))
            
            # Choose p-best (from sorted population)
            pbest_idxs = np.random.randint(0, p_num, curr_pop_size)
            x_pbest = population[pbest_idxs]
            
            # Choose r1 (distinct from i)
            r1 = np.random.randint(0, curr_pop_size, curr_pop_size)
            for i in range(curr_pop_size):
                while r1[i] == i:
                    r1[i] = np.random.randint(0, curr_pop_size)
            x_r1 = population[r1]
            
            # Choose r2 (distinct from i and r1, from Union of Pop + Archive)
            if len(archive) > 0:
                union_pop = np.vstack((population, np.array(archive)))
            else:
                union_pop = population
            
            r2 = np.random.randint(0, len(union_pop), curr_pop_size)
            for i in range(curr_pop_size):
                while r2[i] == i or r2[i] == r1[i]:
                    r2[i] = np.random.randint(0, len(union_pop))
            x_r2 = union_pop[r2]
            
            # Compute Mutant Vector
            f_v = f[:, np.newaxis]
            mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, curr_pop_size)
            mask = np.random.rand(curr_pop_size, dim) < cr[:, np.newaxis]
            mask[np.arange(curr_pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, population)
            
            # Bound Constraint (Clipping)
            trial = np.clip(trial, lb, ub)
            
            # 5. Evaluation and Selection
            new_pop = population.copy()
            new_fit = fitness.copy()
            
            success_mask = np.zeros(curr_pop_size, dtype=bool)
            diff_fitness = np.zeros(curr_pop_size)
            
            for i in range(curr_pop_size):
                if is_time_up(): return best_val
                
                y = func(trial[i])
                
                if y <= fitness[i]:
                    new_pop[i] = trial[i]
                    new_fit[i] = y
                    
                    if y < fitness[i]:
                        success_mask[i] = True
                        diff_fitness[i] = fitness[i] - y
                        archive.append(population[i].copy())
                        
                    if y < best_val:
                        best_val = y
                        best_sol = trial[i].copy()
            
            population = new_pop
            fitness = new_fit
            
            # 6. Memory Update (Weighted Lehmer Mean)
            num_success = np.sum(success_mask)
            if num_success > 0:
                w = diff_fitness[success_mask]
                w_sum = np.sum(w)
                if w_sum > 0:
                    w /= w_sum
                    
                    sf = f[success_mask]
                    scr = cr[success_mask]
                    
                    mean_f = np.sum(w * (sf**2)) / (np.sum(w * sf) + 1e-15)
                    mean_cr = np.sum(w * scr)
                    
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                    mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # Sort population for next iteration (required for p-best)
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # 7. Convergence Check & Local Search Trigger
            # If population variance is very low OR population is minimal, trigger LS and Restart
            std_fit = np.std(fitness)
            if std_fit < 1e-9 or curr_pop_size <= min_pop_size:
                
                # --- MTS-LS1 (Multiple Trajectory Search - Local Search 1) ---
                # Apply coordinate descent on the best solution found so far
                
                # Limit LS iterations to avoid getting stuck
                ls_improved = True
                ls_iter = 0
                max_ls_iter = 30
                
                while ls_improved and ls_iter < max_ls_iter:
                    ls_iter += 1
                    ls_improved = False
                    
                    # Search in random dimension order
                    dims_perm = np.random.permutation(dim)
                    
                    for d in dims_perm:
                        if is_time_up(): return best_val
                        
                        x_curr = best_sol.copy()
                        
                        # Try moving in negative direction
                        x_curr[d] = np.clip(best_sol[d] - 0.5 * sr[d], lb[d], ub[d])
                        val = func(x_curr)
                        
                        if val < best_val:
                            best_val = val
                            best_sol = x_curr.copy()
                            ls_improved = True
                        else:
                            # Try moving in positive direction
                            x_curr[d] = np.clip(best_sol[d] + 0.5 * sr[d], lb[d], ub[d])
                            val = func(x_curr)
                            
                            if val < best_val:
                                best_val = val
                                best_sol = x_curr.copy()
                                ls_improved = True
                            else:
                                # Reduce search radius for this dimension if no improvement
                                sr[d] *= 0.5
                
                # Break the inner loop to Restart L-SHADE
                break

    return best_val
