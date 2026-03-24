#An algorithm code that implements **L-SHADE-RSP (Success-History based Adaptive Differential Evolution with Linear Population Size Reduction)** combined with a **Restart Strategy** and an **MTS-LS1 (Multiple Trajectory Search - Local Search 1)** polish phase.
#
#### Algorithm Overview
#1.  **L-SHADE-RSP Core**: This is a state-of-the-art evolutionary algorithm. It uses:
#    *   **Linear Population Size Reduction (LPSR)**: The population size shrinks over time, shifting focus from exploration to exploitation.
#    *   **Adaptive Parameters**: Mutation ($F$) and crossover ($CR$) rates adapt based on a success history memory ($H=5$).
#    *   **Archive**: Stores recent good solutions to maintain diversity.
#2.  **Time-Based Restarts**: If the population converges early (diversity loss), the algorithm restarts with a fresh population, injecting the global best solution (Elitism) to refine it further or escape local optima.
#3.  **MTS-LS1 Polish**: In the final phase (or last ~5% of time), the algorithm switches to a **Coordinate Descent** local search (MTS-LS1). This method sequentially optimizes each dimension with adaptive step sizes, which is significantly more efficient than random search for refining continuous parameters to high precision.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-RSP with Restart Strategy and MTS-LS1 Polish.
    """
    # --- Timing Setup ---
    start_time = datetime.now()
    
    def get_remaining_seconds():
        return max_time - (datetime.now() - start_time).total_seconds()

    # --- Pre-processing Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracking ---
    best_fitness = float('inf')
    best_sol = None
    
    # --- Algorithm Configuration ---
    # Initial population size: 18 * dim is a robust standard for L-SHADE
    pop_size_init = int(np.clip(18 * dim, 30, 200))
    pop_size_min = 4
    
    # SHADE Memory parameters
    H = 5
    
    # --- Main Loop (Restart Strategy) ---
    # Continue restarts until strictly in the "Polish Phase"
    while get_remaining_seconds() > 0:
        
        # Determine if we should switch to Final Polish Phase
        # Reserve last 5% of time or last 2 seconds (whichever is larger/safer)
        remaining = get_remaining_seconds()
        polish_threshold = max(2.0, max_time * 0.05)
        
        if remaining < polish_threshold:
            break
            
        # --- Initialization for this Session ---
        pop_size = pop_size_init
        
        # Initialize Population (Uniform Random)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol
            fitness[0] = best_fitness
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if get_remaining_seconds() <= 0: return best_fitness
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_sol = pop[i].copy()
                
        # Sort Population by fitness
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Initialize SHADE Memory (M_CR, M_F) to 0.5
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # --- L-SHADE Evolution Loop ---
        while True:
            # Check Time for Polish Phase
            remaining = get_remaining_seconds()
            if remaining < polish_threshold:
                break
            
            # Check Convergence for Restart
            # If population fitness variance is extremely low, restart to explore elsewhere
            if np.max(fitness) - np.min(fitness) < 1e-8:
                break
                
            # 1. Linear Population Size Reduction (LPSR)
            # Adapts population size based on time progress
            progress = 1.0 - (remaining / max_time)
            progress = np.clip(progress, 0.0, 1.0)
            
            target_pop = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
            target_pop = max(pop_size_min, target_pop)
            
            if pop_size > target_pop:
                pop_size = target_pop
                # Truncate worst individuals (pop is sorted)
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Resize Archive to keep it proportional (size ~ 2 * pop_size)
                arc_target = int(pop_size * 2.0)
                if len(archive) > arc_target:
                    del_cnt = len(archive) - arc_target
                    idxs_del = np.random.choice(len(archive), del_cnt, replace=False)
                    archive = [archive[i] for i in range(len(archive)) if i not in idxs_del]
            
            # 2. Parameter Generation
            r_idx = np.random.randint(0, H, pop_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]
            
            # CR ~ Normal(mu, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu, 0.1)
            f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            f[f <= 0] = 0.5 # Correction for negative values
            f = np.clip(f, 0.0, 1.0)
            
            # 3. Mutation Strategy: current-to-pbest/1
            # Dynamic 'p' value reduces from exploration to exploitation
            p_val = 0.2 * (1.0 - progress) + 0.05
            p_count = max(2, int(p_val * pop_size))
            
            pbest_idxs = np.random.randint(0, p_count, pop_size)
            x_pbest = pop[pbest_idxs]
            
            # Select r1 (distinct from i)
            r1 = np.random.randint(0, pop_size, pop_size)
            coll_mask = (r1 == np.arange(pop_size))
            r1[coll_mask] = (r1[coll_mask] + 1) % pop_size
            x_r1 = pop[r1]
            
            # Select r2 (from Union of Pop and Archive)
            if len(archive) > 0:
                pop_all = np.vstack((pop, np.array(archive)))
            else:
                pop_all = pop
            r2 = np.random.randint(0, len(pop_all), pop_size)
            x_r2 = pop_all[r2]
            
            # Compute Mutant Vector
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            mask = np.random.rand(pop_size, dim) < cr[:, None]
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # 5. Selection & Evaluation
            success_f = []
            success_cr = []
            success_diff = []
            
            for i in range(pop_size):
                if get_remaining_seconds() <= 0: return best_fitness
                
                val = func(trial[i])
                
                if val <= fitness[i]:
                    if val < fitness[i]:
                        success_diff.append(fitness[i] - val)
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        archive.append(pop[i].copy())
                        
                    fitness[i] = val
                    pop[i] = trial[i]
                    
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = trial[i].copy()
                        
            # 6. Update History Memory (Weighted Lehmer Mean)
            if len(success_diff) > 0:
                s_diff = np.array(success_diff)
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                
                weights = s_diff / np.sum(s_diff)
                
                # Update M_F
                denom = np.sum(weights * s_f)
                if denom > 0:
                    mean_f = np.sum(weights * s_f**2) / denom
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                
                # Update M_CR
                mean_cr = np.sum(weights * s_cr)
                mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                
                k_mem = (k_mem + 1) % H
                
            # 7. Sort Population
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]

    # --- Final Polish: MTS-LS1 (Multiple Trajectory Search - Local Search 1) ---
    # Uses the remaining time to refine the best solution using Coordinate Descent
    if best_sol is not None:
        current_sol = best_sol.copy()
        current_fit = best_fitness
        
        # Initial search range (starts relatively large to catch nearby basins)
        search_range = (max_b - min_b) * 0.2
        
        dims = np.arange(dim)
        
        while get_remaining_seconds() > 0:
            improved = False
            np.random.shuffle(dims) # Search dimensions in random order
            
            for i in dims:
                if get_remaining_seconds() <= 0: return best_fitness
                
                # Try moving in negative direction
                x_new = current_sol.copy()
                x_new[i] -= search_range[i]
                x_new[i] = np.clip(x_new[i], min_b[i], max_b[i])
                
                val = func(x_new)
                
                if val < current_fit:
                    current_fit = val
                    current_sol = x_new
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = x_new
                    search_range[i] *= 1.5 # Expand range on success
                    improved = True
                else:
                    # Try moving in positive direction
                    x_new = current_sol.copy()
                    x_new[i] += search_range[i]
                    x_new[i] = np.clip(x_new[i], min_b[i], max_b[i])
                    
                    val = func(x_new)
                    
                    if val < current_fit:
                        current_fit = val
                        current_sol = x_new
                        if val < best_fitness:
                            best_fitness = val
                            best_sol = x_new
                        search_range[i] *= 1.5 # Expand range on success
                        improved = True
                    else:
                        # Contract range on failure
                        search_range[i] *= 0.5
            
            # Terminate polish if step sizes are negligible
            if np.max(search_range) < 1e-15:
                break
                
    return best_fitness
