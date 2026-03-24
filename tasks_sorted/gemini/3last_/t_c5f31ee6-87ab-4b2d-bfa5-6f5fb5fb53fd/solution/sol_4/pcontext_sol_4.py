#5. output value is: 0.000100492817281
#5. algorithm code is:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Hybrid Algorithm: SHADE (Success-History Adaptive Differential Evolution)
    integrated with MTS-LS1 (Local Search) and Restart Mechanism.
    
    Improvements:
    1. Uses SHADE for robust global search.
    2. Integrates a lightweight Coordinate Descent (Local Search) triggered by stagnation 
       to refine solutions and break local optima.
    3. Implements soft restarts with elitism to prevent premature convergence.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Adaptive based on dimension
    # Clip to ensure reasonable performance on various dims
    pop_size = int(np.clip(18 * dim, 40, 100))
    
    # SHADE Memory size
    memory_size = 5
    
    # Setup bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Best Solution Tracking
    best_val = float('inf')
    best_sol = None
    
    # Time Check Helper
    def check_limit():
        return datetime.now() - start_time >= time_limit

    # --- Main Loop (Restarts) ---
    while not check_limit():
        
        # 1. Initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Reset Search Range for Local Search on restart
        # Start large (40% of domain) to jump over hills, reduce when local minima found
        search_range = diff_b * 0.4
        
        # Inject Elite (Soft Restart)
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_val
            start_idx = 1
            
        # Evaluate Initial Pop
        for i in range(start_idx, pop_size):
            if check_limit(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
        
        # Memory Initialization
        m_cr = np.full(memory_size, 0.5)
        m_f = np.full(memory_size, 0.5)
        mem_k = 0
        
        archive = []
        stagnation_count = 0
        
        # 2. Evolutionary Cycle
        while not check_limit():
            
            # --- Parameter Adaptation ---
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # Generate CR ~ N(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            # Retry if F <= 0
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = mu_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f = np.clip(f, 0, 1)
            
            # --- Mutation: current-to-pbest/1 ---
            # Sort for p-best
            sorted_idx = np.argsort(fitness)
            p = np.random.uniform(0.05, 0.2)
            num_pbest = max(2, int(pop_size * p))
            pbest_pool = sorted_idx[:num_pbest]
            
            idx_pbest = np.random.choice(pbest_pool, pop_size)
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            
            # Archive handling for r2 (Diversity)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            idx_r2 = np.random.randint(0, len(union_pop), pop_size)
            
            x_pbest = pop[idx_pbest]
            x_r1 = pop[idx_r1]
            x_r2 = union_pop[idx_r2]
            
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            mask = rand_vals < cr[:, None]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            succ_f = []
            succ_cr = []
            diff_fit = []
            improved_any = False
            
            for i in range(pop_size):
                if check_limit(): return best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        archive.append(pop[i].copy())
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diff_fit.append(fitness[i] - f_trial)
                        improved_any = True
                    
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = pop[i].copy()
                        stagnation_count = 0
            
            # Archive resizing
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update Memory (Lehmer Mean)
            if len(succ_f) > 0:
                w = np.array(diff_fit)
                w = w / np.sum(w)
                
                mean_f = np.sum(w * (np.array(succ_f)**2)) / np.sum(w * np.array(succ_f))
                mean_cr = np.sum(w * np.array(succ_cr))
                
                m_f[mem_k] = mean_f
                m_cr[mem_k] = mean_cr
                mem_k = (mem_k + 1) % memory_size
            
            if not improved_any:
                stagnation_count += 1
            
            # --- Local Search (Coordinate Descent) ---
            # Trigger if stagnated for a short while (exploit promising basin)
            if stagnation_count > 10:
                if check_limit(): return best_val
                
                ls_improved = False
                
                # Use a permuted subset of dimensions for speed if dim is large
                check_dims = dim if dim < 20 else 20
                perm_dims = np.random.permutation(dim)[:check_dims]
                
                # Working on a temp copy of best_sol
                curr_x = best_sol.copy()
                curr_f = best_val
                
                for d in perm_dims:
                    if check_limit(): return best_val
                    
                    sr = search_range[d]
                    
                    # Negative Step
                    curr_x[d] = np.clip(curr_x[d] - sr, min_b[d], max_b[d])
                    val = func(curr_x)
                    
                    if val < curr_f:
                        curr_f = val
                        best_val = val
                        best_sol = curr_x.copy()
                        ls_improved = True
                    else:
                        # Positive Step (from original x)
                        # Current is (x - sr). We need (x + sr). So add 2*sr.
                        curr_x[d] = np.clip(curr_x[d] + 2*sr, min_b[d], max_b[d])
                        val = func(curr_x)
                        
                        if val < curr_f:
                            curr_f = val
                            best_val = val
                            best_sol = curr_x.copy()
                            ls_improved = True
                        else:
                            # Restore to original
                            curr_x[d] = np.clip(curr_x[d] - sr, min_b[d], max_b[d])
                            # Shrink search range for this dimension (fine-tuning)
                            search_range[d] *= 0.5
                
                if ls_improved:
                    stagnation_count = 0
                    # Update population: replace the current best index to keep sync
                    b_idx = np.argmin(fitness)
                    pop[b_idx] = best_sol.copy()
                    fitness[b_idx] = best_val

            # --- Restart Condition ---
            # Restart if population variance is negligible or stagnation is too long despite LS
            if np.std(fitness) < 1e-6 or stagnation_count > 25:
                break
                
    return best_val
