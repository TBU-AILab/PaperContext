#The following algorithm implements a **Restarted SHADE (Success-History based Adaptive Differential Evolution)** strategy.
#
#### Algorithm Description
#This approach builds upon the high-performing SHADE algorithm by adding a **restart mechanism** to handle stagnation and local optima, which is critical for minimizing functions within a fixed time limit.
#
#1.  **Core Optimizer (SHADE)**:
#    *   **Adaptive Parameters**: Uses historical memory ($M_{CR}, M_{F}$) to adapt crossover ($CR$) and mutation ($F$) rates for each individual, eliminating the need for manual tuning.
#    *   **Current-to-pbest Mutation**: Guides the search towards the top $p\%$ of best individuals ("p-best") to speed up convergence, while using an **external archive** of inferior solutions to maintain diversity.
#
#2.  **Restart Strategy**:
#    *   **Stagnation Detection**: The algorithm monitors population variance and fitness improvement. If the population converges (low variance) or fails to improve the local best for a set number of generations, it triggers a restart.
#    *   **Elitism Injection**: Upon restarting, the global best solution found so far is injected into the new random population. This ensures the search continues refining the best known basin of attraction or explores new areas without losing progress.
#
#This method combines the fast convergence of SHADE with the robustness of multiple restarts, ensuring efficient use of the available time budget.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarted SHADE (Success-History based Adaptive 
    Differential Evolution) with external archive and restarts.
    """
    start_time = time.time()
    # Safety buffer to return strictly before max_time
    time_limit = start_time + max_time - 0.1

    # --- SHADE Configuration ---
    # Population size: Standard robust sizing. 
    # Capped at 300 to ensure reasonable generations in high dimensions within limited time.
    pop_size = min(300, max(30, int(10 * dim)))
    
    # Memory size for SHADE adaptation
    H = 5
    
    # Pre-process bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Track global best across restarts
    best_global_val = float('inf')
    best_global_sol = None
    
    # --- Main Loop (Restarts) ---
    while True:
        if time.time() > time_limit:
            return best_global_val
            
        # 1. Initialize Population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        
        # Elitism: Inject global best into the new population to guide this restart
        if best_global_sol is not None:
            pop[0] = best_global_sol.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if time.time() > time_limit:
                return best_global_val
            
            # Optimization: Reuse value for injected best solution (assuming deterministic)
            if i == 0 and best_global_sol is not None:
                val = best_global_val
            else:
                val = func(pop[i])
            
            fitness[i] = val
            
            if val < best_global_val:
                best_global_val = val
                best_global_sol = pop[i].copy()
                
        # 2. Initialize SHADE Memory and Archive
        mem_M_CR = np.full(H, 0.5) # Memory for Crossover Rate
        mem_M_F = np.full(H, 0.5)  # Memory for Mutation Factor
        k_mem = 0                  # Memory index pointer
        
        archive = np.zeros((pop_size, dim))
        n_archive = 0
        
        # 3. Stagnation Tracking
        stall_count = 0
        current_run_best = np.min(fitness)
        
        # --- Evolution Loop (Generations) ---
        while True:
            if time.time() > time_limit:
                return best_global_val
            
            # A. Parameter Adaptation
            # Select random memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            mu_cr = mem_M_CR[r_idx]
            mu_f = mem_M_F[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1)
            CR = np.random.normal(mu_cr, 0.1)
            CR = np.clip(CR, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            # Numpy doesn't have loc/scale for standard_cauchy, so we scale manually
            F = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Handle F constraints (valid DE range is typically (0, 1])
            # If F <= 0, regenerate. If F > 1, clip to 1.
            bad_f_mask = F <= 0
            retry = 0
            while np.any(bad_f_mask) and retry < 10:
                n_bad = np.sum(bad_f_mask)
                F[bad_f_mask] = mu_f[bad_f_mask] + 0.1 * np.random.standard_cauchy(n_bad)
                bad_f_mask = F <= 0
                retry += 1
            F = np.clip(F, 0, 1) 
            
            # B. Mutation: DE/current-to-pbest/1 with Archive
            # Sort population by fitness
            sorted_idx = np.argsort(fitness)
            
            # p-best selection: p varies randomly in [2/N, 0.2]
            p = np.random.uniform(2/pop_size, 0.2)
            n_best = int(max(2, pop_size * p))
            pbest_pool = sorted_idx[:n_best]
            
            # Select p-best for each individual
            p_idxs = np.random.choice(pbest_pool, pop_size)
            x_pbest = pop[p_idxs]
            
            # Select r1: Random from pop, r1 != i
            idxs = np.arange(pop_size)
            r1 = np.random.randint(0, pop_size, pop_size)
            # Simple collision handling
            r1[r1 == idxs] = (r1[r1 == idxs] + 1) % pop_size
            x_r1 = pop[r1]
            
            # Select r2: Random from Union(Pop, Archive), r2 != i, r2 != r1
            pool_size = pop_size + n_archive
            r2 = np.random.randint(0, pool_size, pop_size)
            
            # Handle r2 collisions (approximate for speed)
            r2[r2 == idxs] = (r2[r2 == idxs] + 1) % pool_size # r2 != i
            # If r2 is in pop (index < pop_size), ensure r2 != r1
            mask_r2_in_pop = r2 < pop_size
            collision_r1 = mask_r2_in_pop & (r2 == r1)
            r2[collision_r1] = (r2[collision_r1] + 1) % pool_size
            
            # Construct x_r2
            x_r2 = np.zeros((pop_size, dim))
            mask_from_pop = r2 < pop_size
            mask_from_arch = ~mask_from_pop
            
            x_r2[mask_from_pop] = pop[r2[mask_from_pop]]
            if np.any(mask_from_arch):
                arch_idxs = r2[mask_from_arch] - pop_size
                x_r2[mask_from_arch] = archive[arch_idxs]
                
            # Compute Mutant Vector: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            diff_pbest = x_pbest - pop
            diff_r1_r2 = x_r1 - x_r2
            mutant = pop + F[:, None] * diff_pbest + F[:, None] * diff_r1_r2
            
            # C. Crossover (Binomial)
            mask_cr = np.random.rand(pop_size, dim) < CR[:, None]
            # Ensure at least one parameter changes
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cr[idxs, j_rand] = True
            
            trial = np.where(mask_cr, mutant, pop)
            
            # D. Bound Constraints
            trial = np.clip(trial, lb, ub)
            
            # E. Selection and Update
            fit_old = fitness.copy()
            good_cr = []
            good_f = []
            diff_fit = []
            
            for i in range(pop_size):
                if time.time() > time_limit:
                    return best_global_val
                
                f_trial = func(trial[i])
                
                # Greedy Selection
                if f_trial <= fit_old[i]:
                    # Add parent to archive before replacement
                    if n_archive < pop_size:
                        archive[n_archive] = pop[i].copy()
                        n_archive += 1
                    else:
                        # Replace random archive member
                        rand_idx = np.random.randint(0, pop_size)
                        archive[rand_idx] = pop[i].copy()
                        
                    # Update Population
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    # Store successful parameters
                    good_cr.append(CR[i])
                    good_f.append(F[i])
                    diff_fit.append(fit_old[i] - f_trial)
                    
                    # Update Global Best
                    if f_trial < best_global_val:
                        best_global_val = f_trial
                        best_global_sol = trial[i].copy()
            
            # F. Update History Memory
            if good_cr:
                w = np.array(diff_fit)
                sum_w = np.sum(w)
                if sum_w == 0:
                    w = np.ones(len(w)) / len(w)
                else:
                    w = w / sum_w
                
                # Weighted Mean for CR
                m_scr = np.sum(w * np.array(good_cr))
                mem_M_CR[k_mem] = m_scr
                
                # Weighted Lehmer Mean for F
                sf_arr = np.array(good_f)
                denom = np.sum(w * sf_arr)
                if denom == 0:
                    m_sf = 0.5
                else:
                    m_sf = np.sum(w * (sf_arr ** 2)) / denom
                mem_M_F[k_mem] = np.clip(m_sf, 0, 1)
                
                k_mem = (k_mem + 1) % H
                
            # G. Check Convergence / Restart Triggers
            new_run_best = np.min(fitness)
            if new_run_best < current_run_best:
                current_run_best = new_run_best
                stall_count = 0
            else:
                stall_count += 1
                
            # Trigger restart if stagnant or converged
            # Stagnation limit: 50 generations
            # Variance limit: 1e-9 (population collapsed)
            if stall_count > 50 or np.std(fitness) < 1e-9:
                break

    return best_global_val
