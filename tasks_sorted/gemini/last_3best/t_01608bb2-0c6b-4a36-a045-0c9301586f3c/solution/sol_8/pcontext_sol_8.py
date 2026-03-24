#The following algorithm implements **L-SHADE-LPSR**, a state-of-the-art variant of Differential Evolution that won several IEEE CEC competitions.
#
#**Key Improvements over previous attempts:**
#1.  **History-Based Parameter Adaptation (SHADE)**: Instead of simple random parameter selection (jDE), this algorithm maintains a historical memory ($M_{CR}, M_F$) of successful mutation ($F$) and crossover ($CR$) rates. It learns the distribution of optimal parameters over time, allowing it to adapt to different phases of the search.
#2.  **Archive for Diversity**: An external archive stores decent solutions that were recently replaced. The `current-to-pbest` mutation strategy draws specific vectors from this archive, significantly boosting diversity and preventing premature convergence.
#3.  **Linear Population Size Reduction (LPSR)**: The population size linearly decreases from a large initial pool (exploration) to a minimal set (exploitation) as time elapses.
#4.  **Restart Mechanism**: If the population converges (low variance) before `max_time`, the algorithm restarts to find new basins of attraction, preserving the global best solution (Elitism).
#5.  **Vectorized Implementation**: All major operations (mutation, crossover, boundary handling) are vectorized using NumPy for maximum efficiency.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using L-SHADE (Success-History based Adaptive DE)
    with Linear Population Size Reduction (LPSR) and Restart Mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population Sizing: Standard L-SHADE uses 18*dim. 
    # We cap it (min 40, max 300) to ensure good performance within restricted time.
    pop_size_init = int(min(300, max(40, 18 * dim)))
    pop_size_min = 5
    
    # SHADE Memory Parameters
    H_size = 6  # Size of the historical memory
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # Helper: Check for timeout
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- Main Optimization Loop (Restarts allowed) ---
    while not check_timeout():
        
        # 1. Initialize Population for this run
        pop_size = pop_size_init
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Archive (stores strictly dominated solutions to maintain diversity)
        archive = [] 
        
        # Initialize History Memory for F and CR
        M_CR = np.full(H_size, 0.5)
        M_F = np.full(H_size, 0.5)
        k_mem = 0
        
        # Elitism: Inject global best from previous restarts
        start_eval_idx = 0
        if best_sol is not None:
            population[0] = best_sol.copy()
            fitness[0] = best_val
            start_eval_idx = 1
        
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_timeout(): return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = population[i].copy()
                
        # --- Evolutionary Cycle ---
        while True:
            # Time Check
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= max_time:
                return best_val
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate target population size based on progress
            progress = elapsed / max_time
            target_pop = int(round(pop_size_init - (pop_size_init - pop_size_min) * progress))
            target_pop = max(pop_size_min, target_pop)
            
            if pop_size > target_pop:
                # Reduce population: remove worst individuals
                sorted_idxs = np.argsort(fitness)
                keep_idxs = sorted_idxs[:target_pop]
                population = population[keep_idxs]
                fitness = fitness[keep_idxs]
                pop_size = target_pop
                
                # Resize Archive (|A| <= |P|)
                if len(archive) > pop_size:
                    random.shuffle(archive)
                    archive = archive[:pop_size]
            
            # 2. Convergence Check (Trigger Restart)
            # If population variance is too low, we are stuck.
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break
            
            # 3. Parameter Generation (SHADE Strategy)
            # Pick random index from memory
            r_idxs = np.random.randint(0, H_size, pop_size)
            m_cr = M_CR[r_idxs]
            m_f = M_F[r_idxs]
            
            # Generate CR ~ Normal(m_cr, 0.1)
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0, 1)
            
            # Generate F ~ Cauchy(m_f, 0.1)
            # Cauchy = loc + scale * tan(pi * (rand - 0.5))
            F = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Constraint Handling for F
            F = np.minimum(F, 1.0) # Clip > 1
            
            # If F <= 0, regenerate
            bad_f = F <= 0
            while np.any(bad_f):
                n_bad = np.sum(bad_f)
                m_f_bad = m_f[bad_f]
                F[bad_f] = m_f_bad + 0.1 * np.tan(np.pi * (np.random.rand(n_bad) - 0.5))
                bad_f = F <= 0
            
            # 4. Mutation: DE/current-to-pbest/1 with Archive
            # Sort population to find p-best
            sorted_args = np.argsort(fitness)
            
            # p-best selection (top 11%)
            p_rate = 0.11
            n_pbest = max(2, int(pop_size * p_rate))
            pbest_pool = sorted_args[:n_pbest]
            
            # Select pbest for each individual
            pbest_idxs = pbest_pool[np.random.randint(0, n_pbest, pop_size)]
            x_pbest = population[pbest_idxs]
            
            # r1: Distinct from i
            # Shift strategy: r1 in [0, pop-1] excluding i
            idxs_arange = np.arange(pop_size)
            r1_shift = np.random.randint(1, pop_size, pop_size)
            r1 = (idxs_arange + r1_shift) % pop_size
            x_r1 = population[r1]
            
            # r2: Distinct from i and r1, from Union(Population, Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.concatenate((population, archive_np), axis=0)
            else:
                union_pop = population
            
            n_union = len(union_pop)
            r2 = np.random.randint(0, n_union, pop_size)
            
            # Fix collisions for r2 (cannot be i or r1)
            # Note: i and r1 are indices in population.
            # If r2 < pop_size, it points to population, so check collision.
            for _ in range(5):
                is_in_pop = r2 < pop_size
                conflict = np.zeros(pop_size, dtype=bool)
                conflict[is_in_pop] = (r2[is_in_pop] == idxs_arange[is_in_pop]) | \
                                      (r2[is_in_pop] == r1[is_in_pop])
                
                if not np.any(conflict):
                    break
                r2[conflict] = np.random.randint(0, n_union, np.sum(conflict))
            
            x_r2 = union_pop[r2]
            
            # Calculate Mutant Vector
            # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            F_col = F[:, None]
            mutant = population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            cross_rand = np.random.rand(pop_size, dim)
            cross_mask = cross_rand < CR[:, None]
            # Force at least one dimension
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # 6. Bound Handling (Bounce-back / Reflection)
            # Lower bounds
            viol_l = trial_pop < min_b
            if np.any(viol_l):
                trial_pop[viol_l] = 2 * min_b[np.where(viol_l)[1]] - trial_pop[viol_l]
                # Fix double bounce
                trial_pop[trial_pop < min_b] = min_b[np.where(trial_pop < min_b)[1]]
                
            # Upper bounds
            viol_u = trial_pop > max_b
            if np.any(viol_u):
                trial_pop[viol_u] = 2 * max_b[np.where(viol_u)[1]] - trial_pop[viol_u]
                # Fix double bounce
                trial_pop[trial_pop > max_b] = max_b[np.where(trial_pop > max_b)[1]]
            
            # 7. Selection and Memory Update
            fitness_trial = np.zeros(pop_size)
            mask_better = np.zeros(pop_size, dtype=bool)   # trial < parent (Strict improvement)
            mask_replace = np.zeros(pop_size, dtype=bool)  # trial <= parent
            diff_f = np.zeros(pop_size)
            
            for i in range(pop_size):
                if check_timeout(): return best_val
                
                f_try = func(trial_pop[i])
                fitness_trial[i] = f_try
                
                if f_try <= fitness[i]:
                    mask_replace[i] = True
                    if f_try < fitness[i]:
                        mask_better[i] = True
                        diff_f[i] = fitness[i] - f_try
                    
                    if f_try < best_val:
                        best_val = f_try
                        best_sol = trial_pop[i].copy()
            
            # Update Archive: Add parents that were replaced by strictly better children
            better_idxs = np.where(mask_better)[0]
            for idx in better_idxs:
                archive.append(population[idx].copy())
            
            # Trim archive randomly if full
            while len(archive) > pop_size:
                idx_rm = random.randint(0, len(archive)-1)
                archive.pop(idx_rm)
            
            # Update Population
            population[mask_replace] = trial_pop[mask_replace]
            fitness[mask_replace] = fitness_trial[mask_replace]
            
            # Update History Memory (SHADE)
            # Uses Weighted Lehmer Mean of successful F and Weighted Mean of successful CR
            n_imp = np.sum(mask_better)
            if n_imp > 0:
                s_F = F[mask_better]
                s_CR = CR[mask_better]
                s_df = diff_f[mask_better]
                
                total_df = np.sum(s_df)
                if total_df > 0:
                    w = s_df / total_df
                    
                    # Update M_CR
                    mean_cr = np.sum(w * s_CR)
                    M_CR[k_mem] = mean_cr
                    
                    # Update M_F (Lehmer Mean)
                    num = np.sum(w * (s_F ** 2))
                    den = np.sum(w * s_F)
                    mean_f = num / den if den > 0 else 0.5
                    M_F[k_mem] = mean_f
                    
                    k_mem = (k_mem + 1) % H_size

    return best_val
