#The following algorithm implements **Restarting SHADE (Success-History based Adaptive Differential Evolution)**.
#
#**Rationale for Improvement:**
#1.  **State-of-the-Art DE**: SHADE is widely recognized as a superior variant of Differential Evolution compared to jDE or standard DE, using historical memory to adapt parameters ($F$ and $CR$) to the specific problem landscape.
#2.  **Restart Mechanism with Elitism**: Previous results showed that jDE (with restarts) performed better than a single run of SHADE or CMA-ES, likely due to escaping local optima. This algorithm integrates a restart strategy into SHADE. Crucially, it preserves the **global best solution** found so far and injects it into the new population (Elitism) to ensure monotonic improvement.
#3.  **Time Management**: The algorithm strictly monitors the execution time before every batch of expensive function evaluations to ensure the result is returned exactly within limits.
#4.  **Robust Population Sizing**: The population size is dynamically scaled based on dimension but strictly capped (max 80) to ensure the algorithm can perform enough generations to converge even when function evaluations are slow or time is short.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Restarting SHADE (Success-History based Adaptive Differential Evolution).
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Adaptive to dimension but constrained for speed.
    # Small enough to allow many generations, large enough for diversity.
    pop_size = int(10 * dim)
    if pop_size < 30: pop_size = 30
    if pop_size > 80: pop_size = 80
    
    # SHADE Specific Parameters
    H = 5                   # Memory size for historical parameters
    p_best_rate = 0.11      # Greedy factor for current-to-pbest mutation
    arc_rate = 1.4          # Archive size relative to population
    archive_size = int(pop_size * arc_rate)
    
    # Pre-process bounds
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Main Restart Loop ---
    while True:
        # Check if we have enough time to meaningfully start a new run
        if (time.time() - start_time) > max_time - 0.05:
            return global_best_val

        # --- Initialization ---
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best solution from previous restarts
        start_eval_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_eval_idx = 1
            
        # Initialize SHADE Memory (F=0.5, CR=0.5 initially)
        mem_F = np.full(H, 0.5)
        mem_CR = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive = np.zeros((archive_size, dim))
        arc_cnt = 0
        
        # Initial Evaluation
        for i in range(start_eval_idx, pop_size):
            if (time.time() - start_time) > max_time:
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
        
        # --- Evolution Loop ---
        while True:
            # Time check at start of generation
            if (time.time() - start_time) > max_time:
                return global_best_val
            
            # 1. Parameter Generation
            # Randomly select memory index for each individual
            r_idx = np.random.randint(0, H, pop_size)
            mF = mem_F[r_idx]
            mCR = mem_CR[r_idx]
            
            # Generate F using Cauchy distribution
            F = mF + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            # Handle F constraints (if <= 0 fallback to 0.5, if > 1 clip)
            F = np.where(F <= 0, 0.5, F)
            F = np.clip(F, 0, 1)
            
            # Generate CR using Normal distribution
            CR = mCR + 0.1 * np.random.randn(pop_size)
            CR = np.clip(CR, 0, 1)
            
            # 2. Mutation: current-to-pbest/1
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            
            # Identify p-best individuals
            sorted_indices = np.argsort(fitness)
            num_pbest = max(2, int(pop_size * p_best_rate))
            pbest_pool = sorted_indices[:num_pbest]
            
            # Select pbest, r1, r2
            idx_pbest = np.random.choice(pbest_pool, pop_size)
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            
            x_pbest = pop[idx_pbest]
            x_r1 = pop[idx_r1]
            
            # Select r2 from Union(Population, Archive)
            if arc_cnt > 0:
                union_pop = np.vstack((pop, archive[:arc_cnt]))
                idx_r2 = np.random.randint(0, pop_size + arc_cnt, pop_size)
                x_r2 = union_pop[idx_r2]
            else:
                idx_r2 = np.random.randint(0, pop_size, pop_size)
                x_r2 = pop[idx_r2]
                
            # Compute Mutant Vector
            F_col = F[:, None]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            rand_matrix = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            j_mask = np.zeros((pop_size, dim), dtype=bool)
            j_mask[np.arange(pop_size), j_rand] = True
            
            cross_mask = (rand_matrix < CR[:, None]) | j_mask
            trial = np.where(cross_mask, mutant, pop)
            
            # 4. Bound Constraint Handling (Midpoint Correction)
            bad_lower = trial < min_b
            bad_upper = trial > max_b
            trial = np.where(bad_lower, (pop + min_b) * 0.5, trial)
            trial = np.where(bad_upper, (pop + max_b) * 0.5, trial)
            
            # 5. Evaluation and Selection
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            success_F = []
            success_CR = []
            diff_f = []
            parents_to_archive = []
            
            for i in range(pop_size):
                if (time.time() - start_time) > max_time:
                    return global_best_val
                
                f_trial = func(trial[i])
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_sol = trial[i].copy()
                
                # Selection
                if f_trial <= fitness[i]:
                    # Improvement or equal
                    df = fitness[i] - f_trial
                    
                    success_F.append(F[i])
                    success_CR.append(CR[i])
                    diff_f.append(df)
                    
                    parents_to_archive.append(pop[i].copy())
                    
                    new_pop[i] = trial[i]
                    new_fitness[i] = f_trial
            
            pop = new_pop
            fitness = new_fitness
            
            # 6. Archive Update
            if len(parents_to_archive) > 0:
                cands = np.array(parents_to_archive)
                num_cands = len(cands)
                
                if arc_cnt < archive_size:
                    spaces = archive_size - arc_cnt
                    if num_cands <= spaces:
                        archive[arc_cnt : arc_cnt+num_cands] = cands
                        arc_cnt += num_cands
                    else:
                        archive[arc_cnt:] = cands[:spaces]
                        arc_cnt = archive_size
                        # Randomly replace remainder
                        idx_repl = np.random.randint(0, archive_size, num_cands - spaces)
                        archive[idx_repl] = cands[spaces:]
                else:
                    idx_repl = np.random.randint(0, archive_size, num_cands)
                    archive[idx_repl] = cands
            
            # 7. Memory Update (Weighted Lehmer Mean)
            if len(diff_f) > 0:
                S_F = np.array(success_F)
                S_CR = np.array(success_CR)
                diffs = np.array(diff_f)
                
                sum_diff = np.sum(diffs)
                if sum_diff > 0:
                    weights = diffs / sum_diff
                    
                    denom = np.sum(weights * S_F)
                    if denom > 1e-9:
                        mean_F = np.sum(weights * (S_F ** 2)) / denom
                        mem_F[k_mem] = np.clip(mean_F, 0, 1)
                    
                    mean_CR = np.sum(weights * S_CR)
                    mem_CR[k_mem] = np.clip(mean_CR, 0, 1)
                    
                    k_mem = (k_mem + 1) % H
            
            # 8. Convergence Check for Restart
            # If the population has converged (low variance), restart to explore new areas
            if np.max(fitness) - np.min(fitness) < 1e-8:
                break

    return global_best_val
