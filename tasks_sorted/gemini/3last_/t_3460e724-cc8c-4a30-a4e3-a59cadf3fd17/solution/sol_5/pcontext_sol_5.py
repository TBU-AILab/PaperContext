#The following algorithm implements **IPO-jDE (Increasing Population Optimization with Self-Adaptive Differential Evolution)**.
#
#**Rationale for Improvement:**
#1.  **Exploitation-Exploration Balance via IPO**: The previous best result (15.77) used a restart mechanism with a fixed population range. This algorithm starts with a small population to rapidly identify easy local optima. Upon convergence, it **restarts with a doubled population size**. This increasing capacity allows the algorithm to solve simple problems quickly while automatically scaling up computational effort for complex, multimodal landscapes where the previous jDE might have stagnated.
#2.  **Hybrid Mutation Strategy (`current-to-pbest`)**: While the previous jDE used `rand/1`, this solution incorporates the superior `current-to-pbest/1` mutation strategy from SHADE. This directs the search using information from the top $p\%$ of individuals ("p-best") and an external **Archive** of inferior solutions. This combination (Gradient-like direction + Diversity preservation) significantly enhances convergence speed and accuracy.
#3.  **Self-Adaptation (jDE)**: It retains the robust parameter adaptation logic of jDE (random resets of $F$ and $CR$), which maintains higher variance in control parameters than SHADE's history mechanism, preventing premature fixation on specific search behaviors.
#4.  **Optimized Time Management**: The code is fully vectorized (except for the unavoidable function evaluation loop) and checks time constraints strictly at the granular individual level, ensuring the best possible result is returned without timeout violation.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using IPO-jDE (Increasing Population Optimization with 
    Self-Adaptive Differential Evolution).
    """
    start_time = time.time()
    
    # --- Helper: Check remaining time ---
    def check_time():
        return (time.time() - start_time) >= max_time

    # --- Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- IPO Configuration ---
    # Start with a small population for fast convergence on simple basins
    # N_init = max(20, 5 * dim)
    pop_size = max(20, int(5 * dim))
    
    # Cap population to avoid excessive computational cost per generation
    MAX_POP_SIZE = 500
    
    # --- Main Restart Loop ---
    while True:
        # If time is nearly up, don't start a new run
        if (time.time() - start_time) > max_time - 0.05:
            return global_best_val
            
        # --- Initialization ---
        N = pop_size
        
        # Initialize Population
        pop = min_b + np.random.rand(N, dim) * diff_b
        fitness = np.full(N, float('inf'))
        
        # jDE Parameter Initialization
        # F ~ U(0.1, 1.0), CR ~ U(0.0, 1.0)
        F = 0.1 + 0.9 * np.random.rand(N)
        CR = np.random.rand(N)
        
        # Archive (stores replaced individuals to maintain diversity)
        # Capacity is set to 2.0 * pop_size
        arc_capacity = int(N * 2)
        archive = np.zeros((arc_capacity, dim))
        arc_count = 0
        
        # Elitism: Inject the global best solution from previous runs
        # We place it at index 0 and skip its evaluation
        start_eval_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_eval_idx = 1
        
        # Evaluate Initial Population
        for i in range(start_eval_idx, N):
            if check_time(): return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
        
        # --- Evolution Loop ---
        while True:
            if check_time(): return global_best_val
            
            # 1. Sort population (required for p-best selection)
            # We sort pop, fitness, F, and CR in unison based on fitness
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            F = F[sorted_indices]
            CR = CR[sorted_indices]
            
            # 2. Convergence Check (Restart Trigger)
            # If fitness variance is extremely low, we are stuck.
            fit_range = fitness[-1] - fitness[0]
            if fit_range < 1e-8:
                break
                
            # 3. Parameter Adaptation (jDE logic)
            # F_new = 0.1 + 0.9*rand if rand < tau1, else F_old
            # CR_new = rand if rand < tau2, else CR_old
            tau1, tau2 = 0.1, 0.1
            
            rand_f = np.random.rand(N)
            mask_f = rand_f < tau1
            F_trial = np.where(mask_f, 0.1 + 0.9 * np.random.rand(N), F)
            
            rand_cr = np.random.rand(N)
            mask_cr = rand_cr < tau2
            CR_trial = np.where(mask_cr, np.random.rand(N), CR)
            
            # 4. Mutation: current-to-pbest/1
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            
            # Select p-best (top 10%)
            p_top = max(2, int(N * 0.1))
            pbest_indices = np.random.randint(0, p_top, N)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (random from pop, != i)
            r1_indices = np.random.randint(0, N, N)
            # Simple shift to avoid self-selection
            mask_self = (r1_indices == np.arange(N))
            r1_indices[mask_self] = (r1_indices[mask_self] + 1) % N
            x_r1 = pop[r1_indices]
            
            # Select r2 (random from Pop U Archive, != r1, != i)
            # Create a virtual pool for r2
            if arc_count > 0:
                pool = np.vstack((pop, archive[:arc_count]))
            else:
                pool = pop
                
            pool_size = len(pool)
            r2_indices = np.random.randint(0, pool_size, N)
            # We skip strict inequality checks for r2 for speed;
            # DE is robust to occasional degenerate vectors.
            x_r2 = pool[r2_indices]
            
            # Compute Mutant
            F_col = F_trial[:, None]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            rand_mat = np.random.rand(N, dim)
            j_rand = np.random.randint(0, dim, N)
            j_mask = np.zeros((N, dim), dtype=bool)
            j_mask[np.arange(N), j_rand] = True
            
            cross_mask = (rand_mat < CR_trial[:, None]) | j_mask
            trial = np.where(cross_mask, mutant, pop)
            
            # 6. Bound Handling (Midpoint Correction)
            # (parent + bound) / 2 preserves direction better than clipping
            lower_viol = trial < min_b
            upper_viol = trial > max_b
            
            trial = np.where(lower_viol, (pop + min_b) * 0.5, trial)
            trial = np.where(upper_viol, (pop + max_b) * 0.5, trial)
            
            # 7. Evaluation and Selection
            parents_to_archive = []
            
            # Next gen containers
            next_pop = pop.copy()
            next_fitness = fitness.copy()
            next_F = F.copy()
            next_CR = CR.copy()
            
            for i in range(N):
                if check_time(): return global_best_val
                
                f_trial = func(trial[i])
                
                # Global Best Update
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_sol = trial[i].copy()
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    # Trial is better or equal
                    # Add replaced parent to archive list
                    parents_to_archive.append(pop[i].copy())
                    
                    next_pop[i] = trial[i]
                    next_fitness[i] = f_trial
                    # Update parameters to successful trial values
                    next_F[i] = F_trial[i]
                    next_CR[i] = CR_trial[i]
                else:
                    # Parent survives, retain old parameters
                    pass
            
            pop = next_pop
            fitness = next_fitness
            F = next_F
            CR = next_CR
            
            # 8. Archive Update
            if len(parents_to_archive) > 0:
                cands = np.array(parents_to_archive)
                num_cands = len(cands)
                
                space = arc_capacity - arc_count
                if num_cands <= space:
                    archive[arc_count:arc_count+num_cands] = cands
                    arc_count += num_cands
                else:
                    # Fill remaining space
                    if space > 0:
                        archive[arc_count:] = cands[:space]
                        cands = cands[space:]
                        arc_count = arc_capacity
                    
                    # Randomly replace existing members
                    if len(cands) > 0:
                        replace_idx = np.random.randint(0, arc_capacity, len(cands))
                        archive[replace_idx] = cands

        # --- End of Run (Converged) ---
        # Double the population size for the next restart (IPO strategy)
        # This helps in escaping local optima that trapped the smaller population
        pop_size *= 2
        if pop_size > MAX_POP_SIZE:
            pop_size = MAX_POP_SIZE

    return global_best_val
