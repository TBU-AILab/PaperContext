#The output value of the best generated algorithm is: 4.589970908775712
#The best generated algorithm code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the function `func` using Restart-SHADE with IPOP.
    
    SHADE (Success-History based Adaptive Differential Evolution) adapts parameters
    F and CR based on successful updates.
    
    Improvements included:
    1. IPOP (Increasing Population Size) restart strategy to escape local optima.
    2. Large external Archive (2.0 * pop_size) to maintain diversity.
    3. 'current-to-pbest/1' mutation strategy for balancing exploration/exploitation.
    4. Mid-point bound handling (instead of clipping) to better handle solutions near boundaries.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracking ---
    best_fitness = float('inf')
    best_sol = None
    
    # --- Restart Configuration ---
    # Start with a moderate population to ensure fast initial convergence.
    # Clip ensures we don't start too small or unreasonably large.
    initial_pop_size = np.clip(int(10 * dim), 20, 100)
    restart_count = 0
    
    # --- SHADE Memory Configuration ---
    memory_size = 5
    
    # --- Main Loop (Restarts) ---
    while True:
        # Check remaining time
        if datetime.now() - start_time >= time_limit:
            return best_fitness
            
        # IPOP: Increase population size exponentially
        pop_size = int(initial_pop_size * (1.5 ** restart_count))
        
        # Initialize Population
        # Shape: (pop_size, dim)
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # --- Elitism Injection ---
        # Inject the global best solution into the new population to ensure monotony
        start_eval_idx = 0
        if best_sol is not None:
            population[0] = best_sol.copy()
            fitness[0] = best_fitness
            start_eval_idx = 1
            
        # --- Evaluate Initial Population ---
        for i in range(start_eval_idx, pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_sol = population[i].copy()
                
        # --- Initialize SHADE Memory ---
        # M_CR and M_F history
        M_CR = np.full(memory_size, 0.5)
        M_F = np.full(memory_size, 0.5)
        k_mem = 0
        
        # --- Initialize Archive ---
        # Archive stores parent vectors that were successfully replaced.
        # Size factor 2.0 improves diversity maintenance.
        archive_size = int(2.0 * pop_size)
        archive = [] 
        
        # --- Generation Loop ---
        while True:
            # Strict Time Check
            if datetime.now() - start_time >= time_limit:
                return best_fitness
            
            # --- Convergence Detection ---
            # If population fitness variance/range is tiny, we are likely stuck.
            min_fit = np.min(fitness)
            max_fit = np.max(fitness)
            
            if (max_fit - min_fit) < 1e-8:
                break # Break to trigger restart
            
            # --- Parameter Generation (Vectorized) ---
            # Pick random memory index for each individual
            r_idx = np.random.randint(0, memory_size, pop_size)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]
            
            # Generate CR: Normal(M_CR, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F: Cauchy(M_F, 0.1)
            u = np.random.rand(pop_size)
            f = m_f + 0.1 * np.tan(np.pi * (u - 0.5))
            
            # Retry F if <= 0; Clip if > 1
            retry_mask = f <= 0
            while np.any(retry_mask):
                u_retry = np.random.rand(np.sum(retry_mask))
                f[retry_mask] = m_f[retry_mask] + 0.1 * np.tan(np.pi * (u_retry - 0.5))
                retry_mask = f <= 0
            f = np.clip(f, 0.0, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            
            # 1. Sort population by fitness
            sorted_idx = np.argsort(fitness)
            
            # 2. Select p-best (top 11%)
            p = 0.11
            num_pbest = max(2, int(p * pop_size))
            pbest_indices = sorted_idx[:num_pbest]
            
            # For each individual, pick a random pbest
            pbest_selection = np.random.choice(pbest_indices, pop_size)
            x_pbest = population[pbest_selection]
            
            # 3. Select r1 (random from Population)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[r1_indices]
            
            # 4. Select r2 (random from Population U Archive)
            if len(archive) > 0:
                archive_arr = np.array(archive)
                union_pop = np.vstack((population, archive_arr))
            else:
                union_pop = population
            
            r2_indices = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2_indices]
            
            # 5. Compute Mutant Vector
            # Reshape F for broadcasting (pop_size, 1)
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            mask_cross = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            # Ensure at least one dimension changes
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask_cross, mutant, population)
            
            # --- Bound Handling: Mid-point ---
            # If trial is out of bounds, set it to the average of bound and parent.
            # This is more "evolutionary" than clipping to the edge.
            trial_pop = np.where(trial_pop < min_b, (min_b + population) / 2, trial_pop)
            trial_pop = np.where(trial_pop > max_b, (max_b + population) / 2, trial_pop)
            # Clip just in case of float precision issues
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection and History Update Prep ---
            succ_f = []
            succ_cr = []
            diff_fitness = []
            
            # Evaluate Trial Vectors
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_fitness
                
                f_trial = func(trial_pop[i])
                f_old = fitness[i]
                
                if f_trial < f_old:
                    # Successful Update
                    
                    # 1. Add parent to Archive
                    if len(archive) < archive_size:
                        archive.append(population[i].copy())
                    else:
                        # Replace random member
                        rem_idx = np.random.randint(0, len(archive))
                        archive[rem_idx] = population[i].copy()
                        
                    # 2. Record success parameters
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fitness.append(f_old - f_trial)
                    
                    # 3. Update Population
                    population[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    # 4. Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_sol = trial_pop[i].copy()
                        
            # --- Update SHADE Memory ---
            if len(succ_f) > 0:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                w_diff = np.array(diff_fitness)
                
                total_imp = np.sum(w_diff)
                if total_imp > 0:
                    # Weighted Mean
                    w = w_diff / total_imp
                    
                    # Lehmer Mean for F
                    mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                    
                    # Arithmetic Mean for CR
                    mean_cr = np.sum(w * s_cr)
                    
                    # Update Memory at pointer
                    M_F[k_mem] = np.clip(mean_f, 0.0, 1.0)
                    M_CR[k_mem] = np.clip(mean_cr, 0.0, 1.0)
                    
                    # Advance pointer
                    k_mem = (k_mem + 1) % memory_size
        
        # End of Inner Loop -> Prepare Restart
        restart_count += 1
        
    return best_fitness
