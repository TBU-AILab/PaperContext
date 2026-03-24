#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution) with Restarts**.
#
#### Key Improvements
#1.  **Adaptive Parameters (SHADE):** Unlike previous attempts with fixed or random parameters, this algorithm maintains a historical memory (`H=6`) of successful crossover rates ($CR$) and mutation factors ($F$). It automatically learns the best parameters for the specific problem landscape (e.g., separability, multimodality) during the run.
#2.  **Robust Mutation Strategy (`current-to-pbest`):** This strategy balances greediness (moving towards the top 10% best solutions) with diversity (using difference vectors from the population). This prevents the premature convergence issues seen in the `current-to-best` approach while remaining aggressive enough for the time limit.
#3.  **External Archive:** It maintains an archive of inferior solutions recently replaced by better ones. This preserves diversity in the mutation difference vectors ($x_{r1} - x_{r2}$), preventing the search step size from collapsing to zero too quickly.
#4.  **Restart Mechanism:** Given the "limited time" constraint, the algorithm detects stagnation (no improvement for 25 generations) or convergence (low variance) and triggers a hard restart. This maximizes the exploration of different basins of attraction.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive DE) with Restarts.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Balance between speed and diversity.
    # SHADE benefits from slightly larger populations than standard DE, 
    # but we clamp it to ensure generations are fast.
    pop_size = int(np.clip(15 * dim, 30, 100))
    
    # SHADE Parameters
    H = 6  # Memory size for historical parameter adaptation
    mem_M_F = np.full(H, 0.5)  # Memory for Mutation Factor F
    mem_M_CR = np.full(H, 0.5) # Memory for Crossover Rate CR
    k_mem = 0  # Memory index pointer
    p_best_rate = 0.11 # Top 11% used for current-to-pbest (ensures >=2 individuals)
    
    # Pre-process bounds for efficient broadcasting
    bounds_arr = np.array(bounds)
    lb = bounds_arr[:, 0]
    ub = bounds_arr[:, 1]
    diff = ub - lb
    
    global_best_fitness = float('inf')
    
    # --- Main Loop (Restarts) ---
    while True:
        # Check overall time budget
        if time.time() - start_time >= max_time:
            return global_best_fitness
        
        # --- Initialization ---
        pop = lb + np.random.rand(pop_size, dim) * diff
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return global_best_fitness
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_fitness:
                global_best_fitness = val
                
        # External Archive (stores successful parents to maintain diversity)
        # Fixed size equal to population size
        archive = np.zeros((pop_size, dim))
        arc_count = 0
        
        # Convergence trackers
        stagnation_counter = 0
        last_gen_best = np.min(fitness)
        
        # --- Evolution Loop ---
        while True:
            # Time check per generation
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            # 1. Parameter Generation (Adaptive)
            # Pick random memory slot for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_M_F[r_idx]
            m_cr = mem_M_CR[r_idx]
            
            # Generate F using Cauchy distribution (allows occasional large steps)
            F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            # Repair invalid F values (must be > 0)
            retry_mask = F <= 0
            while np.any(retry_mask):
                F[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
                retry_mask = F <= 0
            F = np.minimum(F, 1.0) # Clip max at 1.0
            
            # Generate CR using Normal distribution
            CR = np.random.normal(m_cr, 0.1, pop_size)
            CR = np.clip(CR, 0.0, 1.0)
            
            # 2. Mutation: current-to-pbest/1
            # Sort population to find p-best
            sorted_idx = np.argsort(fitness)
            sorted_pop = pop[sorted_idx]
            
            # Select p-best individuals (randomly from top p%)
            num_pbest = max(2, int(pop_size * p_best_rate))
            pbest_indices = np.random.randint(0, num_pbest, pop_size)
            x_pbest = sorted_pop[pbest_indices]
            
            # Select r1 (random from population)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_indices]
            
            # Select r2 (random from Union of Population and Archive)
            # This is key for SHADE: using archive increases diversity of difference vectors
            if arc_count > 0:
                # Efficiently select from union without full stacking
                union_size = pop_size + arc_count
                r2_indices = np.random.randint(0, union_size, pop_size)
                
                # We need to build the x_r2 array based on indices
                x_r2 = np.zeros((pop_size, dim))
                
                # Indices < pop_size refer to current population
                pop_mask = r2_indices < pop_size
                x_r2[pop_mask] = pop[r2_indices[pop_mask]]
                
                # Indices >= pop_size refer to archive
                arc_mask = ~pop_mask
                arc_lookup = r2_indices[arc_mask] - pop_size
                x_r2[arc_mask] = archive[arc_lookup]
            else:
                r2_indices = np.random.randint(0, pop_size, pop_size)
                x_r2 = pop[r2_indices]
                
            # Compute Mutant Vector: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            F_matrix = F[:, np.newaxis]
            mutant = pop + F_matrix * (x_pbest - pop) + F_matrix * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < CR[:, np.newaxis]
            
            # Ensure at least one parameter comes from mutant (DE standard)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, pop)
            
            # Boundary Constraint (Clip)
            trial_pop = np.clip(trial_pop, lb, ub)
            
            # 4. Selection & Memory Update Preparation
            diff_fitness = [] # Track fitness improvements
            scr_f = []        # Successful F values
            scr_cr = []       # Successful CR values
            
            # Evaluate trials
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return global_best_fitness
                
                f_trial = func(trial_pop[i])
                
                # Greedy selection
                if f_trial <= fitness[i]:
                    # If strictly better, store data for adaptation
                    if f_trial < fitness[i]:
                        diff_fitness.append(fitness[i] - f_trial)
                        scr_f.append(F[i])
                        scr_cr.append(CR[i])
                        
                        # Add parent to archive before replacing it
                        if arc_count < pop_size:
                            archive[arc_count] = pop[i].copy()
                            arc_count += 1
                        else:
                            # If archive full, replace random member
                            rand_idx = np.random.randint(0, pop_size)
                            archive[rand_idx] = pop[i].copy()
                            
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial
            
            # 5. Update Historical Memory (SHADE logic)
            # Uses weighted averages based on fitness improvement magnitude
            if len(diff_fitness) > 0:
                w = np.array(diff_fitness)
                w = w / np.sum(w) # Normalize weights
                
                s_f = np.array(scr_f)
                s_cr = np.array(scr_cr)
                
                # Weighted Lehmer Mean for F
                sum_w_f = np.sum(w * s_f)
                if sum_w_f > 0:
                    mean_f = np.sum(w * (s_f ** 2)) / sum_w_f
                    mem_M_F[k_mem] = np.clip(mean_f, 0, 1)
                
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(w * s_cr)
                mem_M_CR[k_mem] = np.clip(mean_cr, 0, 1)
                
                # Increment memory pointer
                k_mem = (k_mem + 1) % H
                
            # 6. Convergence & Stagnation Check
            current_best = np.min(fitness)
            if abs(current_best - last_gen_best) < 1e-8:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_gen_best = current_best
                
            # Restart if:
            # 1. Stagnation: Best fitness hasn't improved for 25 generations.
            # 2. Convergence: Population variance is extremely low.
            if stagnation_counter > 25 or np.std(fitness) < 1e-8:
                break

    return global_best_fitness
