#To improve upon the best previous result (21.02), I propose an algorithm based on **L-SHADE-OBL (Linear Success-History Adaptive Differential Evolution with Opposition-Based Learning)**.
#
#### Algorithm Explanation
#
#1.  **L-SHADE Structure**: This is an extension of the SHADE algorithm (which performed best previously). The key addition is **Linear Population Size Reduction (LPSR)**. As the search progresses and moves from exploration to exploitation, the algorithm linearly reduces the population size. This forces the computational budget to focus on the best individuals in the later stages, significantly refining the solution precision.
#2.  **Opposition-Based Learning (OBL) Initialization**: Instead of purely random initialization, we generate a random population and its mathematical opposite ($min + max - x$). We evaluate both and keep the fittest half. This provides a vastly superior starting point.
#3.  **External Archive**: Unlike standard DE, this implementation utilizes an external "Archive" of inferior solutions that were recently replaced. The mutation operator pulls from this archive to maintain diversity, preventing premature convergence.
#4.  **Time-Based Budgeting**: Since the constraint is `max_time` (seconds) rather than generations, the linear population reduction is mapped to the elapsed time. The algorithm manages "Episodes" of restarts. If an episode converges or runs out of its allocated time slice, it restarts (injecting the global best) to search new basins of attraction.
#
#### Python Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Opposition-Based Learning and 
    Linear Population Size Reduction (LPSR) based on time budget.
    """
    start_time = datetime.now()
    # Safe buffer to return before timeout
    time_limit = timedelta(seconds=max_time * 0.98)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Best found global solution
    global_best_fitness = float('inf')
    global_best_solution = None

    # Wrapper to handle function errors safely
    def safe_func(x):
        try:
            return func(x)
        except:
            return float('inf')

    # --- Restart Loop ---
    # We treat the optimization as a series of "Episodes" or Restarts.
    # Each episode is allocated a portion of the remaining time.
    
    episode_count = 0
    
    while True:
        # Check global time
        now = datetime.now()
        if now - start_time >= time_limit:
            return global_best_fitness
        
        # Determine budget for this episode
        # Strategy: 1st run gets chunk of time, subsequent runs get smaller chunks 
        # to encourage fast exploration of new areas.
        remaining_seconds = (start_time + time_limit - now).total_seconds()
        
        if remaining_seconds < 0.1:
            return global_best_fitness

        # Budget allocation logic
        if episode_count == 0:
            # First run: give it 40% of time or at least 2 seconds if possible
            episode_budget = max(2.0, remaining_seconds * 0.4)
        else:
            # Subsequent runs: split remaining time
            episode_budget = max(1.0, remaining_seconds * 0.5)
            
        episode_end = now + timedelta(seconds=episode_budget)
        episode_count += 1
        
        # --- L-SHADE Initialization ---
        
        # Initial Population Size (N_init)
        # Scaled by dim, but clamped for performance
        N_init = int(round(18 * dim)) 
        N_init = int(np.clip(N_init, 30, 150))
        
        # Final Population Size (N_min)
        N_min = 4
        
        current_pop_size = N_init
        
        # 1. OBL Initialization
        # Generate random
        pop_rand = min_b + np.random.rand(current_pop_size, dim) * diff_b
        # Generate opposite
        pop_opp = min_b + max_b - pop_rand
        
        # Check bounds for opposite
        mask_oob = (pop_opp < min_b) | (pop_opp > max_b)
        if np.any(mask_oob):
            random_fallback = min_b + np.random.rand(current_pop_size, dim) * diff_b
            pop_opp = np.where(mask_oob, random_fallback, pop_opp)
            
        # Evaluate both sets
        pop_combined = np.vstack((pop_rand, pop_opp))
        fit_combined = np.full(2 * current_pop_size, float('inf'))
        
        for i in range(2 * current_pop_size):
            if datetime.now() >= episode_end: break
            fit_combined[i] = safe_func(pop_combined[i])
            
        # Select best N_init individuals
        sorted_indices = np.argsort(fit_combined)
        pop = pop_combined[sorted_indices[:current_pop_size]].copy()
        fitness = fit_combined[sorted_indices[:current_pop_size]].copy()
        
        # Update Global Best
        if fitness[0] < global_best_fitness:
            global_best_fitness = fitness[0]
            global_best_solution = pop[0].copy()
            
        # Elitism Injection: 
        # If we have a global best from previous runs, insert it to prevent regression
        if global_best_solution is not None:
            # Replace worst
            pop[-1] = global_best_solution.copy()
            fitness[-1] = global_best_fitness
            # Resort
            resort_idx = np.argsort(fitness)
            pop = pop[resort_idx]
            fitness = fitness[resort_idx]

        # --- SHADE Memory & Archive ---
        memory_size = 5
        M_F = np.full(memory_size, 0.5)
        M_CR = np.full(memory_size, 0.5)
        k_mem = 0
        archive = []
        
        # Archive size relative to population
        arc_rate = 2.0 
        
        # --- Evolutionary Loop ---
        while True:
            # Time check
            now_step = datetime.now()
            if now_step >= episode_end or now_step - start_time >= time_limit:
                break
                
            # 2. Linear Population Size Reduction (LPSR)
            # Calculate progress ratio based on time consumed in this episode
            time_consumed = (now_step - now).total_seconds()
            progress = min(1.0, time_consumed / episode_budget)
            
            # Calculate target population size
            N_target = int(round(N_init + (N_min - N_init) * progress))
            N_target = max(N_min, N_target)
            
            # Reduce Population if needed
            if current_pop_size > N_target:
                reduction_amt = current_pop_size - N_target
                # The population is always sorted by fitness at the end of loop? 
                # No, we must find worst to remove.
                sorted_idx = np.argsort(fitness)
                # Keep best N_target
                keep_idx = sorted_idx[:N_target]
                pop = pop[keep_idx]
                fitness = fitness[keep_idx]
                
                current_pop_size = N_target
                
                # Resize Archive
                archive_target_size = int(current_pop_size * arc_rate)
                if len(archive) > archive_target_size:
                    # Randomly drop elements to fit
                    del_count = len(archive) - archive_target_size
                    # It's a list, pop random indices
                    for _ in range(del_count):
                        archive.pop(np.random.randint(0, len(archive)))
            
            # 3. Parameter Generation
            # Randomly select memory slot
            r_idx = np.random.randint(0, memory_size, current_pop_size)
            m_f = M_F[r_idx]
            m_cr = M_CR[r_idx]
            
            # Generate CR: Normal distribution, clipped [0, 1]
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            # Ensure CR is not always 0 (optional usually, but good for diversity)
            CR = np.maximum(CR, 0.05) 
            
            # Generate F: Cauchy distribution
            F = m_f + 0.1 * np.random.standard_cauchy(current_pop_size)
            
            # Check F validity
            # If F > 1, clip to 1. If F <= 0, retry.
            retry_mask = F <= 0
            retries = 0
            while np.any(retry_mask) and retries < 5:
                F[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
                retry_mask = F <= 0
                retries += 1
            F[F <= 0] = 0.5 # Fallback
            F = np.minimum(F, 1.0)
            
            # 4. Mutation Strategy: current-to-pbest/1
            # Sort population for pbest selection
            sorted_indices = np.argsort(fitness)
            
            # p varies randomly in [2/N, 0.2]
            p_min = 2.0 / current_pop_size
            p_val = np.random.uniform(p_min, 0.2)
            top_p_count = int(max(2, p_val * current_pop_size))
            
            top_p_indices = sorted_indices[:top_p_count]
            
            pbest_indices = np.random.choice(top_p_indices, current_pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (distinct from i)
            idxs = np.arange(current_pop_size)
            r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
            # Fix collisions r1 == i
            collision = r1_indices == idxs
            r1_indices[collision] = (r1_indices[collision] + 1) % current_pop_size
            x_r1 = pop[r1_indices]
            
            # Select r2 (distinct from i, r1) from Union(Population, Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((pop, archive_np))
            else:
                union_pop = pop
            
            union_size = len(union_pop)
            r2_indices = np.random.randint(0, union_size, current_pop_size)
            
            # Simple collision handling logic for vectorization
            # Ideally r2 != r1 and r2 != i.
            # We just offset if there is a match to keep it fast.
            r2_indices = np.where(r2_indices == idxs, (r2_indices + 1) % union_size, r2_indices)
            r2_indices = np.where(r2_indices == r1_indices, (r2_indices + 1) % union_size, r2_indices)
            
            x_r2 = union_pop[r2_indices]
            
            # Compute Mutant Vector
            F_col = F[:, np.newaxis]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # Boundary constraint (Bounce back / Random / Clip)
            # Clipping is robust
            mutant = np.clip(mutant, min_b, max_b)
            
            # 5. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, current_pop_size)
            j_rand_mask = np.zeros((current_pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(current_pop_size), j_rand] = True
            
            cross_mask = np.random.rand(current_pop_size, dim) < CR[:, np.newaxis]
            trial_mask = cross_mask | j_rand_mask
            
            trial_pop = np.where(trial_mask, mutant, pop)
            
            # 6. Evaluation and Selection
            success_F = []
            success_CR = []
            diff_fitness = []
            
            # Loop for evaluation (allows strict time checking)
            for i in range(current_pop_size):
                if datetime.now() - start_time >= time_limit:
                    return global_best_fitness
                
                trial_val = safe_func(trial_pop[i])
                
                if trial_val <= fitness[i]:
                    # Successful update
                    if trial_val < fitness[i]:
                        # Store params for memory update
                        success_F.append(F[i])
                        success_CR.append(CR[i])
                        diff_fitness.append(fitness[i] - trial_val)
                        
                        # Add parent to archive
                        archive.append(pop[i].copy())
                        
                    # Update population
                    pop[i] = trial_pop[i]
                    fitness[i] = trial_val
                    
                    if trial_val < global_best_fitness:
                        global_best_fitness = trial_val
                        global_best_solution = trial_pop[i].copy()
            
            # Manage Archive Size post-insertion
            max_archive_size = int(current_pop_size * arc_rate)
            while len(archive) > max_archive_size:
                archive.pop(np.random.randint(0, len(archive)))
                
            # 7. Update Memory (Weighted Lehmer Mean)
            if len(diff_fitness) > 0:
                diff_np = np.array(diff_fitness)
                sf_np = np.array(success_F)
                scr_np = np.array(success_CR)
                
                # Weights based on fitness improvement
                weights = diff_np / np.sum(diff_np)
                
                # Update M_F (Lehmer Mean)
                if np.sum(weights * sf_np) > 0:
                    mean_f = np.sum(weights * (sf_np ** 2)) / np.sum(weights * sf_np)
                    M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                
                # Update M_CR (Weighted Mean)
                mean_cr = np.sum(weights * scr_np)
                M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
                
                k_mem = (k_mem + 1) % memory_size
            
            # 8. Convergence Check
            # If population fitness variance is extremely low, restart to save time
            fit_spread = np.max(fitness) - np.min(fitness)
            if fit_spread < 1e-9:
                break
                
    return global_best_fitness
