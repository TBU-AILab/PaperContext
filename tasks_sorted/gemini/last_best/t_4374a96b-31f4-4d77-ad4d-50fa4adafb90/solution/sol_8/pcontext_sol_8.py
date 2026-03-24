#An improved, fully operational, and self-contained algorithm based on **SHADE (Success-History based Adaptive Differential Evolution) with Restarts**. 
#
##### Logic & Improvements
#1.  **SHADE Algorithm**: This is a significant upgrade over jDE. It utilizes a **Historical Memory** ($M_{CR}, M_F$) to adapt control parameters ($F$ and $CR$) based on the success of previous generations. This allows the algorithm to learn the landscape features (e.g., separability, multimodality) more effectively than simple self-adaptation.
#2.  **External Archive**: An archive is implemented to maintain population diversity. This is crucial for the **current-to-pbest/1** mutation strategy (used here), preventing premature convergence—a likely reason for the failure of the previous "current-to-pbest" attempt.
#3.  **Restarts**: To ensure the algorithm keeps searching for the global minimum within the `max_time`, a restart mechanism triggers when the population fitness variance becomes negligible or stagnation is detected.
#4.  **Vectorization**: The implementation is heavily vectorized using NumPy for efficiency, minimizing Python loop overhead.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using SHADE (Success-History based Adaptive DE) with Restarts.
    """
    start_time = time.time()
    time_limit = max_time - 0.05  # Buffer
    
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    diff = ub - lb
    
    # SHADE Parameters
    # Dynamic population size: balances exploration and speed
    pop_size = int(np.clip(10 * dim, 30, 80))
    H = 5  # History memory size
    
    # Global best tracker
    best_fitness = float('inf')
    
    # Main Restart Loop
    while True:
        if time.time() - start_time > time_limit:
            return best_fitness
            
        # --- Initialization ---
        # Initialize History Memory
        mem_F = np.full(H, 0.5)
        mem_CR = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Population
        pop = lb + np.random.rand(pop_size, dim) * diff
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if time.time() - start_time > time_limit:
                return best_fitness
            val = func(pop[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
        
        # Initialize Archive
        # We pre-allocate for speed but manage size with n_arch
        archive = np.zeros((pop_size, dim))
        n_arch = 0
        
        # Stagnation counter for restart
        stagnation_counter = 0
        last_gen_best = np.min(fitness)
        
        # --- Evolutionary Cycle ---
        while True:
            # Time Check
            if time.time() - start_time > time_limit:
                return best_fitness
            
            # 1. Parameter Generation (SHADE)
            # Pick random index from memory
            r_idx = np.random.randint(0, H, pop_size)
            mF = mem_F[r_idx]
            mCR = mem_CR[r_idx]
            
            # Generate F using Cauchy distribution: C(mF, 0.1)
            # F = mF + 0.1 * tan(pi * (rand - 0.5))
            F = mF + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Fix F <= 0 (regenerate) and F > 1 (clip)
            bad_F = F <= 0
            while np.any(bad_F):
                F[bad_F] = mF[bad_F] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_F)) - 0.5))
                bad_F = F <= 0
            F = np.minimum(F, 1.0)
            
            # Generate CR using Normal distribution: N(mCR, 0.1)
            CR = np.random.normal(mCR, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # 2. Mutation: current-to-pbest/1
            # Sort population to find p-best
            sorted_idx = np.argsort(fitness)
            # p is typically 0.11 (top 11%)
            num_pbest = max(2, int(0.11 * pop_size))
            pbest_indices = sorted_idx[:num_pbest]
            
            # Select random pbest for each individual
            pbest_selection = np.random.choice(pbest_indices, pop_size)
            x_pbest = pop[pbest_selection]
            
            # Select r1 (from Pop, r1 != i)
            r1 = np.random.randint(0, pop_size, pop_size)
            cols = r1 == np.arange(pop_size)
            while np.any(cols):
                r1[cols] = np.random.randint(0, pop_size, np.sum(cols))
                cols = r1 == np.arange(pop_size)
            x_r1 = pop[r1]
            
            # Select r2 (from Pop U Archive, r2 != i, r2 != r1)
            n_total = pop_size + n_arch
            r2 = np.random.randint(0, n_total, pop_size)
            
            # Handling r2 collisions (simplified for speed)
            # We only strictly enforce r2 != i and r2 != r1 if r2 is in current pop
            r2_is_pop = r2 < pop_size
            c1 = (r2 == np.arange(pop_size)) & r2_is_pop
            c2 = (r2 == r1) & r2_is_pop
            while np.any(c1 | c2):
                redo = c1 | c2
                r2[redo] = np.random.randint(0, n_total, np.sum(redo))
                r2_is_pop = r2 < pop_size
                c1 = (r2 == np.arange(pop_size)) & r2_is_pop
                c2 = (r2 == r1) & r2_is_pop
                
            # Construct x_r2
            x_r2 = np.zeros((pop_size, dim))
            mask_pop = r2 < pop_size
            x_r2[mask_pop] = pop[r2[mask_pop]]
            if n_arch > 0:
                # Indices in archive
                mask_arch = ~mask_pop
                x_r2[mask_arch] = archive[r2[mask_arch] - pop_size]
            
            # Compute Mutant Vector
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, lb, ub)
            
            # 3. Crossover (Binomial)
            rand_j = np.random.rand(pop_size, dim)
            mask = rand_j < CR[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            
            # 4. Selection
            trial_fitness = np.zeros(pop_size)
            for i in range(pop_size):
                if time.time() - start_time > time_limit:
                    return best_fitness
                trial_fitness[i] = func(trial[i])
                if trial_fitness[i] < best_fitness:
                    best_fitness = trial_fitness[i]
            
            # Identify improvements
            mask_better = trial_fitness <= fitness
            n_better = np.sum(mask_better)
            
            # 5. Update Archive, Memory, and Population
            if n_better > 0:
                # Add parents to archive before they are replaced
                replaced = pop[mask_better]
                if n_arch < pop_size:
                    n_space = pop_size - n_arch
                    if n_better <= n_space:
                        archive[n_arch:n_arch+n_better] = replaced
                        n_arch += n_better
                    else:
                        # Fill rest of space
                        archive[n_arch:] = replaced[:n_space]
                        n_arch = pop_size
                        # Replace random spots with remainder
                        n_rem = n_better - n_space
                        idx_rep = np.random.randint(0, pop_size, n_rem)
                        archive[idx_rep] = replaced[n_space:]
                else:
                    # Archive full, replace random spots
                    idx_rep = np.random.randint(0, pop_size, n_better)
                    archive[idx_rep] = replaced
                
                # Update History Memory (Weighted Lehmer Mean)
                # We calculate weights based on fitness improvement
                diff_fit = fitness[mask_better] - trial_fitness[mask_better]
                # Filter out zero improvement (if any) to avoid div by zero
                # Though <= implies potentially 0.
                valid_diff = diff_fit > 0
                if np.any(valid_diff):
                    diff_fit = diff_fit[valid_diff]
                    w = diff_fit / np.sum(diff_fit)
                    
                    scr = CR[mask_better][valid_diff]
                    sf = F[mask_better][valid_diff]
                    
                    mCR_new = np.sum(w * scr)
                    mF_new = np.sum(w * sf**2) / (np.sum(w * sf) + 1e-9)
                    
                    mem_CR[k_mem] = mCR_new
                    mem_F[k_mem] = mF_new
                    k_mem = (k_mem + 1) % H
                
                # Update Population
                pop[mask_better] = trial[mask_better]
                fitness[mask_better] = trial_fitness[mask_better]
            
            # 6. Restart Criteria
            current_gen_best = np.min(fitness)
            if np.abs(current_gen_best - last_gen_best) < 1e-9:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_gen_best = current_gen_best
                
            # Restart if variance is extremely low or stagnation is prolonged
            if np.std(fitness) < 1e-8 or stagnation_counter > 40:
                break

    return best_fitness
