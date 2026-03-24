#An improved, fully operational, and self-contained algorithm based on **SHADE (Success-History based Adaptive Differential Evolution) with Restarts and LHS Initialization**.
#
#### Improvements & Logic
#1.  **L-SHADE Foundation**: The algorithm implements the core mechanics of SHADE (a top-performing evolutionary algorithm). It uses **Historical Memory** ($M_{CR}, M_F$) to adapt mutation ($F$) and crossover ($CR$) parameters for each individual based on successful past updates.
#2.  **External Archive**: An archive is utilized to maintain population diversity. This allows the usage of the greedy but fast-converging `current-to-pbest/1` mutation strategy without premature stagnation.
#3.  **LHS Initialization**: Unlike standard random initialization, this uses **Latin Hypercube Sampling (LHS)** to generate a stratified initial population. This ensures better coverage of the search space, which was a strength of the previous best algorithm (jDE).
#4.  **Optimized Vectorization**: The Python overhead is minimized by fully vectorizing the generation of candidate indices (`r1`, `r2`) and parameter sampling, avoiding slow loops.
#5.  **Robust Restart Mechanism**: To handle multimodal landscapes and stagnation, the algorithm monitors population variance. If convergence is detected (or stagnation persists), it triggers a restart to explore new basins of attraction within the remaining time.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using SHADE (Success-History based Adaptive DE) with Restarts
    and Latin Hypercube Sampling (LHS) initialization.
    """
    start_time = time.time()
    # Safety buffer to ensure return before timeout
    time_limit = max_time - 0.05
    
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # --- Parameter Setup ---
    # Population size: Fixed robust size to balance exploration/exploitation rate within time limits
    # Clamped between 60 and 120 ensures efficiency for both low and high dimensions
    pop_size = int(np.clip(20 * dim, 60, 120))
    
    # SHADE Memory Parameters
    H = 5  # History size
    
    # Global Best Tracker
    best_val = float('inf')
    
    # --- Main Restart Loop ---
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return best_val
            
        # 1. Initialization: Latin Hypercube Sampling (LHS)
        # Stratified sampling ensures better initial coverage than pure random
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(lb[d], ub[d], pop_size + 1)
            # Uniformly sample within each bin
            samples = np.random.uniform(edges[:-1], edges[1:])
            # Shuffle to mix dimensions
            np.random.shuffle(samples)
            pop[:, d] = samples
            
        # Evaluate Initial Population
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if time.time() - start_time > time_limit:
                return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                
        # Initialize SHADE Memory (F=0.5, CR=0.5 initially)
        mem_M_CR = np.full(H, 0.5)
        mem_M_F = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        # Stores inferior solutions to maintain diversity for mutation
        archive = np.zeros((pop_size, dim))
        n_arch = 0
        
        # Stagnation tracking
        stag_count = 0
        last_gen_best = np.min(fitness)
        
        # --- Evolutionary Cycle ---
        while True:
            # Time Check
            if time.time() - start_time > time_limit:
                return best_val
            
            # 2. Parameter Adaptation (SHADE)
            # Pick random index from memory
            r_idx = np.random.randint(0, H, pop_size)
            r_M_CR = mem_M_CR[r_idx]
            r_M_F = mem_M_F[r_idx]
            
            # Generate CR: Normal(M_CR, 0.1), clipped [0, 1]
            CR = np.random.normal(r_M_CR, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # Generate F: Cauchy(M_F, 0.1)
            # F = loc + scale * tan(pi * (rand - 0.5))
            u_rand = np.random.rand(pop_size)
            F = r_M_F + 0.1 * np.tan(np.pi * (u_rand - 0.5))
            
            # Constraint handling for F
            # If F <= 0, regenerate. If F > 1, clip to 1.
            mask_neg = F <= 0
            while np.any(mask_neg):
                n_neg = np.sum(mask_neg)
                F[mask_neg] = r_M_F[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(n_neg) - 0.5))
                mask_neg = F <= 0
            F = np.minimum(F, 1.0)
            
            # 3. Mutation: current-to-pbest/1
            # Sort population to identify p-best
            sorted_idx = np.argsort(fitness)
            # Top p% individuals (p controls greediness)
            # Randomize p slightly or fix it? Fixed 15% is robust.
            p_count = max(2, int(0.15 * pop_size))
            pbest_indices_pool = sorted_idx[:p_count]
            
            # Assign a pbest for each individual
            pbest_selection = np.random.choice(pbest_indices_pool, pop_size)
            x_pbest = pop[pbest_selection]
            
            # Select r1: Random from population, r1 != i
            idxs = np.arange(pop_size)
            r1 = np.random.randint(0, pop_size, pop_size)
            # Fix collisions r1 == i
            conflict = r1 == idxs
            while np.any(conflict):
                r1[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
                conflict = r1 == idxs
            x_r1 = pop[r1]
            
            # Select r2: Random from Union(Population, Archive), r2 != i, r2 != r1
            n_total = pop_size + n_arch
            r2 = np.random.randint(0, n_total, pop_size)
            
            # Conflict logic:
            # If r2 is in population (r2 < pop_size), it must not equal i or r1.
            # If r2 is in archive (r2 >= pop_size), it is distinct by definition (copy).
            r2_in_pop = r2 < pop_size
            conflict = (r2_in_pop & ((r2 == idxs) | (r2 == r1)))
            while np.any(conflict):
                r2[conflict] = np.random.randint(0, n_total, np.sum(conflict))
                r2_in_pop = r2 < pop_size
                conflict = (r2_in_pop & ((r2 == idxs) | (r2 == r1)))
                
            # Build x_r2 matrix
            x_r2 = np.zeros((pop_size, dim))
            mask_pop = r2 < pop_size
            x_r2[mask_pop] = pop[r2[mask_pop]]
            
            if n_arch > 0:
                mask_arch = ~mask_pop
                # Map global index to archive index
                arch_idx = r2[mask_arch] - pop_size
                x_r2[mask_arch] = archive[arch_idx]
                
            # Compute Mutant Vectors
            # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            F_col = F[:, None]
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            rand_j = np.random.rand(pop_size, dim)
            mask_cross = rand_j < CR[:, None]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[idxs, j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # Bound Handling (Clip)
            trial = np.clip(trial, lb, ub)
            
            # 5. Selection and Evaluation
            trial_fitness = np.zeros(pop_size)
            
            # Evaluate trials
            for i in range(pop_size):
                if time.time() - start_time > time_limit:
                    return best_val
                
                f_t = func(trial[i])
                trial_fitness[i] = f_t
                
                if f_t < best_val:
                    best_val = f_t
                    
            # Identify successful updates
            mask_better = trial_fitness < fitness
            n_better = np.sum(mask_better)
            
            # 6. Update Archive and Memory
            if n_better > 0:
                # Update Archive: Move replaced parents to archive
                parents = pop[mask_better]
                
                if n_arch < pop_size:
                    n_space = pop_size - n_arch
                    if n_better <= n_space:
                        archive[n_arch : n_arch + n_better] = parents
                        n_arch += n_better
                    else:
                        # Fill remaining space
                        archive[n_arch : pop_size] = parents[:n_space]
                        n_arch = pop_size
                        # Replace random spots with remainder
                        rem = n_better - n_space
                        idx_rep = np.random.randint(0, pop_size, rem)
                        archive[idx_rep] = parents[n_space:]
                else:
                    # Archive full, random replacement
                    idx_rep = np.random.randint(0, pop_size, n_better)
                    archive[idx_rep] = parents
                    
                # Update Memory (Weighted Lehmer Mean)
                diff_v = fitness[mask_better] - trial_fitness[mask_better]
                total_diff = np.sum(diff_v)
                
                if total_diff > 0:
                    w = diff_v / total_diff
                    
                    s_CR = CR[mask_better]
                    s_F = F[mask_better]
                    
                    mean_CR = np.sum(w * s_CR)
                    mean_F = np.sum(w * s_F**2) / (np.sum(w * s_F) + 1e-12)
                    
                    mem_M_CR[k_mem] = mean_CR
                    mem_M_F[k_mem] = mean_F
                    k_mem = (k_mem + 1) % H
                
                # Update Population
                pop[mask_better] = trial[mask_better]
                fitness[mask_better] = trial_fitness[mask_better]
                
            # 7. Restart Criteria
            # Check for population stagnation
            current_gen_best = np.min(fitness)
            if np.abs(current_gen_best - last_gen_best) < 1e-10:
                stag_count += 1
            else:
                stag_count = 0
                last_gen_best = current_gen_best
                
            # If variance is negligible or stagnation persists, restart
            if stag_count > 30 or np.std(fitness) < 1e-8:
                break

    return best_val
