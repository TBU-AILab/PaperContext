#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 129.05719788297492
#   (Standard Differential Evolution)
#2. output value is: 24.19030358384603
#   (jDE with LHS and Restart)
#3. output value is: 1.1117314285781248
#   (SHADE with LHS, Archive, and Restart)
#
#Here is an improved algorithm code. It implements **L-SHADE with Linear Population Size Reduction** combined with a **Local Search (Polish)** mechanism.
#1.  **L-SHADE**: It linearly reduces the population size from a high value (exploration) to a low value (exploitation) as time progresses. This forces convergence.
#2.  **Adaptive Local Search**: A coordinate descent method is triggered whenever a new best solution is found. Crucially, the step size of the local search adapts based on the standard deviation of the current population, allowing it to fine-tune solutions whether the search is in early or late stages.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Success-History Adaptive Differential Evolution with 
    Linear Population Size Reduction) and a population-adaptive Local Search.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population Sizing (Linear Reduction)
    # Start with a large population for exploration, reduce to min_pop_size for fast convergence.
    pop_size_init = max(30, int(18 * dim))
    pop_size_min = 4
    
    # SHADE Memory Parameters
    H = 6  # Memory size
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive (stores inferior solutions to preserve diversity)
    archive = [] 
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    diff_b = upper_b - lower_b
    
    # --- Helper Functions ---
    def check_timeout():
        return (time.time() - start_time) >= max_time

    def init_pop(size):
        # Latin Hypercube Sampling (LHS)
        p = np.zeros((size, dim))
        for d in range(dim):
            edges = np.linspace(lower_b[d], upper_b[d], size + 1)
            vals = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(vals)
            p[:, d] = vals
        return p

    def local_search(x_curr, f_curr, current_pop):
        """
        Coordinate Descent Polish.
        Step sizes are derived from the population's standard deviation 
        to adapt to the current convergence state.
        """
        x_new = x_curr.copy()
        f_new = f_curr
        improved = False
        
        # Calculate adaptive step sizes based on population spread
        pop_std = np.std(current_pop, axis=0)
        
        # Two passes: coarse and fine relative to distribution
        factors = [0.5, 0.05] 
        
        for factor in factors:
            if check_timeout(): break
            
            # Base step on std dev; fallback to domain range if std dev is near zero
            step = pop_std * factor
            mask_small = step < (diff_b * 1e-9)
            if np.any(mask_small):
                step[mask_small] = diff_b[mask_small] * 1e-4
            
            # Iterate over dimensions in random order
            for d in np.random.permutation(dim):
                if check_timeout(): break
                
                old_val = x_new[d]
                
                # Try positive step
                x_new[d] = np.clip(old_val + step[d], lower_b[d], upper_b[d])
                ft = func(x_new)
                
                if ft < f_new:
                    f_new = ft
                    improved = True
                    continue # Keep change, move to next dimension
                
                # Try negative step
                x_new[d] = np.clip(old_val - step[d], lower_b[d], upper_b[d])
                ft = func(x_new)
                
                if ft < f_new:
                    f_new = ft
                    improved = True
                    continue
                
                # Revert if no improvement
                x_new[d] = old_val
                
        return x_new, f_new, improved

    # --- Initialization ---
    pop_size = pop_size_init
    population = init_pop(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    best_sol = None
    best_fitness = float('inf')
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if check_timeout():
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()
            
    # --- Main Optimization Loop ---
    while not check_timeout():
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate progress ratio based on time
        elapsed = time.time() - start_time
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # Calculate target population size
        target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
        target_size = max(pop_size_min, target_size)
        
        # Reduce population if needed
        if pop_size > target_size:
            # Sort by fitness (best at index 0)
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Truncate worst individuals
            population = population[:target_size]
            fitness = fitness[:target_size]
            pop_size = target_size
            
            # Resize Archive (L-SHADE rule: |A| <= |P| * rate)
            arc_max = int(pop_size * 2.0)
            if len(archive) > arc_max:
                import random
                random.shuffle(archive)
                archive = archive[:arc_max]

        # 2. Sort Population
        # Necessary for correct p-best selection
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Sync global best
        if fitness[0] < best_fitness:
            best_fitness = fitness[0]
            best_sol = population[0].copy()
        
        # 3. Parameter Generation (SHADE)
        # Randomly select memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_f = mem_f[r_idx]
        mu_cr = mem_cr[r_idx]
        
        # Generate CR (Normal Distribution)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy Distribution)
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        # Retry if F <= 0
        while True:
            mask = f <= 0
            if not np.any(mask): break
            f[mask] = np.random.standard_cauchy(np.sum(mask)) * 0.1 + mu_f[mask]
        f = np.minimum(f, 1.0)
        
        # 4. Mutation: current-to-pbest/1
        # p is randomized between 0.05 and 0.2 (L-SHADE style exploration/exploitation balance)
        p = np.random.uniform(0.05, 0.2)
        p_num = max(2, int(p * pop_size))
        
        # Select x_pbest (random from top p%)
        idx_pbest = np.random.randint(0, p_num, pop_size)
        x_pbest = population[idx_pbest]
        
        # Select x_r1 (random from population)
        idx_r1 = np.random.randint(0, pop_size, pop_size)
        # Simple shift to avoid self-selection
        mask_self = (idx_r1 == np.arange(pop_size))
        idx_r1[mask_self] = (idx_r1[mask_self] + 1) % pop_size
        x_r1 = population[idx_r1]
        
        # Select x_r2 (random from Population U Archive)
        if len(archive) > 0:
            arc_arr = np.array(archive)
            union_pop = np.concatenate((population, arc_arr), axis=0)
        else:
            union_pop = population
        idx_r2 = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[idx_r2]
        
        # Compute Mutant Vectors
        f_v = f[:, None]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        mutant = np.clip(mutant, lower_b, upper_b)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross = rand_vals < cr[:, None]
        mask_cross[np.arange(pop_size), j_rand] = True
        trial = np.where(mask_cross, mutant, population)
        
        # 6. Evaluation and Selection
        success_f = []
        success_cr = []
        diff_f = []
        
        for i in range(pop_size):
            if check_timeout(): return best_fitness
            
            t_val = func(trial[i])
            
            if t_val <= fitness[i]:
                # Improvement or equal
                if t_val < fitness[i]:
                    archive.append(population[i].copy())
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_f.append(fitness[i] - t_val)
                
                population[i] = trial[i]
                fitness[i] = t_val
                
                if t_val < best_fitness:
                    best_fitness = t_val
                    best_sol = trial[i].copy()
                    
                    # --- Trigger Local Search ---
                    # Polish the new best solution immediately
                    ls_x, ls_f, improved = local_search(best_sol, best_fitness, population)
                    if improved:
                        best_fitness = ls_f
                        best_sol = ls_x
                        population[i] = best_sol
                        fitness[i] = best_fitness

        # Maintain Archive Size
        arc_limit = int(pop_size * 2.0)
        while len(archive) > arc_limit:
            idx_drop = np.random.randint(0, len(archive))
            archive.pop(idx_drop)
            
        # 7. Update Memory (Weighted Lehmer Mean)
        if len(diff_f) > 0:
            w = np.array(diff_f) / np.sum(diff_f)
            sf = np.array(success_f)
            scr = np.array(success_cr)
            
            mean_f = np.sum(w * (sf**2)) / np.sum(w * sf)
            mem_f[k_mem] = mean_f
            
            mean_cr = np.sum(w * scr)
            mem_cr[k_mem] = mean_cr
            
            k_mem = (k_mem + 1) % H
            
        # 8. Restart Mechanism (Soft)
        # Only if population has collapsed (convergence)
        if np.std(fitness) < 1e-9:
             # Keep best, re-initialize others
             # We do NOT reset pop_size to avoid breaking the LPSR curve
             new_p = init_pop(pop_size - 1)
             population[1:] = new_p
             # Set fitness to inf to ensure re-evaluation logic works if needed, 
             # though we evaluate immediately here:
             for k in range(1, pop_size):
                 if check_timeout(): return best_fitness
                 v = func(population[k])
                 fitness[k] = v
                 if v < best_fitness:
                     best_fitness = v
                     best_sol = population[k].copy()
             
             # Reset Memory
             mem_f.fill(0.5)
             mem_cr.fill(0.5)
             archive = []
             
    return best_fitness
