#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 129.057
#   (Standard Differential Evolution)
#2. output value is: 24.190
#   (jDE with LHS and Restart)
#3. output value is: 1.112
#   (SHADE with LHS, Archive, and Restart)
#
#Here is an improved algorithm code. It implements **L-SHADE-RSP (Linear Success-History Adaptive Differential Evolution with Restart and Midpoint Bound Correction)**.
#1.  **L-SHADE**: Improves upon SHADE by linearly reducing the population size from a large initial value to a small value over time. This maximizes exploration in the early phase and convergence speed in the late phase.
#2.  **Midpoint Bound Correction**: When a mutant violates a boundary, the value is set to the midpoint between the parent and the bound (instead of clipping). This preserves diversity and search direction near the edges of the search space.
#3.  **Efficiency**: It uses batched time checks and efficient numpy array indexing to minimize overhead.
#4.  **Restart**: If the population converges (low standard deviation) before time runs out, the algorithm restarts with a fresh population (keeping the global best), recalculating the reduction schedule for the remaining time.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Success-History Adaptive Differential Evolution)
    with Restart and Midpoint Bound Correction.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population sizing
    # N_init: Initial population size. 18*dim is a standard L-SHADE heuristic.
    # We constrain it to [30, 250] to ensure robustness across different dimensions/time limits.
    raw_n_init = 18 * dim
    N_init = max(30, min(250, raw_n_init))
    N_min = 4 # Minimum population size at end of run
    
    # Archive size parameter (factor of population size)
    # Stores inferior solutions to maintain diversity.
    arc_rate = 2.0
    
    # SHADE Memory Parameters
    H = 6 # History size
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global Best Tracker
    best_fitness = float('inf')
    best_sol = None
    
    # --- Helper Functions ---
    def check_timeout():
        return (time.time() - start_time) >= max_time
        
    def init_population(size):
        # Latin Hypercube Sampling (LHS)
        pop = np.zeros((size, dim))
        for d in range(dim):
            edges = np.linspace(lb[d], ub[d], size + 1)
            points = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points)
            pop[:, d] = points
        return pop

    # --- Initialization ---
    pop_size = N_init
    population = init_population(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    # Archive (Pre-allocated for max capacity)
    max_arc_size = int(N_init * arc_rate)
    archive = np.zeros((max_arc_size, dim))
    arc_count = 0
    
    # Initial Evaluation
    # Check time less frequently to reduce overhead
    check_freq = max(1, int(pop_size / 5))
    
    for i in range(pop_size):
        if i % check_freq == 0 and check_timeout():
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()
            
    # --- Optimization Loop ---
    # Variables to track linear reduction schedule across restarts
    run_start_time = start_time
    run_total_time = max_time 

    while not check_timeout():
        
        # 1. Linear Population Size Reduction (LPSR)
        curr_time = time.time()
        elapsed = curr_time - run_start_time
        
        if run_total_time < 1e-9: progress = 1.0
        else: progress = elapsed / run_total_time
        
        if progress > 1.0: progress = 1.0
        
        # Calculate target population size based on progress
        target_size = int(round((N_min - N_init) * progress + N_init))
        target_size = max(N_min, target_size)
        
        # Reduce population if needed
        if pop_size > target_size:
            # Sort by fitness
            sort_indices = np.argsort(fitness)
            population = population[sort_indices]
            fitness = fitness[sort_indices]
            
            # Truncate worst individuals
            population = population[:target_size]
            fitness = fitness[:target_size]
            pop_size = target_size
            
            # Reduce Archive active count if needed to maintain ratio
            current_arc_max = int(pop_size * arc_rate)
            if arc_count > current_arc_max:
                arc_count = current_arc_max

        # 2. Parameter Generation (SHADE)
        # Randomly select memory index
        r_idx = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idx]
        mu_f = mem_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        # Retry if F <= 0
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            f[mask_bad] = np.random.standard_cauchy(np.sum(mask_bad)) * 0.1 + mu_f[mask_bad]
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: current-to-pbest/1
        # Sort population to identify p-best
        sort_indices = np.argsort(fitness)
        sorted_pop = population[sort_indices]
        
        # Random p value in [2/N, 0.2]
        p_vals = np.random.uniform(2.0/pop_size, 0.2, pop_size)
        n_pbest = (p_vals * pop_size).astype(int)
        n_pbest = np.maximum(2, n_pbest)
        
        # Select x_pbest
        rand_idx = (np.random.rand(pop_size) * n_pbest).astype(int)
        x_pbest = sorted_pop[rand_idx]
        
        # Select x_r1 (random != i)
        idx_r1 = np.random.randint(0, pop_size, pop_size)
        mask_coll = idx_r1 == np.arange(pop_size)
        idx_r1[mask_coll] = (idx_r1[mask_coll] + 1) % pop_size
        x_r1 = population[idx_r1]
        
        # Select x_r2 (random from Union of Population and Archive)
        union_size = pop_size + arc_count
        idx_r2 = np.random.randint(0, union_size, pop_size)
        
        mask_in_pop = idx_r2 < pop_size
        mask_in_arc = ~mask_in_pop
        
        x_r2 = np.empty((pop_size, dim))
        x_r2[mask_in_pop] = population[idx_r2[mask_in_pop]]
        if np.any(mask_in_arc):
            arc_indices = idx_r2[mask_in_arc] - pop_size
            x_r2[mask_in_arc] = archive[arc_indices]
            
        # Compute Mutant Vectors
        f_v = f[:, None]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        
        # 4. Bound Correction (Midpoint)
        # If mutant exceeds bounds, place it halfway between parent and bound.
        mask_l = mutant < lb
        if np.any(mask_l):
            r, c = np.where(mask_l)
            mutant[r, c] = (population[r, c] + lb[c]) * 0.5
            
        mask_u = mutant > ub
        if np.any(mask_u):
            r, c = np.where(mask_u)
            mutant[r, c] = (population[r, c] + ub[c]) * 0.5
            
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # 6. Evaluation and Selection
        success_f = []
        success_cr = []
        diff_f = []
        
        for i in range(pop_size):
            if i % check_freq == 0 and check_timeout(): return best_fitness
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Improvement or equal
                if f_trial < fitness[i]:
                    # Add parent to archive
                    if arc_count < max_arc_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Replace random archive member
                        k = np.random.randint(0, max_arc_size)
                        archive[k] = population[i].copy()
                        
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_f.append(fitness[i] - f_trial)
                
                fitness[i] = f_trial
                population[i] = trial[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
        
        # 7. Update History Memory (Weighted Lehmer Mean)
        if len(diff_f) > 0:
            w = np.array(diff_f)
            w = w / np.sum(w)
            
            sf = np.array(success_f)
            scr = np.array(success_cr)
            
            mean_f = np.sum(w * (sf**2)) / np.sum(w * sf)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            mean_cr = np.sum(w * scr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 8. Restart Mechanism
        # If population diversity collapses (convergence), restart.
        if np.std(fitness) < 1e-9:
            # Check if sufficient time remains
            curr_time = time.time()
            remaining = max_time - (curr_time - start_time)
            if remaining < 0.05: break 
            
            # Reset schedule for the remaining time
            run_start_time = curr_time
            run_total_time = remaining
            
            # Reset Population size and contents
            pop_size = N_init
            population = init_population(pop_size)
            fitness = np.full(pop_size, float('inf'))
            
            # Elitism: Inject best found solution
            population[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset Memory and Archive
            arc_count = 0
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            k_mem = 0
            
            # Evaluate new population (skip elite at 0)
            for i in range(1, pop_size):
                if i % check_freq == 0 and check_timeout(): return best_fitness
                
                val = func(population[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = population[i].copy()
                    
    return best_fitness
