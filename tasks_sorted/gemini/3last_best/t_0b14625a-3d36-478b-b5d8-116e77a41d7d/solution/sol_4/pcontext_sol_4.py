#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 129.057
#   (Standard Differential Evolution)
#2. output value is: 24.190
#   (jDE with LHS and Restart)
#3. output value is: 1.112
#   (SHADE with LHS, Archive, and Restart)
#
#Here is an improved algorithm code. It implements **SHADE with Restart and Midpoint Bound Correction**.
#1.  **SHADE**: Retains the Success-History Adaptive parameter mechanism which proved most effective (1.11).
#2.  **Midpoint Bound Correction**: Instead of simply clipping values to bounds (which makes solutions "stick" to the edges), this version sets the value halfway between the parent and the bound. This preserves the search direction and population diversity near boundaries, often critical for global minima located near edges.
#3.  **Efficiency**: Time checks are batched to reduce system call overhead, and population initialization uses Latin Hypercube Sampling (LHS) for better initial coverage.
#4.  **Restart**: A diversity-based restart mechanism is kept to escape local optima in multimodal functions.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History Adaptive Differential Evolution)
    with Latin Hypercube Sampling, External Archive, Restart, and Midpoint Bound Correction.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Standard heuristic 20*D, clamped to ensure sufficient exploration.
    pop_size = max(30, int(20 * dim))
    
    # SHADE Memory Parameters
    H = 6  # History memory size
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive Parameters
    # Stores inferior solutions to maintain diversity for the 'current-to-pbest' mutation
    arc_rate = 2.0
    max_arc_size = int(pop_size * arc_rate)
    archive = np.empty((max_arc_size, dim))
    arc_count = 0
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    
    # Global Best Tracker
    best_fitness = float('inf')
    best_sol = None
    
    # Time Check Helper
    # Check time every N evaluations to reduce system call overhead (significant in Python)
    check_interval = max(1, int(pop_size / 2))

    def get_lhs_population(size):
        # Latin Hypercube Sampling (LHS)
        pop = np.zeros((size, dim))
        for d in range(dim):
            edges = np.linspace(lower_b[d], upper_b[d], size + 1)
            points = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points)
            pop[:, d] = points
        return pop

    # --- Initialization ---
    population = get_lhs_population(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if i % check_interval == 0:
            if (time.time() - start_time) >= max_time:
                return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()

    # --- Main Optimization Loop ---
    while (time.time() - start_time) < max_time:
        
        # 1. Restart Mechanism
        # Trigger if population has collapsed (convergence)
        if np.std(fitness) < 1e-9:
            # Re-initialize population using LHS
            population = get_lhs_population(pop_size)
            # Elitism: carry over the best solution found so far
            population[0] = best_sol.copy()
            fitness.fill(float('inf'))
            fitness[0] = best_fitness
            
            # Reset SHADE memory and archive to allow adaptation to new search phase
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            k_mem = 0
            arc_count = 0
            
            # Evaluate new population (skipping elite at index 0)
            for i in range(1, pop_size):
                if i % check_interval == 0:
                    if (time.time() - start_time) >= max_time:
                        return best_fitness
                
                val = func(population[i])
                fitness[i] = val
                
                if val < best_fitness:
                    best_fitness = val
                    best_sol = population[i].copy()
            continue

        # 2. Parameter Generation (SHADE)
        # Select random memory index
        r_idx = np.random.randint(0, H, pop_size)
        mu_f = mem_f[r_idx]
        mu_cr = mem_cr[r_idx]
        
        # CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(mu_f, 0.1)
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        # Retry if F <= 0
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            f[mask_bad] = np.random.standard_cauchy(np.sum(mask_bad)) * 0.1 + mu_f[mask_bad]
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: current-to-pbest/1
        # Sort population to find top individuals
        sorted_indices = np.argsort(fitness)
        sorted_pop = population[sorted_indices]
        
        # Select x_pbest (random from top p%)
        # p is randomized between 2/N and 0.2 to balance greediness
        p_min = 2.0 / pop_size
        p = np.random.uniform(p_min, 0.2, pop_size)
        p_num = (p * pop_size).astype(int)
        p_num = np.maximum(2, p_num)
        
        # Vectorized selection of pbest
        rand_sel = np.random.randint(0, 100000, pop_size)
        idx_pbest_sorted = rand_sel % p_num
        x_pbest = sorted_pop[idx_pbest_sorted]
        
        # Select x_r1 (random != i)
        idx_r1 = np.random.randint(0, pop_size, pop_size)
        # Handle collisions (r1 == i)
        mask_coll = (idx_r1 == np.arange(pop_size))
        idx_r1[mask_coll] = (idx_r1[mask_coll] + 1) % pop_size
        x_r1 = population[idx_r1]
        
        # Select x_r2 (random from Population U Archive)
        if arc_count > 0:
            archive_active = archive[:arc_count]
            union_pop = np.concatenate((population, archive_active), axis=0)
        else:
            union_pop = population
            
        idx_r2 = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[idx_r2]
        
        # Compute Mutant Vectors
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        diff = (x_pbest - population) + (x_r1 - x_r2)
        mutant = population + f[:, None] * diff
        
        # 4. Bound Handling (Midpoint Correction)
        # If a mutant violates bounds, place it halfway between parent and bound.
        # This is superior to clipping as it maintains search direction.
        mask_l = mutant < lower_b
        if np.any(mask_l):
            rows, cols = np.where(mask_l)
            mutant[rows, cols] = (population[rows, cols] + lower_b[cols]) / 2.0
            
        mask_u = mutant > upper_b
        if np.any(mask_u):
            rows, cols = np.where(mask_u)
            mutant[rows, cols] = (population[rows, cols] + upper_b[cols]) / 2.0
            
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        # Force at least one dimension to come from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # 6. Evaluation and Selection
        success_f = []
        success_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if i % check_interval == 0:
                if (time.time() - start_time) >= max_time:
                    return best_fitness
            
            t_val = func(trial[i])
            
            if t_val <= fitness[i]:
                # Improvement or equal
                if t_val < fitness[i]:
                    # Add parent to archive
                    if arc_count < max_arc_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Replace random archive member
                        rand_idx = np.random.randint(0, max_arc_size)
                        archive[rand_idx] = population[i].copy()
                        
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_fitness.append(fitness[i] - t_val)
                
                fitness[i] = t_val
                population[i] = trial[i]
                
                if t_val < best_fitness:
                    best_fitness = t_val
                    best_sol = trial[i].copy()
        
        # 7. Update Memory (Weighted Lehmer Mean)
        if len(diff_fitness) > 0:
            df = np.array(diff_fitness)
            w = df / np.sum(df)
            
            sf = np.array(success_f)
            scr = np.array(success_cr)
            
            # Update F
            mean_f = np.sum(w * (sf ** 2)) / np.sum(w * sf)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            # Update CR
            mean_cr = np.sum(w * scr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H

    return best_fitness
