#Here is a fully operational, self-contained Python implementation of the **SHADE (Success-History based Adaptive Differential Evolution)** algorithm. 
#
#SHADE is a state-of-the-art improvement over JADE and standard Differential Evolution. It consistently outperforms them by maintaining a historical memory of successful control parameters ($F$ and $CR$) and utilizing an external archive to maintain population diversity, preventing premature convergence to local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution).
    
    Key Features:
    1. Historical Memory: Learns optimal F (Mutation) and CR (Crossover) parameters over time.
    2. External Archive: Stores previously good solutions to maintain diversity.
    3. Current-to-pBest Mutation: Guides search towards best solutions while using the archive 
       to define difference vectors, balancing exploration and exploitation.
    """
    
    # --- 1. Setup & Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def is_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- 2. Configuration ---
    # Population size: Robust heuristic, clipped for performance on high dims
    pop_size = int(np.clip(10 * dim, 30, 100))
    
    # SHADE Memory Parameters
    H = 5  # Size of historical memory
    mem_cr = np.full(H, 0.5) # Memory for Crossover Probability
    mem_f = np.full(H, 0.5)  # Memory for Mutation Factor
    k_mem = 0                # Memory index pointer
    
    # External Archive (stores replaced inferior solutions)
    archive = [] 
    
    # --- 3. Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Random initial population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if is_timeout(): return best_fitness
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # --- 4. Main Optimization Loop ---
    while not is_timeout():
        
        # A. Parameter Generation
        # Assign a random memory index (1 to H) to each individual
        r_indices = np.random.randint(0, H, pop_size)
        
        # Generate CR_i using Normal Distribution around memory
        cr = np.random.normal(mem_cr[r_indices], 0.1)
        cr = np.clip(cr, 0, 1) # Clip to [0, 1]
        
        # Generate F_i using Cauchy Distribution around memory
        # Cauchy(loc, scale) = loc + scale * tan(pi * (rand - 0.5))
        f = mem_f[r_indices] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Sanitize F values
        f[f > 1] = 1.0  # Clamp at 1.0
        f[f <= 0] = 0.5 # If non-positive, reset to default (robust fallback)
        
        # Sort population to identify top p-best individuals
        sorted_indices = np.argsort(fitness)
        
        # Containers for successful parameters in this generation
        succ_cr = []
        succ_f = []
        diff_fitness = []
        
        # B. Evolution Cycle (Iterate through population)
        for i in range(pop_size):
            if is_timeout(): return best_fitness
            
            # --- Mutation Strategy: DE/current-to-pbest/1 with Archive ---
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # 1. Select p-best (randomly from top p%)
            # p varies randomly between 2/pop_size and 0.2 to diversify pressure
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            top_count = int(pop_size * p_val)
            pbest_idx = np.random.choice(sorted_indices[:max(2, top_count)])
            x_pbest = pop[pbest_idx]
            
            # 2. Select r1 (random from population, distinct from i)
            r1 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            x_r1 = pop[r1]
            
            # 3. Select r2 (random from Union(Population, Archive), distinct from i and r1)
            # This logic avoids constructing the full union array explicitly for speed
            r2_valid = False
            x_r2 = None
            
            current_archive_len = len(archive)
            union_size = pop_size + current_archive_len
            
            while not r2_valid:
                rand_idx = np.random.randint(union_size)
                if rand_idx < pop_size:
                    # Selected from population
                    if rand_idx != i and rand_idx != r1:
                        x_r2 = pop[rand_idx]
                        r2_valid = True
                else:
                    # Selected from archive
                    x_r2 = archive[rand_idx - pop_size]
                    r2_valid = True
            
            # Compute Mutant Vector
            mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            # Create Trial Vector
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < cr[i]
            mask[j_rand] = True # Ensure at least one dimension is changed
            
            trial = np.where(mask, mutant, pop[i])
            
            # --- Bound Handling ---
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            trial_val = func(trial)
            
            if trial_val < fitness[i]:
                # Improvement found
                improvement = fitness[i] - trial_val
                
                # Add old solution to archive
                archive.append(pop[i].copy())
                
                # Update population
                pop[i] = trial
                fitness[i] = trial_val
                
                # Store successful parameters for memory update
                succ_cr.append(cr[i])
                succ_f.append(f[i])
                diff_fitness.append(improvement)
                
                # Update Global Best
                if trial_val < best_fitness:
                    best_fitness = trial_val
                    best_sol = trial.copy()
        
        # Manage Archive Size (Randomly remove if it exceeds population size)
        while len(archive) > pop_size:
            archive.pop(np.random.randint(len(archive)))

        # C. Update Historical Memory
        if len(succ_cr) > 0:
            succ_cr = np.array(succ_cr)
            succ_f = np.array(succ_f)
            diff_fitness = np.array(diff_fitness)
            
            # Calculate weights based on fitness improvement magnitude
            # Better improvements have higher influence on memory update
            total_diff = np.sum(diff_fitness)
            weights = diff_fitness / total_diff
            
            # Update Memory CR (Weighted Mean)
            mean_cr = np.sum(weights * succ_cr)
            
            # Update Memory F (Weighted Lehmer Mean)
            # Lehmer mean gives more weight to larger F values, helpful for exploration
            mean_f = np.sum(weights * (succ_f**2)) / np.sum(weights * succ_f)
            
            # Update memory slot k
            mem_cr[k_mem] = mean_cr
            mem_f[k_mem] = mean_f
            
            # Advance memory pointer (circular buffer)
            k_mem = (k_mem + 1) % H
            
        # D. Soft Restart (Anti-Stagnation)
        # If population variance is extremely low, randomize the worst 50% to inject diversity
        if np.std(fitness) < 1e-9:
            worst_indices = sorted_indices[pop_size//2:]
            for idx in worst_indices:
                pop[idx] = min_b + np.random.rand(dim) * diff_b
                # Re-evaluate immediately to keep logic consistent
                if is_timeout(): return best_fitness
                fitness[idx] = func(pop[idx])

    return best_fitness
