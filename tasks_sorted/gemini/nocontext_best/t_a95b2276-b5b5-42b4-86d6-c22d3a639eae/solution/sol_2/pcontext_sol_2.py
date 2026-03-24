#Here is a fully operational implementation using the **JADE (Adaptive Differential Evolution)** algorithm.
#
#This algorithm improves upon standard Differential Evolution by self-adapting its control parameters ($F$ and $CR$) during execution. It learns which parameter values result in successful mutations for the specific function landscape. Additionally, it uses a **"current-to-pbest"** mutation strategy, which guides the search towards the best solutions found so far while maintaining population diversity, resulting in faster convergence and lower output values.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using JADE (Adaptive Differential Evolution) 
    with 'current-to-pbest' mutation and dynamic parameter adaptation.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Balance between exploration (size) and speed (generations)
    # 10*dim is standard, but we clip to ensure responsiveness within time limits.
    pop_size = int(10 * dim)
    pop_size = np.clip(pop_size, 20, 75)
    
    # JADE Adaptive Parameter Initialization
    mu_cr = 0.5    # Mean Crossover Rate
    mu_f = 0.5     # Mean Mutation Factor
    c_adapt = 0.1  # Adaptation learning rate
    top_p = 0.05   # Percentage of top individuals to use as p-best
    
    # Pre-allocate arrays for efficiency
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # Initialize population uniformly within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    fitness = np.zeros(pop_size)
    best_fitness = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            # If time runs out during init, return best found (or inf)
            return best_fitness if best_fitness != float('inf') else float('inf')
            
        val = func(population[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val

    # --- Main Optimization Loop ---
    while True:
        # Check overall time before starting a generation overhead
        if (time.time() - start_time) >= max_time:
            return best_fitness
            
        # 1. Parameter Generation
        # Generate F_i and CR_i for each individual based on current means
        # CR ~ Normal(mu_cr, 0.1), clipped to [0, 1]
        cr_vals = np.random.normal(mu_cr, 0.1, pop_size)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        # F ~ Cauchy(mu_f, 0.1). 
        # Note: standard_cauchy gives dist around 0. We scale and shift.
        f_vals = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        # Truncate F: If F <= 0, set to small value (0.05). If F > 1, clip to 1.
        f_vals[f_vals <= 0] = 0.05
        f_vals = np.clip(f_vals, 0, 1)

        # 2. Mutation: current-to-pbest/1
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # Identify p-best individuals (top p%)
        sorted_idx = np.argsort(fitness)
        num_top = max(1, int(pop_size * top_p))
        top_indices = sorted_idx[:num_top]
        
        # Randomly assign a p-best index to each individual
        pbest_indices = np.random.choice(top_indices, pop_size)
        
        # Select r1 and r2 randomly
        # (Standard optimization allows r1, r2 to occasionally collide for speed in Python)
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Vectorized mutation calculation
        x = population
        x_pbest = population[pbest_indices]
        x_r1 = population[r1]
        x_r2 = population[r2]
        
        # Reshape F for broadcasting: (pop_size, 1)
        f_col = f_vals[:, np.newaxis]
        
        # Calculate mutant vectors
        mutants = x + f_col * (x_pbest - x) + f_col * (x_r1 - x_r2)
        
        # Bound Constraints: Clip to search space
        mutants = np.clip(mutants, min_b, max_b)
        
        # 3. Crossover
        # Create trial vectors U based on CR
        rand_matrix = np.random.rand(pop_size, dim)
        
        # Ensure at least one dimension is changed (j_rand)
        j_rand = np.random.randint(0, dim, pop_size)
        j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
        j_rand_mask[np.arange(pop_size), j_rand] = True
        
        cr_col = cr_vals[:, np.newaxis]
        # Condition: rand < CR OR dimension is the forced dimension
        mask = (rand_matrix < cr_col) | j_rand_mask
        
        trials = np.where(mask, mutants, population)
        
        # 4. Selection and Adaptation Lists
        successful_f = []
        successful_cr = []
        
        # Evaluate trials sequentially
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_fitness
            
            trial_fitness = func(trials[i])
            
            # Greedy Selection
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trials[i]
                
                # Update Best
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                
                # Store successful parameters for adaptation
                successful_f.append(f_vals[i])
                successful_cr.append(cr_vals[i])
        
        # 5. Adapt Parameters (Lehmer Mean for F, Arithmetic Mean for CR)
        if len(successful_f) > 0:
            succ_f = np.array(successful_f)
            succ_cr = np.array(successful_cr)
            
            # Update mu_cr
            mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(succ_cr)
            
            # Update mu_f (Lehmer mean: sum(f^2) / sum(f))
            sum_f = np.sum(succ_f)
            if sum_f > 0:
                lehmer_mean = np.sum(succ_f ** 2) / sum_f
                mu_f = (1 - c_adapt) * mu_f + c_adapt * lehmer_mean

    return best_fitness
