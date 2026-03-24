#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 24.19030358384603
#
# 1. algorithm code is:
##The output values and codes for the last generated algorithms are as follows:
##1. output value is: 129.05719788297492
##
## 1. algorithm code is:
###Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is robust for global optimization problems and works well within time constraints by iteratively improving a population of solutions.
###
##import numpy as np
##import time
##
##def run(func, dim, bounds, max_time):
##    """
##    Minimizes a function using Differential Evolution (DE) within a time limit.
##    """
##    
##    # --- Configuration ---
##    # Population size: Higher allows more exploration but slower generations.
##    # 15 * dim is a standard rule of thumb. 
##    # We clip it to ensure it's at least 5 (needed for mutation logic) and not excessive.
##    pop_size = max(5, int(15 * dim))
##    
##    # DE Hyperparameters
##    mutation_factor = 0.8  # (F) Weighting of the difference vector (0.5 - 1.0)
##    crossover_prob = 0.7   # (CR) Probability of crossover (0.5 - 1.0)
##    
##    # Timing
##    start_time = time.time()
##    
##    # --- Initialization ---
##    
##    # Prepare bounds as numpy arrays for vectorization
##    bounds_np = np.array(bounds)
##    lower_bounds = bounds_np[:, 0]
##    upper_bounds = bounds_np[:, 1]
##    diff_bounds = upper_bounds - lower_bounds
##    
##    # Initialize population: pop_size x dim
##    # Random values scaled to the bounds
##    population = lower_bounds + np.random.rand(pop_size, dim) * diff_bounds
##    
##    # Evaluate initial population
##    fitness = np.zeros(pop_size)
##    best_idx = 0
##    best_fitness = float('inf')
##    
##    for i in range(pop_size):
##        # Check time constraint during initialization
##        if (time.time() - start_time) >= max_time:
##            # If we time out during init, return best found so far (or inf if none)
##            return best_fitness if best_fitness != float('inf') else func(population[i])
##            
##        val = func(population[i])
##        fitness[i] = val
##        
##        if val < best_fitness:
##            best_fitness = val
##            best_idx = i
##
##    # --- Main Optimization Loop ---
##    
##    while True:
##        # Loop over the population
##        for i in range(pop_size):
##            
##            # 1. Check Time Constraint strictly inside the loop
##            if (time.time() - start_time) >= max_time:
##                return best_fitness
##
##            # 2. Mutation
##            # Select 3 random distinct indices (a, b, c) that are not i
##            candidates = list(range(pop_size))
##            candidates.remove(i)
##            a, b, c = np.random.choice(candidates, 3, replace=False)
##            
##            # Create mutant vector: V = X_a + F * (X_b - X_c)
##            x_a = population[a]
##            x_b = population[b]
##            x_c = population[c]
##            
##            mutant = x_a + mutation_factor * (x_b - x_c)
##            
##            # Clip mutant to ensure it stays within bounds
##            mutant = np.clip(mutant, lower_bounds, upper_bounds)
##            
##            # 3. Crossover
##            # Create trial vector by mixing target vector (population[i]) and mutant
##            cross_points = np.random.rand(dim) < crossover_prob
##            
##            # Ensure at least one parameter is changed from the mutant to avoid stagnation
##            if not np.any(cross_points):
##                cross_points[np.random.randint(0, dim)] = True
##                
##            trial = np.where(cross_points, mutant, population[i])
##            
##            # 4. Selection
##            f_trial = func(trial)
##            
##            # Greedy selection: if trial is better or equal, replace current individual
##            if f_trial <= fitness[i]:
##                fitness[i] = f_trial
##                population[i] = trial
##                
##                # Update global best if needed
##                if f_trial < best_fitness:
##                    best_fitness = f_trial
##        
##    return best_fitness
##
##
##
##The output value of the best generated algorithm is: 129.05719788297492
##
## The best generated algorithm code:
###Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is robust for global optimization problems and works well within time constraints by iteratively improving a population of solutions.
###
##import numpy as np
##import time
##
##def run(func, dim, bounds, max_time):
##    """
##    Minimizes a function using Differential Evolution (DE) within a time limit.
##    """
##    
##    # --- Configuration ---
##    # Population size: Higher allows more exploration but slower generations.
##    # 15 * dim is a standard rule of thumb. 
##    # We clip it to ensure it's at least 5 (needed for mutation logic) and not excessive.
##    pop_size = max(5, int(15 * dim))
##    
##    # DE Hyperparameters
##    mutation_factor = 0.8  # (F) Weighting of the difference vector (0.5 - 1.0)
##    crossover_prob = 0.7   # (CR) Probability of crossover (0.5 - 1.0)
##    
##    # Timing
##    start_time = time.time()
##    
##    # --- Initialization ---
##    
##    # Prepare bounds as numpy arrays for vectorization
##    bounds_np = np.array(bounds)
##    lower_bounds = bounds_np[:, 0]
##    upper_bounds = bounds_np[:, 1]
##    diff_bounds = upper_bounds - lower_bounds
##    
##    # Initialize population: pop_size x dim
##    # Random values scaled to the bounds
##    population = lower_bounds + np.random.rand(pop_size, dim) * diff_bounds
##    
##    # Evaluate initial population
##    fitness = np.zeros(pop_size)
##    best_idx = 0
##    best_fitness = float('inf')
##    
##    for i in range(pop_size):
##        # Check time constraint during initialization
##        if (time.time() - start_time) >= max_time:
##            # If we time out during init, return best found so far (or inf if none)
##            return best_fitness if best_fitness != float('inf') else func(population[i])
##            
##        val = func(population[i])
##        fitness[i] = val
##        
##        if val < best_fitness:
##            best_fitness = val
##            best_idx = i
##
##    # --- Main Optimization Loop ---
##    
##    while True:
##        # Loop over the population
##        for i in range(pop_size):
##            
##            # 1. Check Time Constraint strictly inside the loop
##            if (time.time() - start_time) >= max_time:
##                return best_fitness
##
##            # 2. Mutation
##            # Select 3 random distinct indices (a, b, c) that are not i
##            candidates = list(range(pop_size))
##            candidates.remove(i)
##            a, b, c = np.random.choice(candidates, 3, replace=False)
##            
##            # Create mutant vector: V = X_a + F * (X_b - X_c)
##            x_a = population[a]
##            x_b = population[b]
##            x_c = population[c]
##            
##            mutant = x_a + mutation_factor * (x_b - x_c)
##            
##            # Clip mutant to ensure it stays within bounds
##            mutant = np.clip(mutant, lower_bounds, upper_bounds)
##            
##            # 3. Crossover
##            # Create trial vector by mixing target vector (population[i]) and mutant
##            cross_points = np.random.rand(dim) < crossover_prob
##            
##            # Ensure at least one parameter is changed from the mutant to avoid stagnation
##            if not np.any(cross_points):
##                cross_points[np.random.randint(0, dim)] = True
##                
##            trial = np.where(cross_points, mutant, population[i])
##            
##            # 4. Selection
##            f_trial = func(trial)
##            
##            # Greedy selection: if trial is better or equal, replace current individual
##            if f_trial <= fitness[i]:
##                fitness[i] = f_trial
##                population[i] = trial
##                
##                # Update global best if needed
##                if f_trial < best_fitness:
##                    best_fitness = f_trial
##        
##    return best_fitness
##
##
##Improve.
##
#import numpy as np
#import time
#
#def run(func, dim, bounds, max_time):
#    """
#    Minimizes a function using Self-Adaptive Differential Evolution (jDE) 
#    with Latin Hypercube Sampling initialization and restart mechanism.
#    
#    Improvements over standard DE:
#    1. jDE: Self-adapts mutation (F) and crossover (CR) parameters for each individual, 
#       removing the need for manual tuning and allowing dynamic balance of exploration/exploitation.
#    2. Latin Hypercube Sampling (LHS): Initializes population more uniformly than random sampling.
#    3. Restart Mechanism: Re-initializes population if diversity drops (convergence) 
#       to escape local minima, while preserving the best found solution.
#    """
#    start_time = time.time()
#    
#    # --- Configuration ---
#    # Population size: Balance between exploration and iteration speed.
#    # We use 10*dim (min 20) to allow for more generations and faster convergence per restart.
#    pop_size = max(20, int(10 * dim))
#    
#    # Helper for Time Check
#    def time_is_up():
#        return (time.time() - start_time) >= max_time
#
#    # Pre-process bounds
#    bounds_np = np.array(bounds)
#    lower_b = bounds_np[:, 0]
#    upper_b = bounds_np[:, 1]
#    diff_b = upper_b - lower_b
#    
#    # Global best tracker
#    best_fitness = float('inf')
#    best_sol = None
#
#    # --- Initialization with Latin Hypercube Sampling (LHS) ---
#    # LHS ensures the initial population covers the space more uniformly than random.
#    population = np.zeros((pop_size, dim))
#    for d in range(dim):
#        # Create strata
#        edges = np.linspace(lower_b[d], upper_b[d], pop_size + 1)
#        # Sample uniformly within strata
#        points = np.random.uniform(edges[:-1], edges[1:])
#        # Shuffle to break correlation between dimensions
#        np.random.shuffle(points)
#        population[:, d] = points
#
#    fitness = np.zeros(pop_size)
#    
#    # Evaluate Initial Population
#    for i in range(pop_size):
#        if time_is_up():
#            return best_fitness if best_sol is not None else float('inf')
#        
#        val = func(population[i])
#        fitness[i] = val
#        
#        if val < best_fitness:
#            best_fitness = val
#            best_sol = population[i].copy()
#
#    # --- jDE Parameters ---
#    # Each individual maintains its own control parameters F and CR
#    F = np.full(pop_size, 0.5)
#    CR = np.full(pop_size, 0.9)
#    tau1 = 0.1 # Probability to update F
#    tau2 = 0.1 # Probability to update CR
#    
#    # --- Main Optimization Loop ---
#    while not time_is_up():
#        
#        # 1. Restart Mechanism
#        # If population diversity is too low (convergence), restart while keeping the elite.
#        # We use standard deviation of fitness as a computationally cheap proxy for convergence.
#        if np.std(fitness) < 1e-6 and not time_is_up():
#            # Re-initialize population randomly
#            population = lower_b + np.random.rand(pop_size, dim) * diff_b
#            # Keep the global best solution (Elitism)
#            population[0] = best_sol
#            fitness[0] = best_fitness
#            
#            # Reset adaptive parameters for the new population
#            F = np.full(pop_size, 0.5)
#            CR = np.full(pop_size, 0.9)
#            
#            # Re-evaluate new population (skipping index 0 which is already evaluated)
#            for i in range(1, pop_size):
#                if time_is_up(): return best_fitness
#                val = func(population[i])
#                fitness[i] = val
#                if val < best_fitness:
#                    best_fitness = val
#                    best_sol = population[i].copy()
#            continue
#
#        # 2. Adaptation Step (jDE Logic)
#        # Generate potential new parameters for trial vectors
#        # mask_F/CR determines which individuals update their parameters
#        mask_F = np.random.rand(pop_size) < tau1
#        mask_CR = np.random.rand(pop_size) < tau2
#        
#        # Sample new values: F in [0.1, 1.0], CR in [0.0, 1.0]
#        F_new = 0.1 + 0.9 * np.random.rand(pop_size)
#        CR_new = np.random.rand(pop_size)
#        
#        # Apply updates temporarily for trial vector generation
#        F_trial = np.where(mask_F, F_new, F)
#        CR_trial = np.where(mask_CR, CR_new, CR)
#        
#        # 3. Mutation, Crossover, and Selection
#        # We iterate to allow strict time checking inside the generation
#        for i in range(pop_size):
#            if time_is_up(): return best_fitness
#            
#            # Select 3 distinct random indices distinct from i
#            # Using a list comprehension + random choice is robust
#            idxs = [idx for idx in range(pop_size) if idx != i]
#            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
#            
#            # Mutation: DE/rand/1
#            mutant = population[r1] + F_trial[i] * (population[r2] - population[r3])
#            
#            # Boundary Constraint Handling (Clip)
#            mutant = np.clip(mutant, lower_b, upper_b)
#            
#            # Binomial Crossover
#            cross_points = np.random.rand(dim) < CR_trial[i]
#            # Ensure at least one variable is changed to avoid duplicating the parent
#            j_rand = np.random.randint(dim)
#            cross_points[j_rand] = True
#            
#            trial = np.where(cross_points, mutant, population[i])
#            
#            # Evaluation
#            f_trial = func(trial)
#            
#            # Selection (Greedy)
#            if f_trial <= fitness[i]:
#                population[i] = trial
#                fitness[i] = f_trial
#                # If trial was successful, update the individual's F and CR
#                F[i] = F_trial[i]
#                CR[i] = CR_trial[i]
#                
#                # Update global best
#                if f_trial < best_fitness:
#                    best_fitness = f_trial
#                    best_sol = trial.copy()
#                    
#    return best_fitness
#
#
#2. output value is: 129.05719788297492
#
# 2. algorithm code is:
##Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is robust for global optimization problems and works well within time constraints by iteratively improving a population of solutions.
##
#import numpy as np
#import time
#
#def run(func, dim, bounds, max_time):
#    """
#    Minimizes a function using Differential Evolution (DE) within a time limit.
#    """
#    
#    # --- Configuration ---
#    # Population size: Higher allows more exploration but slower generations.
#    # 15 * dim is a standard rule of thumb. 
#    # We clip it to ensure it's at least 5 (needed for mutation logic) and not excessive.
#    pop_size = max(5, int(15 * dim))
#    
#    # DE Hyperparameters
#    mutation_factor = 0.8  # (F) Weighting of the difference vector (0.5 - 1.0)
#    crossover_prob = 0.7   # (CR) Probability of crossover (0.5 - 1.0)
#    
#    # Timing
#    start_time = time.time()
#    
#    # --- Initialization ---
#    
#    # Prepare bounds as numpy arrays for vectorization
#    bounds_np = np.array(bounds)
#    lower_bounds = bounds_np[:, 0]
#    upper_bounds = bounds_np[:, 1]
#    diff_bounds = upper_bounds - lower_bounds
#    
#    # Initialize population: pop_size x dim
#    # Random values scaled to the bounds
#    population = lower_bounds + np.random.rand(pop_size, dim) * diff_bounds
#    
#    # Evaluate initial population
#    fitness = np.zeros(pop_size)
#    best_idx = 0
#    best_fitness = float('inf')
#    
#    for i in range(pop_size):
#        # Check time constraint during initialization
#        if (time.time() - start_time) >= max_time:
#            # If we time out during init, return best found so far (or inf if none)
#            return best_fitness if best_fitness != float('inf') else func(population[i])
#            
#        val = func(population[i])
#        fitness[i] = val
#        
#        if val < best_fitness:
#            best_fitness = val
#            best_idx = i
#
#    # --- Main Optimization Loop ---
#    
#    while True:
#        # Loop over the population
#        for i in range(pop_size):
#            
#            # 1. Check Time Constraint strictly inside the loop
#            if (time.time() - start_time) >= max_time:
#                return best_fitness
#
#            # 2. Mutation
#            # Select 3 random distinct indices (a, b, c) that are not i
#            candidates = list(range(pop_size))
#            candidates.remove(i)
#            a, b, c = np.random.choice(candidates, 3, replace=False)
#            
#            # Create mutant vector: V = X_a + F * (X_b - X_c)
#            x_a = population[a]
#            x_b = population[b]
#            x_c = population[c]
#            
#            mutant = x_a + mutation_factor * (x_b - x_c)
#            
#            # Clip mutant to ensure it stays within bounds
#            mutant = np.clip(mutant, lower_bounds, upper_bounds)
#            
#            # 3. Crossover
#            # Create trial vector by mixing target vector (population[i]) and mutant
#            cross_points = np.random.rand(dim) < crossover_prob
#            
#            # Ensure at least one parameter is changed from the mutant to avoid stagnation
#            if not np.any(cross_points):
#                cross_points[np.random.randint(0, dim)] = True
#                
#            trial = np.where(cross_points, mutant, population[i])
#            
#            # 4. Selection
#            f_trial = func(trial)
#            
#            # Greedy selection: if trial is better or equal, replace current individual
#            if f_trial <= fitness[i]:
#                fitness[i] = f_trial
#                population[i] = trial
#                
#                # Update global best if needed
#                if f_trial < best_fitness:
#                    best_fitness = f_trial
#        
#    return best_fitness
#
#
#
#The output value of the best generated algorithm is: 24.19030358384603
#
# The best generated algorithm code:
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive Differential Evolution)
    with Latin Hypercube Sampling, External Archive, and a Restart Mechanism.
    
    Features:
    - SHADE: Uses historical memory to adapt F and CR parameters.
    - current-to-pbest/1 mutation: Balances exploitation (pbest) and exploration.
    - External Archive: Maintains population diversity by storing inferior solutions.
    - Restart: Re-initializes population if stagnation is detected, preserving the elite solution.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Higher allows better exploration.
    pop_size = max(30, int(15 * dim))
    
    # SHADE Memory parameters
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive
    # Stores displaced parent vectors to maintain diversity in mutation
    arc_rate = 2.0
    max_arc_size = int(pop_size * arc_rate)
    archive = np.empty((max_arc_size, dim))
    arc_count = 0
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    diff_b = upper_b - lower_b
    
    # Global Best
    best_fitness = float('inf')
    best_sol = None
    
    # --- Helper: Time Check ---
    def time_is_up():
        return (time.time() - start_time) >= max_time

    # --- Initialization (LHS) ---
    def init_population():
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(lower_b[d], upper_b[d], pop_size + 1)
            points = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points)
            pop[:, d] = points
        return pop

    population = init_population()
    fitness = np.full(pop_size, float('inf'))

    # Evaluate Initial Population
    for i in range(pop_size):
        if time_is_up():
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()

    # --- Main Loop ---
    while not time_is_up():
        
        # 1. Restart Mechanism
        # If population diversity is too low (standard deviation of fitness), restart.
        if np.std(fitness) < 1e-6:
            # Re-init population
            population = init_population()
            # Elitism: Keep best found solution
            population[0] = best_sol
            
            # Reset Memory and Archive for fresh start
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            k_mem = 0
            arc_count = 0
            
            # Re-evaluate
            fitness[0] = best_fitness
            for i in range(1, pop_size):
                if time_is_up(): return best_fitness
                val = func(population[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = population[i].copy()
            continue

        # 2. Parameter Generation (SHADE)
        # Select memory index for each individual
        idx_mem = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[idx_mem]
        mu_f = mem_f[idx_mem]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        # Use standard_cauchy and scale. Retry if <= 0.
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        while True:
            mask_le0 = f <= 0
            if not np.any(mask_le0): break
            f[mask_le0] = np.random.standard_cauchy(np.sum(mask_le0)) * 0.1 + mu_f[mask_le0]
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: current-to-pbest/1
        # Sort to find p-best
        sorted_indices = np.argsort(fitness)
        # Randomize p slightly for robustness (L-SHADE style)
        p_val = np.random.uniform(0.05, 0.2)
        num_pbest = max(2, int(p_val * pop_size))
        
        # Select x_pbest
        pbest_choices = np.random.randint(0, num_pbest, pop_size)
        idx_pbest = sorted_indices[pbest_choices]
        x_pbest = population[idx_pbest]
        
        # Select x_r1 (random from population)
        idx_r1 = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[idx_r1]
        
        # Select x_r2 (random from Population U Archive)
        if arc_count > 0:
            union_pop = np.concatenate((population, archive[:arc_count]), axis=0)
        else:
            union_pop = population
            
        idx_r2 = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[idx_r2]
        
        # Compute Mutant Vectors
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        f_col = f[:, None]
        mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
        mutant = np.clip(mutant, lower_b, upper_b)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        # Force at least one dimension from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # 5. Selection and Memory Update
        success_cr = []
        success_f = []
        diff_fitness = []
        
        for i in range(pop_size):
            if time_is_up(): return best_fitness
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Improvement or equal: Update Population
                
                # Add parent to archive if trial is strictly better
                if f_trial < fitness[i]:
                    if arc_count < max_arc_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Replace random member
                        rand_idx = np.random.randint(0, max_arc_size)
                        archive[rand_idx] = population[i].copy()
                        
                    # Record success parameters
                    success_cr.append(cr[i])
                    success_f.append(f[i])
                    diff_fitness.append(fitness[i] - f_trial)
                
                fitness[i] = f_trial
                population[i] = trial[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
                    
        # 6. Update History Memory
        if len(success_cr) > 0:
            s_cr = np.array(success_cr)
            s_f = np.array(success_f)
            d_f = np.array(diff_fitness)
            
            # Weighted Lehmer Mean
            total_imp = np.sum(d_f)
            if total_imp > 0:
                weights = d_f / total_imp
                
                # Update F (Lehmer mean: sum(w*f^2)/sum(w*f))
                mean_f = np.sum(weights * (s_f**2)) / np.sum(weights * s_f)
                mem_f[k_mem] = mean_f
                
                # Update CR (Weighted mean)
                mean_cr = np.sum(weights * s_cr)
                mem_cr[k_mem] = mean_cr
                
                k_mem = (k_mem + 1) % H

    return best_fitness
