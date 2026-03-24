#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 129.05719788297492
#
# 1. algorithm code is:
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
#The output value of the best generated algorithm is: 129.05719788297492
#
# The best generated algorithm code:
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
#Improve.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE) 
    with Latin Hypercube Sampling initialization and restart mechanism.
    
    Improvements over standard DE:
    1. jDE: Self-adapts mutation (F) and crossover (CR) parameters for each individual, 
       removing the need for manual tuning and allowing dynamic balance of exploration/exploitation.
    2. Latin Hypercube Sampling (LHS): Initializes population more uniformly than random sampling.
    3. Restart Mechanism: Re-initializes population if diversity drops (convergence) 
       to escape local minima, while preserving the best found solution.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Balance between exploration and iteration speed.
    # We use 10*dim (min 20) to allow for more generations and faster convergence per restart.
    pop_size = max(20, int(10 * dim))
    
    # Helper for Time Check
    def time_is_up():
        return (time.time() - start_time) >= max_time

    # Pre-process bounds
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    diff_b = upper_b - lower_b
    
    # Global best tracker
    best_fitness = float('inf')
    best_sol = None

    # --- Initialization with Latin Hypercube Sampling (LHS) ---
    # LHS ensures the initial population covers the space more uniformly than random.
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        # Create strata
        edges = np.linspace(lower_b[d], upper_b[d], pop_size + 1)
        # Sample uniformly within strata
        points = np.random.uniform(edges[:-1], edges[1:])
        # Shuffle to break correlation between dimensions
        np.random.shuffle(points)
        population[:, d] = points

    fitness = np.zeros(pop_size)
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if time_is_up():
            return best_fitness if best_sol is not None else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()

    # --- jDE Parameters ---
    # Each individual maintains its own control parameters F and CR
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    tau1 = 0.1 # Probability to update F
    tau2 = 0.1 # Probability to update CR
    
    # --- Main Optimization Loop ---
    while not time_is_up():
        
        # 1. Restart Mechanism
        # If population diversity is too low (convergence), restart while keeping the elite.
        # We use standard deviation of fitness as a computationally cheap proxy for convergence.
        if np.std(fitness) < 1e-6 and not time_is_up():
            # Re-initialize population randomly
            population = lower_b + np.random.rand(pop_size, dim) * diff_b
            # Keep the global best solution (Elitism)
            population[0] = best_sol
            fitness[0] = best_fitness
            
            # Reset adaptive parameters for the new population
            F = np.full(pop_size, 0.5)
            CR = np.full(pop_size, 0.9)
            
            # Re-evaluate new population (skipping index 0 which is already evaluated)
            for i in range(1, pop_size):
                if time_is_up(): return best_fitness
                val = func(population[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = population[i].copy()
            continue

        # 2. Adaptation Step (jDE Logic)
        # Generate potential new parameters for trial vectors
        # mask_F/CR determines which individuals update their parameters
        mask_F = np.random.rand(pop_size) < tau1
        mask_CR = np.random.rand(pop_size) < tau2
        
        # Sample new values: F in [0.1, 1.0], CR in [0.0, 1.0]
        F_new = 0.1 + 0.9 * np.random.rand(pop_size)
        CR_new = np.random.rand(pop_size)
        
        # Apply updates temporarily for trial vector generation
        F_trial = np.where(mask_F, F_new, F)
        CR_trial = np.where(mask_CR, CR_new, CR)
        
        # 3. Mutation, Crossover, and Selection
        # We iterate to allow strict time checking inside the generation
        for i in range(pop_size):
            if time_is_up(): return best_fitness
            
            # Select 3 distinct random indices distinct from i
            # Using a list comprehension + random choice is robust
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            # Mutation: DE/rand/1
            mutant = population[r1] + F_trial[i] * (population[r2] - population[r3])
            
            # Boundary Constraint Handling (Clip)
            mutant = np.clip(mutant, lower_b, upper_b)
            
            # Binomial Crossover
            cross_points = np.random.rand(dim) < CR_trial[i]
            # Ensure at least one variable is changed to avoid duplicating the parent
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Evaluation
            f_trial = func(trial)
            
            # Selection (Greedy)
            if f_trial <= fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                # If trial was successful, update the individual's F and CR
                F[i] = F_trial[i]
                CR[i] = CR_trial[i]
                
                # Update global best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial.copy()
                    
    return best_fitness
