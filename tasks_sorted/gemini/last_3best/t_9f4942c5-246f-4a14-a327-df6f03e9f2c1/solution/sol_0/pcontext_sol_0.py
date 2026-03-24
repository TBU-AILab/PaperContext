import numpy as np
import time

def run(func, dim, bounds, max_time):
    # Algorithm: Differential Evolution (DE)
    # This is a stochastic global optimization algorithm capable of handling 
    # non-differentiable, non-linear, and multimodal objective functions.
    
    start_time = time.time()
    
    # --- Hyperparameters ---
    # Population size: Kept moderate to ensure generations run within strict time limits.
    # A size of 15-30 is often sufficient for DE to converge reasonably well on many problems.
    pop_size = 20 
    if dim > 10:
        pop_size = 30
        
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_prob = 0.9   # Crossover probability (CR)

    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population
    # Population matrix: (pop_size, dim)
    population = np.zeros((pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    best = np.inf

    # --- Initialization Phase ---
    # Generate random initial vectors and evaluate them.
    # Crucial: Check time limit even during initialization.
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best
        
        # Generate individual within bounds
        population[i] = min_b + np.random.rand(dim) * diff_b
        
        # Evaluate
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Optimization Loop ---
    while True:
        # Iterate through population
        for i in range(pop_size):
            # Check time constraints
            if (time.time() - start_time) >= max_time:
                return best
            
            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != current (i)
            # Efficiently pick indices
            candidates = list(range(pop_size))
            candidates.remove(i)
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # 2. Crossover: Binomial crossover
            # Select dimensions to inherit from mutant based on CR
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one dimension is taken from mutant (standard DE rule)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Create trial vector
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Boundary Handling
            # Clip values to stay within [low, high]
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            # Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better or equal, replace target
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best found so far
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
