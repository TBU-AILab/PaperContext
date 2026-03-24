#Here is the implementation of the Differential Evolution algorithm, a robust metaheuristic for global optimization of continuous functions, tailored to fit the provided template and constraints.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the output of 'func' using Differential Evolution (DE) within 'max_time'.
    
    DE is chosen for its efficiency in handling non-differentiable, non-linear, 
    and multi-modal optimization problems without requiring gradient information.
    """
    
    start_time = time.time()
    
    # --- DE Hyperparameters ---
    # Population size: A trade-off between diversity and speed.
    # We clamp it between 20 and 60 to ensure the algorithm performs some generations 
    # even if max_time is short or dim is high.
    pop_size = int(np.clip(10 * dim, 20, 60))
    
    # F (Mutation Factor): Controls the amplification of the differential variation.
    # Typically between [0.5, 1.0].
    F = 0.8 
    
    # CR (Crossover Probability): Controls the number of changing parameters.
    # Typically between [0.8, 1.0] for dependent variables.
    CR = 0.9

    # --- Initialization ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population uniformly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track fitness
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Strict time check before every expensive function call
        if time.time() - start_time >= max_time:
            return best
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Loop ---
    # DE/rand/1/bin strategy
    while True:
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best
            
            # 1. Mutation
            # Select 3 distinct random individuals (a, b, c) distinct from current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            mutant = a + F * (b - c)
            
            # 2. Crossover
            # Generate mask based on CR
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter comes from mutant (standard DE guarantee)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Constrain to bounds
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                if f_trial < best:
                    best = f_trial
                    
    return best
