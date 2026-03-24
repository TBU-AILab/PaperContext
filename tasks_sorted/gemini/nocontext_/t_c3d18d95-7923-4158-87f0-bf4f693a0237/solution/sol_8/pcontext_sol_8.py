#Here is a fully operational, self-contained Python algorithm based on **Differential Evolution**. This method is robust for global optimization, handling non-differentiable functions well, and is much more efficient than random search.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm
    within a specified time limit.
    """
    
    # --- Configuration ---
    # Record start time to manage the time budget
    start_time = time.time()
    
    # DE Hyperparameters
    # Population size: heuristic is 10-15 * dim. 
    # We cap it to ensure the algorithm gets enough generations if dim is high.
    pop_size = max(10, 10 * dim)
    if pop_size > 50: 
        pop_size = 50
        
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_prob = 0.7   # Crossover probability (CR)

    # --- Initialization ---
    
    # Convert bounds to numpy arrays for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population: random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness array to store cost of each agent
    fitness = np.full(pop_size, float('inf'))
    
    # Track global best
    best = float('inf')
    best_vector = None

    # Evaluate Initial Population
    # We check time strictly here in case the function is very slow
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val
            best_vector = population[i].copy()

    # --- Main Optimization Loop ---
    while True:
        # Check overall time budget at the start of the generation
        if time.time() - start_time >= max_time:
            break
            
        for i in range(pop_size):
            # Check time budget before every function call for safety
            if time.time() - start_time >= max_time:
                return best

            # 1. Mutation (Strategy: rand/1)
            # Pick 3 distinct random agents (a, b, c) excluding current agent i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Mutant vector = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant vector to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover (Binomial)
            # Select indices to replace with mutant values
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one dimension comes from the mutant (standard DE requirement)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            # Create trial vector
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            # Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better or equal, replace current
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if found
                if f_trial < best:
                    best = f_trial
                    best_vector = trial.copy()

    return best
