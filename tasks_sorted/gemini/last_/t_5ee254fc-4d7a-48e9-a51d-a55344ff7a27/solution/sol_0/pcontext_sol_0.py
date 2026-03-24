import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the set of input parameter values that lead to the minimum output value
    within a limited time using the Differential Evolution algorithm.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: 15 * dim provides a good balance between diversity and speed
    pop_size = max(10, 15 * dim)
    mutation_factor = 0.8  # F factor
    crossover_prob = 0.9   # CR factor
    
    # Convert bounds to numpy arrays for efficient vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = np.abs(max_b - min_b)
    
    # Initialize Population
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness array
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Time check inside initialization loop to ensure responsiveness
        if (datetime.now() - start_time) >= time_limit:
            # If time runs out during init, return best found so far
            return best_fitness if best_fitness != float('inf') else func(population[0])
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # Main Differential Evolution Loop
    while True:
        for i in range(pop_size):
            # Check time limit
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness

            # Mutation: Select 3 distinct random individuals (a, b, c) excluding current (i)
            # We generate random indices until we find valid ones
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Compute mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # Crossover: Create trial vector
            # Pick random numbers for every dimension
            rand_vals = np.random.rand(dim)
            
            # Create a mask where crossover occurs
            mask = rand_vals < crossover_prob
            
            # Ensure at least one parameter comes from the mutant (standard DE requirement)
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            # Construct trial vector
            trial = np.where(mask, mutant, population[i])
            
            # Boundary Handling: Clip trial vector to be within bounds
            trial = np.clip(trial, min_b, max_b)

            # Selection: Evaluate trial vector
            try:
                f_trial = func(trial)
            except Exception:
                # Fallback if func fails for specific parameters
                f_trial = float('inf')

            # Greedy selection: if trial is better, replace current
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best found solution
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
