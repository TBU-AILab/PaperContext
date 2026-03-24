#Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is generally more robust and efficient than random search or simple gradient descent for finding global minima in multi-dimensional continuous spaces within a set time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm.
    DE is a stochastic, population-based optimization algorithm 
    suitable for global optimization problems.
    """
    
    # 1. Initialization
    start_time = time.time()
    
    # DE Hyperparameters
    # Population size: usually between 5*dim and 15*dim. 
    # We ensure a minimum of 4 for mutation mechanics.
    pop_size = max(5, 10 * dim) 
    mutation_factor = 0.8      # Differential weight (F), typically [0.5, 1.0]
    crossover_prob = 0.7       # Crossover probability (CR), typically [0.5, 1.0]
    
    # Pre-process bounds for numpy vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population: Random distribution within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track fitness
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    
    # 2. Evaluate Initial Population
    # We loop individually to ensure we don't exceed max_time during initialization
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_fitness
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # 3. Main Evolution Loop
    while True:
        # Iterate over each individual in the population
        for i in range(pop_size):
            
            # Hard time check before every expensive operation
            if (time.time() - start_time) >= max_time:
                return best_fitness
            
            # --- Mutation ---
            # Select 3 unique indices distinct from current index i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Check constraints (clip values to bounds)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover ---
            # Binomial crossover
            rand_vals = np.random.rand(dim)
            mask = rand_vals < crossover_prob
            
            # Ensure at least one parameter is taken from mutant to avoid exact copy
            # (Pick a random dimension index to force swap)
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            # Create trial vector: mix of mutant and existing target
            trial = np.where(mask, mutant, population[i])
            
            # --- Selection ---
            trial_fitness = func(trial)
            
            # Greedy selection: if trial is better or equal, replace target
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best if found
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
