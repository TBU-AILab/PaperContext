#Here is a fully operational, self-contained implementation using the **Differential Evolution** algorithm. This method is generally more robust and efficient than random search for finding global minima in continuous parameter spaces within a bounded timeframe.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # We set a population size relative to dimension, but capped 
    # to ensure we can perform generations if the function is computationally expensive.
    pop_size = max(20, int(10 * dim))
    if pop_size > 100:
        pop_size = 100
        
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_prob = 0.7   # Crossover probability (CR)
    
    # Prepare bounds as numpy arrays for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # Create random vectors within the specified bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    
    # Evaluate initial population
    # We iterate manually to check time constraints frequently
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # Main Evolution Loop
    while True:
        # Check time at the start of the generation
        if (datetime.now() - start_time) >= time_limit:
            return best
            
        # Iterate through each individual in the population
        for i in range(pop_size):
            # Check time before creating trial vector and evaluating
            if (datetime.now() - start_time) >= time_limit:
                return best
            
            # 1. Mutation Strategy (DE/rand/1)
            # Select 3 random distinct indices (a, b, c) that are not the current index i
            # We pick 4 to guarantee 3 valid ones after filtering i
            candidates = np.random.choice(pop_size, 4, replace=False)
            candidates = candidates[candidates != i]
            # Take the first 3 valid candidates
            a_idx, b_idx, c_idx = candidates[:3]
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant vector to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover
            # Create trial vector by mixing mutant and target vector
            rand_vals = np.random.rand(dim)
            # Ensure at least one parameter comes from the mutant
            j_rand = np.random.randint(dim)
            cross_mask = (rand_vals < crossover_prob)
            cross_mask[j_rand] = True
            
            trial_vector = np.where(cross_mask, mutant, population[i])
            
            # 3. Selection
            trial_fitness = func(trial_vector)
            
            # Greedy selection: if trial is better, replace target
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness
                    
    return best
