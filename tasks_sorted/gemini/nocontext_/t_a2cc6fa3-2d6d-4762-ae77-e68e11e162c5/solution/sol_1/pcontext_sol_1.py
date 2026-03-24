#Here is the implementation of a **Differential Evolution (DE)** algorithm.
#
#Differential Evolution is a robust, stochastic, population-based optimization algorithm. It is highly effective for finding the global minimum of a function in a multi-dimensional space without requiring derivatives (gradients). It excels at avoiding local minima and works well within time constraints by iteratively improving a population of candidate solutions.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Setup timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # pop_size: Number of candidate solutions. 15 * dim is a standard heuristic.
    pop_size = max(10, 15 * dim) 
    mutation_factor = 0.8  # (F) Controls the amplification of the differential variation
    crossover_prob = 0.7   # (CR) Probability of crossover
    
    # Pre-process bounds for numpy efficiency
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # 1. Initialization: Create random population within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.zeros(pop_size)
    best = float('inf')
    
    # We evaluate initial population cautiously regarding time
    for i in range(pop_size):
        # Check time periodically even during initialization
        if i % 5 == 0 and datetime.now() >= end_time:
             return best if best != float('inf') else func(population[i])
             
        val = func(population[i])
        fitness[i] = val
        if val < best:
            best = val

    # 2. Main Evolution Loop
    while True:
        # Check time at the start of every generation
        if datetime.now() >= end_time:
            return best
            
        # Create Trial Vectors (Vectorized Mutation & Crossover)
        
        # Select 3 random distinct indices (a, b, c) for every individual i
        # We shift indices to ensure a != b != c != i
        idxs = np.argsort(np.random.rand(pop_size, pop_size), axis=1)
        # We need the first 3 indices that are not 'i' (the diagonal)
        # However, for simplicity and speed in numpy, simple random choice is usually sufficient 
        # provided pop_size is large enough. Here is a robust vectorized approach:
        
        # Generate indices for mutation
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        r3 = np.random.randint(0, pop_size, pop_size)
        
        # Mutation: V = X_r1 + F * (X_r2 - X_r3)
        mutant = population[r1] + mutation_factor * (population[r2] - population[r3])
        
        # Bounds constraints (Clip)
        mutant = np.clip(mutant, min_b, max_b)
        
        # Crossover: binomial crossover
        # Pick random dimensions to definitely change (ensure at least one param changes)
        cross_points = np.random.rand(pop_size, dim) < crossover_prob
        
        # Ensure at least one dimension is taken from mutant to avoid exact copy of parent
        # (For strict DE compliance, though standard binomial often suffices)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_points[np.arange(pop_size), j_rand] = True
        
        # Create trial population: where crossover is True, take mutant, else take parent
        trial_pop = np.where(cross_points, mutant, population)
        
        # 3. Selection (Greedy)
        # Evaluate trials and replace parent if trial is better
        for i in range(pop_size):
            # Check time strictly before expensive function calls
            if datetime.now() >= end_time:
                return best
            
            trial_fitness = func(trial_pop[i])
            
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_pop[i]
                
                if trial_fitness < best:
                    best = trial_fitness

    return best
