import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Setup timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Differential Evolution Parameters
    # We choose a population size relative to dimension but capped to ensure 
    # the algorithm performs generations within limited time.
    pop_size = min(max(20, 10 * dim), 50) 
    mutation_factor = 0.6  # F
    crossover_prob = 0.7   # CR
    
    # Parse bounds into numpy arrays for vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population uniformly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitnesses = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')

    # Evaluate initial population
    for i in range(pop_size):
        # Time check before expensive function evaluation
        if datetime.now() >= end_time:
            return best_fitness
        
        val = func(population[i])
        fitnesses[i] = val
        
        if val < best_fitness:
            best_fitness = val
            
    # Main Differential Evolution Loop
    while True:
        for i in range(pop_size):
            # Strict time check
            if datetime.now() >= end_time:
                return best_fitness
            
            # Mutation: DE/rand/1 strategy
            # Select 3 distinct random individuals other than the current one (i)
            idxs = [idx for idx in range(pop_size) if idx != i]
            # Use np.random.choice to pick 3 indices
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector
            mutant = a + mutation_factor * (b - c)
            
            # Crossover: Binomial
            rand_vals = np.random.rand(dim)
            cross_points = rand_vals < crossover_prob
            
            # Ensure at least one parameter comes from the mutant (to avoid duplicating parent)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # Bound Handling: Clip values to stay within specified bounds
            trial = np.clip(trial, min_b, max_b)
            
            # Selection: Greedy selection
            f_trial = func(trial)
            
            if f_trial < fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    
    return best_fitness
