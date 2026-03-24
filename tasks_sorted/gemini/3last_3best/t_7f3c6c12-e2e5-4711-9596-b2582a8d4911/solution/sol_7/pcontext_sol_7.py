import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size logic: balance between diversity and speed
    pop_size = max(10, 5 * dim)
    if pop_size > 50:
        pop_size = 50
        
    F = 0.8  # Mutation factor
    CR = 0.9 # Crossover probability
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population: shape (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track fitness and best solution
    pop_fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Time check
        if datetime.now() - start_time >= limit:
            return best_val
            
        val = func(population[i])
        pop_fitness[i] = val
        
        if val < best_val:
            best_val = val
            
    # Main Differential Evolution Loop
    while True:
        for i in range(pop_size):
            # Strict time check inside the loop
            if datetime.now() - start_time >= limit:
                return best_val
            
            # Mutation: Select 3 distinct random indices excluding current i
            # Using list comprehension for index filtering
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # DE/rand/1 mutation strategy
            mutant = a + F * (b - c)
            
            # Crossover: Binomial
            cross_points = np.random.rand(dim) < CR
            # Ensure at least one parameter comes from the mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial = np.where(cross_points, mutant, population[i])
            
            # Boundary Constraint Handling: Clip values to bounds
            trial = np.clip(trial, min_b, max_b)
            
            # Selection: Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection
            if f_trial < pop_fitness[i]:
                pop_fitness[i] = f_trial
                population[i] = trial
                if f_trial < best_val:
                    best_val = f_trial
                    
    return best_val
