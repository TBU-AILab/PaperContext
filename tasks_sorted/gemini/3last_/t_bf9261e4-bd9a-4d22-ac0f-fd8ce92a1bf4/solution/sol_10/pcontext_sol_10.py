import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Parameters
    # Adjust population size based on dimensions, bounded to maintain speed
    pop_size = max(10, min(100, 10 * dim))
    mutation_factor = 0.8
    crossover_rate = 0.7
    
    # Pre-process bounds for efficiency
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population randomly within bounds
    # population shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    
    # Initial evaluation of the population
    for i in range(pop_size):
        # Check time before potentially expensive function call
        if datetime.now() - start_time >= limit:
            return best_val
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            
    # Main Differential Evolution Loop
    while True:
        for i in range(pop_size):
            # Check time constraint strictly within the inner loop
            if datetime.now() - start_time >= limit:
                return best_val
            
            # Mutation: Select 3 distinct random individuals (a, b, c) != i
            # We generate indices excluding i
            idxs = [idx for idx in range(pop_size) if idx != i]
            selected = np.random.choice(idxs, 3, replace=False)
            
            a = population[selected[0]]
            b = population[selected[1]]
            c = population[selected[2]]
            
            mutant = a + mutation_factor * (b - c)
            
            # Crossover: Create trial vector
            # Randomly accept mutation values based on crossover rate
            cross_points = np.random.rand(dim) < crossover_rate
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Boundary Handling: Clip values to stay within bounds
            trial = np.clip(trial, min_b, max_b)
            
            # Selection: Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                if f_trial < best_val:
                    best_val = f_trial

    return best_val
