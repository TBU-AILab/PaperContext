#Here is a fully operational and self-contained Python algorithm using Differential Evolution to minimize the output value within a limited time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the output of 'func' using Differential Evolution within 'max_time'.
    """
    # Initialize timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size logic: scale with dimension but keep within reasonable limits for speed
    pop_size = min(max(20, 10 * dim), 100) 
    mutation_factor = 0.8
    crossover_prob = 0.7

    # Parse bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population (pop_size x dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best = float('inf')
    
    # Evaluate Initial Population
    for i in range(pop_size):
        # Time check before expensive function call
        if datetime.now() >= end_time:
            return best
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # Main Optimization Loop
    while True:
        # Check time at generation start
        if datetime.now() >= end_time:
            break
            
        for i in range(pop_size):
            # Strict time check per individual to ensure we don't exceed max_time
            if datetime.now() >= end_time:
                return best
            
            # 1. Mutation: Select 3 distinct random individuals excluding current 'i'
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            x_a = population[a_idx]
            x_b = population[b_idx]
            x_c = population[c_idx]
            
            # DE/rand/1 mutation vector
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # 2. Crossover: Binomial
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one parameter is changed
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Bound Constraints: Clip values to stay within bounds
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection: Evaluate and greedy keep
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
