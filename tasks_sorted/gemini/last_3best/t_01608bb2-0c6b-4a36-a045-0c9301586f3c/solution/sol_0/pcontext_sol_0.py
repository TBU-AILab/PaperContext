#Here is a fully operational, self-contained Python implementation using the **Differential Evolution** algorithm. 
#
#This algorithm is chosen because it is highly effective for global optimization of continuous, black-box functions and converges significantly faster than random search while being robust against local minima.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Differential Evolution Hyperparameters ---
    # Population size: larger allows more diversity, but slows down generations.
    # We cap it to ensure the algorithm runs enough iterations within the time limit.
    pop_size = max(10, 5 * dim) 
    if pop_size > 50: 
        pop_size = 50
        
    mutation_factor = 0.8     # Scaling factor (F)
    crossover_prob = 0.7      # Crossover probability (CR)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Generate initial random population within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    # We initialize 'best' immediately to ensure we have a result even if time is very short
    fitnesses = np.zeros(pop_size)
    best = float('inf')
    
    for i in range(pop_size):
        # Check time even during initialization
        if (datetime.now() - start) >= limit:
            return best
            
        val = func(population[i])
        fitnesses[i] = val
        if val < best:
            best = val

    # --- Optimization Loop ---
    # We loop until the time limit is reached
    while (datetime.now() - start) < limit:
        
        for i in range(pop_size):
            # Check time constraint at individual level to exit gracefully and quickly
            if (datetime.now() - start) >= limit:
                return best

            # 1. Mutation: Select 3 distinct random agents (a, b, c) != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            a_idx, b_idx, c_idx = random.sample(candidates, 3)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant vector to stay within search space bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector by mixing target(i) and mutant
            # Generate random mask based on crossover probability
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one index is taken from mutant to introduce change
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection: Evaluate trial vector
            f_trial = func(trial)

            # If trial is better, replace the individual in the population
            if f_trial < fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial
                
                # Update global best found so far
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
