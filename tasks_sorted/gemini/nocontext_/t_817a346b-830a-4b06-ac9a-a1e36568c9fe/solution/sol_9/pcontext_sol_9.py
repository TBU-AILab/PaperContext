#Here is the algorithm implementation. It utilizes **Differential Evolution**, a robust global optimization algorithm suited for continuous parameter spaces. It is designed to converge towards a minimum efficiently and handles the time constraint by checking the elapsed time before every new function evaluation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    
    # --- Differential Evolution Parameters ---
    # Population size: Dynamic based on dimension, but capped for speed
    pop_size = max(10, dim * 5)
    if pop_size > 50: 
        pop_size = 50
    
    mutation_factor = 0.8   # F: Controls the amplification of differential variation
    crossover_prob = 0.7    # CR: Controls the probability of crossover
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Check time constraint
        if (datetime.now() - start) >= timedelta(seconds=max_time):
            return best
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Optimization Loop ---
    while True:
        for i in range(pop_size):
            # Check time constraint at every step
            if (datetime.now() - start) >= timedelta(seconds=max_time):
                return best
            
            # 1. Mutation (DE/rand/1 strategy)
            # Select 3 distinct random indices distinct from current i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            
            # Create mutant vector: a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Enforce bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover (Binomial)
            # Create a mask for crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one dimension is changed (to avoid duplicating parent)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Create trial vector
            trial = np.where(cross_points, mutant, pop[i])
            
            # 3. Selection
            f_trial = func(trial)
            
            # Greedy selection: if trial is better or equal, replace parent
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best found so far
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
