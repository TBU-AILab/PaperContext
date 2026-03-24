#Here is a fully operational, self-contained Python algorithm using **Differential Evolution (DE)**.
#
#DE is a stochastic, population-based optimization algorithm that is highly effective for minimizing non-linear and non-differentiable continuous space functions. It is significantly more efficient than random search, as it uses differences between existing solutions to guide the search toward better areas within the bounds.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Initialize best found solution fitness
    best = float('inf')

    # --- Configuration ---
    # Population size: adaptive to dimension but capped to ensure speed
    pop_size = max(10, int(dim * 10))
    if pop_size > 40:
        pop_size = 40
        
    mutation_factor = 0.8  # F: Weighting factor [0, 2]
    crossover_prob = 0.7   # CR: Crossover probability [0, 1]

    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Storage for population
    population = np.zeros((pop_size, dim))
    fitnesses = np.full(pop_size, float('inf'))

    # --- 1. Initialization Phase ---
    # We loop explicitly to safely handle very short time limits
    for i in range(pop_size):
        if (datetime.now() - start) >= limit:
            return best
            
        # Create random individual within bounds
        ind = min_b + np.random.rand(dim) * diff_b
        
        # Evaluate
        val = func(ind)
        
        population[i] = ind
        fitnesses[i] = val
        
        if val < best:
            best = val

    # --- 2. Evolution Phase (DE/rand/1/bin) ---
    while True:
        # Iterate through population
        for i in range(pop_size):
            # Check time constraints strictly before expensive operations
            if (datetime.now() - start) >= limit:
                return best

            # -- Mutation --
            # Select 3 distinct random indices excluding current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            # Safety check for very small populations
            if len(candidates) < 3:
                continue
                
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # Enforce bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # -- Crossover --
            # Binomial crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])

            # -- Selection --
            # Evaluate trial vector
            f_trial = func(trial_vector)

            # Greedy selection: if trial is better, replace existing
            if f_trial < fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial_vector
                
                # Update global best
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
