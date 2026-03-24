#Here is the fully operational and self-contained Python code implementing a **Differential Evolution** algorithm. This algorithm is robust for finding global minima in continuous search spaces without requiring gradient information.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: Scale with dimension, but ensure a minimum size for diversity
    pop_size = max(10, 5 * dim)
    mutation_factor = 0.8  # F: Scaling factor for mutation
    crossover_prob = 0.9   # CR: Probability of crossover

    # Pre-process bounds for efficient numpy vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    scores = np.full(pop_size, float('inf'))
    best = float('inf')

    # Evaluate initial population
    for i in range(pop_size):
        # Check time constraint
        if (datetime.now() - start) >= limit:
            return best
            
        val = func(population[i])
        scores[i] = val
        
        # Update global best
        if val < best:
            best = val

    # Main Evolution Loop
    while True:
        # Iterate over the population
        for i in range(pop_size):
            # Check time constraint frequently
            if (datetime.now() - start) >= limit:
                return best
            
            # --- Mutation (DE/rand/1 strategy) ---
            # Select 3 distinct random indices from population, excluding current index i
            # We sample 4 to ensure we have enough after filtering out 'i'
            idxs = np.random.choice(pop_size, 4, replace=False)
            idxs = idxs[idxs != i]
            a, b, c = idxs[:3]
            
            # Create mutant vector: V = X_a + F * (X_b - X_c)
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            
            # Enforce bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover ---
            # Create trial vector by mixing target vector and mutant vector
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one parameter comes from the mutant (avoid duplicating parent)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # --- Selection ---
            # Greedy selection: if trial is better, replace parent
            trial_score = func(trial)
            
            if trial_score < scores[i]:
                scores[i] = trial_score
                population[i] = trial
                
                # Update global best if found
                if trial_score < best:
                    best = trial_score

    # return fitness of the best found solution
    return best
