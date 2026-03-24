#Here is a fully operational and self-contained Python algorithm using **Differential Evolution**, a robust global optimization method well-suited for finding minima in continuous search spaces without requiring derivatives.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size is scaled by dimension but capped to ensure iterations within limited time
    pop_size = int(min(max(20, 5 * dim), 100))
    mutation_factor = 0.8  # F in DE literature
    crossover_prob = 0.7   # CR in DE literature
    
    # parse bounds into numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population: shape (pop_size, dim)
    # Randomly distributed within the specified bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness_scores = np.full(pop_size, float('inf'))
    
    best = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        # Time check before expensive function call
        if (datetime.now() - start_time) >= time_limit:
            return best
            
        score = func(population[i])
        fitness_scores[i] = score
        
        if score < best:
            best = score

    # Main DE Loop
    indices = np.arange(pop_size)
    
    while (datetime.now() - start_time) < time_limit:
        for i in range(pop_size):
            # Strict time check inside generation loop
            if (datetime.now() - start_time) >= time_limit:
                return best
            
            # Select 3 distinct random individuals other than current i
            candidates = np.random.choice(np.delete(indices, i), 3, replace=False)
            a, b, c = population[candidates]
            
            # Mutation step (Strategy: DE/rand/1)
            mutant = a + mutation_factor * (b - c)
            
            # Crossover step (Binomial)
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one parameter is taken from mutant to avoid duplication
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])
            
            # Boundary handling: clip values to stay within bounds
            trial_vector = np.clip(trial_vector, min_b, max_b)
            
            # Selection step
            trial_fitness = func(trial_vector)
            
            # Greedy selection: if trial is better, replace current individual
            if trial_fitness < fitness_scores[i]:
                fitness_scores[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
