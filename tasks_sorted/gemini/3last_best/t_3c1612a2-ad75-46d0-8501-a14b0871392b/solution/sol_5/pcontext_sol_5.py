import numpy as np
import time

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = time.time()
    
    # Differential Evolution Configuration
    # Population size is adapted to dimension but clamped to reasonable limits
    pop_size = int(max(10, min(100, 10 * dim)))
    mutation_factor = 0.8
    crossover_prob = 0.7
    
    # Prepare bounds arrays for vectorized operations
    bounds_np = np.array(bounds)
    lower_bound = bounds_np[:, 0]
    upper_bound = bounds_np[:, 1]
    bound_diff = upper_bound - lower_bound
    
    # Initialize population randomly within bounds
    # population shape: (pop_size, dim)
    population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    
    # Evaluate initial population
    # We loop explicitly to check time constraints frequently
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_fitness
        
        score = func(population[i])
        fitness[i] = score
        
        if score < best_fitness:
            best_fitness = score

    # Main optimization loop
    while True:
        for i in range(pop_size):
            # Check time constraints at each step
            if time.time() - start_time >= max_time:
                return best_fitness
            
            # Mutation: Select 3 distinct random individuals different from current i
            # DE/rand/1 strategy
            candidates = np.random.randint(0, pop_size, 3)
            # Ensure indices are distinct and not equal to current index i
            while i in candidates or len(np.unique(candidates)) < 3:
                candidates = np.random.randint(0, pop_size, 3)
            
            a, b, c = candidates
            
            # Create mutant vector
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            
            # Constrain mutant to bounds
            mutant = np.clip(mutant, lower_bound, upper_bound)
            
            # Crossover: Binomial
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one parameter comes from the mutant (to avoid duplicating parent)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial_vector = np.where(cross_points, mutant, population[i])
            
            # Selection: Evaluate trial vector
            trial_score = func(trial_vector)
            
            # Greedy selection: replace if equal or better
            if trial_score <= fitness[i]:
                fitness[i] = trial_score
                population[i] = trial_vector
                
                # Update global best
                if trial_score < best_fitness:
                    best_fitness = trial_score
                    
    return best_fitness
