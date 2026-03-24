import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing and algorithm state
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    best = float('inf')

    # Differential Evolution Configuration
    # Population size: Enough to explore, but capped to ensure generations run within max_time
    pop_size = max(10, int(10 * dim))
    if pop_size > 50: 
        pop_size = 50
    
    mutation_factor = 0.8  # F
    crossover_prob = 0.7   # CR

    # Pre-process bounds for vectorized operations
    bounds_array = np.array(bounds)
    min_b = bounds_array[:, 0]
    max_b = bounds_array[:, 1]
    diff_b = max_b - min_b

    # Initialize Population
    # Create random vectors within the specified bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness_scores = np.full(pop_size, float('inf'))

    # Evaluate Initial Population
    # We loop manually to allow checking the timer between function calls
    for i in range(pop_size):
        if (datetime.now() - start_time) >= limit:
            return best
        
        val = func(population[i])
        fitness_scores[i] = val
        
        if val < best:
            best = val

    # Main Evolutionary Loop
    while True:
        # Check time at the start of the generation
        if (datetime.now() - start_time) >= limit:
            return best

        for i in range(pop_size):
            # Strict time check before every individual calculation
            if (datetime.now() - start_time) >= limit:
                return best

            # --- Mutation ---
            # Select 3 distinct indices excluding the current individual 'i'
            # We select from the whole population range and filter logic
            candidates = np.random.choice(pop_size, 4, replace=False)
            candidates = [idx for idx in candidates if idx != i]
            
            # We need exactly 3 distinct neighbors
            # If random selection accidentally included 'i' and we removed it, 
            # we might have fewer than 3. In that rare case, just pick safe defaults.
            if len(candidates) < 3:
                # Fallback mechanism to ensure we have 3 distinct indices
                pool = list(range(pop_size))
                pool.remove(i)
                idxs = np.random.choice(pool, 3, replace=False)
            else:
                idxs = candidates[:3]

            a = population[idxs[0]]
            b = population[idxs[1]]
            c = population[idxs[2]]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant vector to remain within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover ---
            # Binomial Crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is taken from the mutant (promotes diversity)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])

            # --- Selection ---
            trial_fitness = func(trial_vector)

            # Greedy selection: if trial is better, it replaces the parent
            if trial_fitness < fitness_scores[i]:
                fitness_scores[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness

    return best
