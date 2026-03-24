import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a specified time limit.
    DE is a robust global optimization algorithm suitable for real-valued, multidimensional problems.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)

    # Convert bounds to numpy arrays for efficient vector operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # DE Hyperparameters
    # Population size: Balance between diversity and computational cost per generation.
    # We use a dynamic size capped at 50 to ensure many generations can run within max_time.
    pop_size = min(50, max(10, 10 * dim))
    mutation_factor = 0.8  # F: Weighting factor for mutation
    crossover_prob = 0.7   # CR: Crossover probability

    # Initialize population: Randomly distributed within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness array
    fitnesses = np.full(pop_size, float('inf'))
    best_fitness = float('inf')

    # Evaluate initial population
    # We loop explicitly to check the timer between potentially expensive function calls
    for i in range(pop_size):
        if (datetime.now() - start_time) >= limit:
            return best_fitness
        
        val = func(population[i])
        fitnesses[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # Main Evolution Loop
    while True:
        for i in range(pop_size):
            # Strict time check before every new evaluation
            if (datetime.now() - start_time) >= limit:
                return best_fitness

            # 1. Mutation
            # Select 3 distinct random individuals (a, b, c) from population, distinct from current (i)
            candidates = [idx for idx in range(pop_size) if idx != i]
            idxs = np.random.choice(candidates, 3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # 2. Crossover
            # Generate trial vector by mixing mutant and target vector
            # Ensure at least one dimension comes from the mutant to avoid duplicating the parent
            cross_points = np.random.rand(dim) < crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Boundary Constraint
            # Clip the trial vector to ensure it stays within defined bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection
            # Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better, replace parent
            if f_trial <= fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial
                
                # Update global best
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
