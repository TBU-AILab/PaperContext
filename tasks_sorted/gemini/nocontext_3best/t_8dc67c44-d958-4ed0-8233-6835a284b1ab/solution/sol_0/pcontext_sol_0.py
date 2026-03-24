#Here is a fully operational, self-contained Python implementation using **Differential Evolution**.
#
#Differential Evolution is a robust evolutionary algorithm suitable for black-box optimization. It is generally more efficient than random search, as it uses the differences between randomly selected solutions to direct the search toward better areas of the parameter space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm.
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    # --- DE Hyperparameters ---
    # Population size: usually 10x dimension is a good rule of thumb.
    # We cap it at 50 to ensure we get some iterations done if max_time is short.
    pop_size = max(5, min(50, 10 * dim))
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_rate = 0.7   # Crossover probability (CR)

    # --- Setup Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- Initialize Population ---
    # Create random vectors within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness_scores = []
    best = float('inf')
    best_vector = None

    for i in range(pop_size):
        # Time check during initialization
        if datetime.now() >= end_time:
            return best if best != float('inf') else func(population[0])

        score = func(population[i])
        fitness_scores.append(score)

        if score < best:
            best = score
            best_vector = population[i]

    fitness_scores = np.array(fitness_scores)

    # --- Optimization Loop ---
    while datetime.now() < end_time:
        for i in range(pop_size):
            # Strict time check before every expensive function call
            if datetime.now() >= end_time:
                return best

            # 1. Mutation: Select 3 distinct random agents (a, b, c) != i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]

            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Constrain mutant to bounds (clip)
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector
            # Binomial crossover: pick parameter from mutant if rand < CR
            cross_points = np.random.rand(dim) < crossover_rate
            
            # Ensure at least one parameter is taken from mutant to ensure evolution
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection: Greedy approach
            trial_fitness = func(trial_vector)

            if trial_fitness <= fitness_scores[i]:
                population[i] = trial_vector
                fitness_scores[i] = trial_fitness
                
                # Update global best
                if trial_fitness < best:
                    best = trial_fitness
                    best_vector = trial_vector

    # return fitness of the best found solution
    return best
