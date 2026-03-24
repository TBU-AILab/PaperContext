#Here is a fully operational implementation using the **Differential Evolution** algorithm.
#
#Differential Evolution is significantly more efficient than random search for continuous parameter optimization. It maintains a population of candidate solutions and evolves them by mixing existing candidates (mutation and crossover) to explore the search space and converge on a minimum.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution within a time limit.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- Hyperparameters ---
    # Population size: larger explores better but is slower per generation.
    # We use a dynamic size based on dimension, clamped to reasonable limits.
    pop_size = max(5, 4 * dim)
    if pop_size > 50: 
        pop_size = 50  # Cap population to ensure iterations in short timeframes

    mutation_factor = 0.8   # (F) Weighting for differential vector
    crossover_prob = 0.7    # (CR) Probability of crossover

    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff = max_b - min_b

    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff
    
    # store fitness of each individual
    fitness_scores = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')

    # --- Initial Evaluation ---
    # We must evaluate the initial random population to establish a baseline
    for i in range(pop_size):
        # Check time constraint strictly
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness

        score = func(population[i])
        fitness_scores[i] = score

        if score < best_fitness:
            best_fitness = score

    # --- Evolution Loop ---
    # DE Strategy: rand/1/bin
    while True:
        for i in range(pop_size):
            # Check time constraint at individual level for responsiveness
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # Enforce bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector
            # Select random dimensions to accept from mutant based on CR
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one dimension is changed so trial != target
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection: Greedy approach
            trial_fitness = func(trial)

            if trial_fitness < fitness_scores[i]:
                # Trial is better, replace the individual
                population[i] = trial
                fitness_scores[i] = trial_fitness
                
                # Update global best if needed
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
