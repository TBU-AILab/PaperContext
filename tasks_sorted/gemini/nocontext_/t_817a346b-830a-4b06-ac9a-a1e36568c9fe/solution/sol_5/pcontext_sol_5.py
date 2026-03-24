#Here is a fully operational, self-contained Python algorithm based on **Differential Evolution (DE)**.
#
#Differential Evolution is a robust evolutionary algorithm suitable for global optimization problems. It handles non-differentiable, non-linear, and multimodal cost functions effectively. This implementation manages the population of candidate solutions and evolves them until the time limit is reached.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution algorithm within a time limit.
    """
    # 1. Initialization and Setup
    start_time = datetime.now()
    # We use a small buffer (0.05s) to ensure we return before a strict external timeout
    end_time = start_time + timedelta(seconds=max_time) - timedelta(milliseconds=50)

    # algorithm parameters
    # Population size: usually 5 to 10 times the dimension is sufficient
    pop_size = max(10, 5 * dim)
    mutation_factor = 0.8       # F in DE literature (usually 0.5 to 1.0)
    crossover_prob = 0.7        # CR in DE literature (usually 0.6 to 0.9)

    # Convert bounds to numpy arrays for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = np.fabs(max_b - min_b)

    # Initialize population: pop_size x dim matrix
    # Random values within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.zeros(pop_size)
    best_fitness = float('inf')
    best_idx = -1

    for i in range(pop_size):
        # Time check before expensive function evaluation
        if datetime.now() >= end_time:
            return best_fitness if best_fitness != float('inf') else func(population[0])

        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # 2. Main Evolution Loop
    while True:
        # Check overall time
        if datetime.now() >= end_time:
            return best_fitness

        # Iterate through the population
        for i in range(pop_size):
            # Check time frequently (e.g., inside the generation loop) 
            # to handle cases where func() is slow.
            if datetime.now() >= end_time:
                return best_fitness

            # --- Mutation ---
            # Select 3 distinct random indices (a, b, c) different from current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # Ensure mutant stays within bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover ---
            # Create trial vector by mixing target (population[i]) and mutant
            # based on crossover probability
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is changed (taken from mutant)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            trial_vector = np.where(cross_points, mutant, population[i])

            # --- Selection ---
            f_trial = func(trial_vector)

            # Greedy selection: if trial is better, replace the individual
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_vector
                
                # Update global best immediately if applicable
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
