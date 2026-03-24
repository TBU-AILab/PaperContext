#Here is the operational Python code implementing a **Differential Evolution** algorithm. This algorithm is robust for finding global minima in continuous search spaces and includes granular time checks to respect the `max_time` constraint.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the objective function 'func' using the Differential Evolution algorithm.
    It operates within the specified 'max_time' by checking the clock after every function evaluation.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # 1. Parameter Setup
    # Extract lower and upper bounds as numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # DE Hyperparameters
    # Population size is set to balance exploration with the need for speed.
    # We ensure a minimum of 20 to allow for distinct mutation candidates.
    pop_size = int(max(20, min(50, 5 * dim)))
    crossover_prob = 0.7  # Probability of accepting mutant parameter
    
    # 2. Initialization
    population = []
    fitnesses = []
    best_fitness = float('inf')

    # Helper to check strict time limits
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    # Generate initial population
    for _ in range(pop_size):
        if is_time_up():
            return best_fitness
        
        # Random vector within bounds
        ind = min_b + np.random.rand(dim) * diff_b
        
        try:
            val = func(ind)
        except Exception:
            val = float('inf') # Fallback for failed evaluations

        population.append(ind)
        fitnesses.append(val)

        if val < best_fitness:
            best_fitness = val

    # Convert to numpy arrays for efficient indexing in the main loop
    population = np.array(population)
    fitnesses = np.array(fitnesses)
    pop_indices = list(range(pop_size))

    # 3. Main Optimization Loop (Differential Evolution)
    while not is_time_up():
        for i in range(pop_size):
            if is_time_up():
                return best_fitness

            # --- Mutation (DE/rand/1) ---
            # Pick 3 random distinct individuals other than the current one (i)
            candidates = pop_indices[:i] + pop_indices[i+1:]
            a, b, c = random.sample(candidates, 3)
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]

            # Mutation factor F (dithered between 0.5 and 1.0 for robustness)
            F = 0.5 + 0.5 * np.random.rand()
            mutant = x_a + F * (x_b - x_c)

            # --- Crossover (Binomial) ---
            # Create mask: True means take from mutant, False from target
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one dimension is changed to avoid stagnation
            cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # --- Bound Handling ---
            trial = np.clip(trial, min_b, max_b)

            # --- Selection ---
            try:
                f_trial = func(trial)
            except Exception:
                f_trial = float('inf')

            # Greedy selection: if trial is better or equal, it replaces the target
            if f_trial <= fitnesses[i]:
                population[i] = trial
                fitnesses[i] = f_trial
                
                # Update global best immediately
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
