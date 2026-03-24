import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a time limit.
    DE is a robust global optimization algorithm that works well for non-differentiable
    and nonlinear functions without requiring gradient information.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Balance between diversity (high) and convergence speed (low).
    # We use a dynamic size based on dimension, but capped to ensure we get 
    # enough generations within the time limit.
    pop_size = max(5, min(40, 8 * dim))
    
    # Differential Evolution Parameters
    crossover_prob = 0.9  # Probability of recombining with the mutant
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    # Best solution found so far (in case we time out immediately)
    # Using the first individual as a placeholder
    
    for i in range(pop_size):
        # Time check inside initialization loop
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness

        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Optimization Loop ---
    while True:
        # Loop through the population
        for i in range(pop_size):
            # Check time constraints at every individual evaluation step
            # to ensure strict adherence to max_time
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness

            # 1. Mutation: DE/rand/1 strategy
            # Select 3 random distinct indices excluding current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            vec_a = population[a_idx]
            vec_b = population[b_idx]
            vec_c = population[c_idx]

            # Dithering: randomize mutation factor F slightly between 0.5 and 1.0
            # to prevent stagnation
            f = 0.5 + np.random.random() * 0.5
            
            mutant = vec_a + f * (vec_b - vec_c)
            
            # Constrain mutant to bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Binomial
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one parameter is changed
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection
            trial_fitness = func(trial_vector)

            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
