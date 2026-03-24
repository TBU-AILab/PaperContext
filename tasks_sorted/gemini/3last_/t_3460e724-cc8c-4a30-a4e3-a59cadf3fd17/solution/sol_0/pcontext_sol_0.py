import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function within the specified time limit using 
    Differential Evolution (DE).
    """
    start_time = time.time()
    
    # --- Hyperparameters ---
    # Population size: Standard is 10*dim, but we cap it to ensure 
    # the algorithm iterates sufficiently given the time constraint.
    pop_size = max(10, 5 * dim) 
    if pop_size > 60: 
        pop_size = 60
        
    mutation_factor = 0.8  # F: Weighting factor for differential vector
    crossover_prob = 0.7   # CR: Probability of crossover
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    fitness = np.full(pop_size, np.inf)
    best_val = np.inf
    
    # Evaluate initial population
    # We iterate manually to check time constraints between function calls
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # --- Main Loop (Generations) ---
    while True:
        # Check total time elapsed at the start of generation
        if (time.time() - start_time) >= max_time:
            return best_val

        # 1. Mutation: V = X_a + F * (X_b - X_c)
        # Generate random indices for a, b, c
        # Note: While standard DE enforces distinct indices (a != b != c != i),
        # allowing random selection with replacement is a robust approximation 
        # for vectorized implementation in constrained scripts.
        idx_a = np.random.randint(0, pop_size, pop_size)
        idx_b = np.random.randint(0, pop_size, pop_size)
        idx_c = np.random.randint(0, pop_size, pop_size)

        mutant = population[idx_a] + mutation_factor * (population[idx_b] - population[idx_c])

        # 2. Bounds Handling (Clip values to stay inside search space)
        mutant = np.clip(mutant, min_b, max_b)

        # 3. Crossover (Binomial)
        cross_points = np.random.rand(pop_size, dim) < crossover_prob
        trial_pop = np.where(cross_points, mutant, population)

        # 4. Selection
        # Evaluate trial vectors and accept if fitness improves
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val

            f_trial = func(trial_pop[i])

            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                if f_trial < best_val:
                    best_val = f_trial

    return best_val
