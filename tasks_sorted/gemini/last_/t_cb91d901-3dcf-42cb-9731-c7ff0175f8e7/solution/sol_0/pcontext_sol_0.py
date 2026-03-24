#Here is a fully operational and self-contained Python algorithm implementing **Differential Evolution (DE)**.
#
#DE is a stochastic, population-based optimization algorithm that is highly effective for minimizing non-linear, non-differentiable continuous space functions. This implementation includes strict time management to ensure the result is returned within the `max_time` constraint.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a black-box function using Differential Evolution within a time limit.
    """
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # Differential Evolution Parameters
    # Population size: Scales with dimension but capped to ensure responsiveness
    # Range [10, 50] is generally robust for limited-time scenarios
    pop_size = max(10, min(50, int(dim * 5)))
    
    mutation_factor = 0.6  # Scaling factor (F)
    crossover_rate = 0.8   # Crossover probability (CR)

    # Pre-process bounds into numpy arrays for vectorized operations
    bounds_arr = np.array(bounds)
    min_bound = bounds_arr[:, 0]
    max_bound = bounds_arr[:, 1]
    diff_bound = max_bound - min_bound

    # Initialize Population
    # Structure: pop_size x dim
    population = min_bound + np.random.rand(pop_size, dim) * diff_bound
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')

    # Initial Evaluation
    # We evaluate sequentially to allow early exit if time is very tight
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_fitness
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # Main Optimization Loop
    while True:
        # Check overall time before starting a new generation
        if datetime.now() - start_time >= time_limit:
            return best_fitness

        for i in range(pop_size):
            # Strict time check before every function evaluation
            if datetime.now() - start_time >= time_limit:
                return best_fitness

            # 1. Mutation (DE/rand/1 strategy)
            # Select 3 distinct random indices different from current index i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            # Generate mutant vector
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # Clip mutant vector to stay within defined bounds
            mutant = np.clip(mutant, min_bound, max_bound)
            
            # 2. Crossover (Binomial)
            rand_vals = np.random.rand(dim)
            cross_mask = rand_vals < crossover_rate
            
            # Ensure at least one parameter is inherited from the mutant
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial_vector = np.where(cross_mask, mutant, population[i])
            
            # 3. Selection
            trial_fitness = func(trial_vector)
            
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best found so far
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
