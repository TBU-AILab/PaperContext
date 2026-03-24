#Here is a fully operational, self-contained implementation of a Differential Evolution algorithm. This algorithm is chosen because it is robust against local minima and does not require gradient information (derivative-free), making it ideal for black-box function minimization within a time constraint.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using the Differential Evolution (DE) algorithm
    within a specified time limit.
    """
    # 1. Setup Time Constraints
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    # 2. Algorithm Hyperparameters
    # Population size: 10x-15x dimension is standard. 
    # We ensure a minimum of 10 agents to maintain diversity.
    pop_size = max(10, 15 * dim)
    
    # Crossover Probability (CR)
    crossover_rate = 0.7 
    # Differential Weight (F) is randomized (dithered) in the loop between 0.5 and 1.0

    # 3. Initialize Population
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Generate random population within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness tracking
    best_fitness = float('inf')
    fitness_scores = np.full(pop_size, float('inf'))

    # Evaluate initial population
    # We loop explicitly to check time constraints frequently during initialization
    for i in range(pop_size):
        if is_time_up():
            # If time runs out during initialization, return the best found so far
            # If no valid fitness found yet, evaluate the first one and return
            if best_fitness == float('inf'):
                return func(population[0])
            return best_fitness
        
        score = func(population[i])
        fitness_scores[i] = score
        
        if score < best_fitness:
            best_fitness = score

    # 4. Main Differential Evolution Loop
    # Strategy: DE/rand/1/bin
    while not is_time_up():
        for i in range(pop_size):
            if is_time_up():
                return best_fitness

            # --- Mutation ---
            # Select 3 random, distinct indices distinct from current index i
            # We select 3 indices from the range [0, pop_size)
            # If we pick 'i' or duplicates, we retry (simple rejection sampling)
            a, b, c = i, i, i
            while a == i or b == i or c == i or a == b or a == c or b == c:
                a, b, c = np.random.randint(0, pop_size, 3)

            # Dithering F between 0.5 and 1.0 improves convergence stability
            f_weight = np.random.uniform(0.5, 1.0)
            
            # Mutant Vector: v = a + F * (b - c)
            mutant = population[a] + f_weight * (population[b] - population[c])

            # --- Crossover ---
            # Binomial Crossover
            rand_vals = np.random.rand(dim)
            # Ensure at least one parameter is taken from mutant (fixed index)
            j_rand = np.random.randint(dim)
            
            # Mask: True where we take from mutant, False where we keep from target
            cross_mask = (rand_vals < crossover_rate)
            cross_mask[j_rand] = True
            
            # Create trial vector
            trial = np.where(cross_mask, mutant, population[i])

            # --- Bounds Handling ---
            # Clip values to stay within the defined search space
            trial = np.clip(trial, min_b, max_b)

            # --- Selection ---
            trial_fitness = func(trial)

            if trial_fitness < fitness_scores[i]:
                fitness_scores[i] = trial_fitness
                population[i] = trial
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    # return fitness of the best found solution
    return best_fitness
