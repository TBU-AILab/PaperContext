#Here is a fully operational, self-contained implementation of the Differential Evolution algorithm within the requested template. Differential Evolution is a robust method for finding the global minimum of a function within a multi-dimensional space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Setup timing constraints
    start = datetime.now()
    max_duration = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: Standard is 10*dim, but we clamp it to ensure 
    # the algorithm performs enough generations within limited time.
    pop_size = max(10, 10 * dim) 
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_rate = 0.9   # Crossover probability (CR)

    # Initialize bounds arrays for vectorization
    # bounds is a list of [min, max], convert to numpy arrays
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # Initialize Population
    # Create random vectors within the bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_idx = -1

    # Evaluate Initial Population
    for i in range(pop_size):
        # Time check
        if datetime.now() - start >= max_duration:
            return best_fitness

        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # Main Optimization Loop
    while True:
        for i in range(pop_size):
            # Strict time check inside the inner loop to respect max_time
            if datetime.now() - start >= max_duration:
                return best_fitness

            # 1. Mutation: Select 3 distinct random vectors a, b, c not equal to i
            idxs = [idx for idx in range(pop_size) if idx != i]
            # We need 3 random indices
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # 2. Crossover: Create trial vector u from x[i] and mutant v
            # Binomial crossover
            cross_points = np.random.rand(dim) < crossover_rate
            # Ensure at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Boundary Constraint: Clip values to stay within bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection: Evaluate trial vector
            trial_fitness = func(trial)

            # Greedy selection: if trial is better, replace existing
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best found so far
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    
    # return fitness of the best found solution
    return best_fitness
