#Here is a fully operational, self-contained Python implementation of the **Differential Evolution** algorithm. 
#
#Differential Evolution is a robust, stochastic, derivative-free optimization method that is generally more efficient than random search for finding global minima in continuous search spaces.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    # Define the time limit as a timedelta object
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: Standard heuristic is 10 * dim. 
    # We enforce a minimum of 4 to allow for the mutation strategy (requires 3 other agents).
    pop_size = max(4, int(10 * dim)) 
    mutation_factor = 0.8   # (F) Controls the amplification of the differential variation
    crossover_prob = 0.7    # (CR) Controls the probability of parameter mixing

    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize Population: Uniform distribution within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness array
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')

    # Evaluate Initial Population
    # We iterate manually to check the time limit after every evaluation
    for i in range(pop_size):
        if (datetime.now() - start_time) >= limit:
            # If time runs out during initialization, return the best found so far
            # or evaluate the first one to return a valid float.
            if best == float('inf'):
                return func(population[0])
            return best

        val = func(population[i])
        fitness[i] = val
        if val < best:
            best = val

    # Main Optimization Loop
    while (datetime.now() - start_time) < limit:
        for i in range(pop_size):
            # Strict time check inside the inner loop to respect max_time
            if (datetime.now() - start_time) >= limit:
                return best

            # --- Mutation Step (DE/rand/1) ---
            # Select 3 distinct random individuals (a, b, c) from population, none equal to i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant vector to stay within search space bounds
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover Step (Binomial) ---
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is taken from the mutant (promotes diversity)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Create trial vector
            trial_vector = np.where(cross_points, mutant, population[i])

            # --- Selection Step ---
            # Evaluate objective function
            trial_fitness = func(trial_vector)

            # Greedy selection: if trial is better or equal, replace the parent
            if trial_fitness <= fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
