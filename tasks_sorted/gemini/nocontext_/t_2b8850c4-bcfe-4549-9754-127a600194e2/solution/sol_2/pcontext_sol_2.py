#Here is the fully operational, self-contained Python code implementing a Differential Evolution algorithm. This algorithm is robust for finding global minima in continuous spaces and checks the time limit strictly before every function evaluation to ensure compliance with `max_time`.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Finds the input parameters that minimize the output of func within max_time 
    using a Differential Evolution algorithm.
    """
    # Start the timer immediately
    start_time = time.time()
    
    # Pre-process bounds into numpy arrays for faster vectorized operations
    bounds_array = np.array(bounds)
    lower_bounds = bounds_array[:, 0]
    upper_bounds = bounds_array[:, 1]
    diff_bounds = upper_bounds - lower_bounds

    # --- Hyperparameters ---
    # Population size: Standard heuristic is 10 to 15 times the dimension.
    # We use a dynamic size but limit it to ensure we don't spend too long initializing.
    pop_size = max(10, 15 * dim)
    
    # Differential Evolution parameters
    mutation_factor = 0.8        # [0, 2] Controls amplification of differential variation
    crossover_probability = 0.7  # [0, 1] Controls probability of parameter inheritance

    # Initialize Population containers
    population = np.empty((pop_size, dim))
    fitness_values = np.empty(pop_size)
    best_fitness = float('inf')

    # --- Initialization Phase ---
    # We initialize individuals one by one to allow early exit if max_time is very short
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_fitness

        # Generate random individual within bounds
        ind = lower_bounds + np.random.rand(dim) * diff_bounds
        population[i] = ind
        
        # Evaluate fitness
        val = func(ind)
        fitness_values[i] = val
        
        # Track global best
        if val < best_fitness:
            best_fitness = val

    # --- Evolution Phase ---
    while True:
        # Loop through the population
        for i in range(pop_size):
            # Check time limit before every calculation
            if time.time() - start_time >= max_time:
                return best_fitness

            # 1. Mutation: Select 3 distinct random indices distinct from current i
            # Using a while loop is generally faster than generating a list of exclusions for small selections
            a, b, c = i, i, i
            while a == i: 
                a = np.random.randint(0, pop_size)
            while b == i or b == a: 
                b = np.random.randint(0, pop_size)
            while c == i or c == a or c == b: 
                c = np.random.randint(0, pop_size)

            p_a, p_b, p_c = population[a], population[b], population[c]

            # Create mutant vector: v = a + F * (b - c)
            mutant = p_a + mutation_factor * (p_b - p_c)
            
            # Clip mutant to ensure it stays within specified bounds
            mutant = np.clip(mutant, lower_bounds, upper_bounds)

            # 2. Crossover: Create trial vector
            # Select which dimensions come from mutant based on crossover probability
            cross_points = np.random.rand(dim) < crossover_probability
            
            # Ensure at least one dimension is changed (standard DE requirement)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection: Greedy approach
            trial_fitness = func(trial_vector)

            # If trial is better, replace the target individual
            if trial_fitness < fitness_values[i]:
                fitness_values[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best if found
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness

    return best_fitness
