#Here is the Python code implementing a Differential Evolution algorithm to find the minimum output value within the specified time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # Population size: usually set between 10 and 20 times the dimension.
    # We use a minimum of 20 to ensure diversity even in low dimensions.
    pop_size = max(20, 15 * dim)
    
    # Differential weight (F) and Crossover probability (CR)
    # F typically lies in [0.5, 1.0], CR in [0.5, 1.0]
    F = 0.6
    CR = 0.9

    # Pre-process bounds for efficient vectorized operations
    bounds_arr = np.array(bounds)
    min_bound = bounds_arr[:, 0]
    max_bound = bounds_arr[:, 1]
    diff_bound = max_bound - min_bound

    # Initialize Population
    # Create random vectors uniformly distributed within bounds
    # Shape: (pop_size, dim)
    population = min_bound + np.random.rand(pop_size, dim) * diff_bound
    
    # Array to store fitness of each individual
    fitness_scores = np.full(pop_size, float('inf'))
    
    # Track the global best solution found so far
    best = float('inf')
    
    # ---------------------------------------------------------
    # Initialization Phase
    # ---------------------------------------------------------
    # Evaluate the initial population. We check the time constraint 
    # strictly inside this loop to handle cases with very short max_time.
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best
        
        # Call the objective function
        val = func(population[i])
        fitness_scores[i] = val
        
        # Update global best
        if val < best:
            best = val

    # ---------------------------------------------------------
    # Evolution Phase (Main Loop)
    # ---------------------------------------------------------
    # We iterate through the population and apply Mutation, Crossover, and Selection.
    # This loop runs continuously until the time limit is reached.
    
    idx = 0
    indices = np.arange(pop_size)

    while True:
        # Check time constraint
        if (datetime.now() - start_time) >= time_limit:
            return best

        # --- Mutation ---
        # Select 3 distinct random individuals (a, b, c) from population, 
        # ensuring they are distinct from the current target vector (idx).
        
        # Generate a candidate pool excluding the current index
        candidates_mask = (indices != idx)
        candidates = indices[candidates_mask]
        
        # Select 3 parents randomly
        parents = np.random.choice(candidates, 3, replace=False)
        a_idx, b_idx, c_idx = parents
        
        vec_a = population[a_idx]
        vec_b = population[b_idx]
        vec_c = population[c_idx]
        
        # Create mutant vector: v = a + F * (b - c)
        mutant = vec_a + F * (vec_b - vec_c)
        
        # --- Crossover ---
        # Perform binomial crossover
        rand_vals = np.random.rand(dim)
        cross_mask = rand_vals < CR
        
        # Ensure at least one dimension is taken from the mutant (to guarantee change)
        j_rand = np.random.randint(0, dim)
        cross_mask[j_rand] = True
        
        # Construct trial vector
        trial = np.where(cross_mask, mutant, population[idx])
        
        # --- Boundary Handling ---
        # Clip values to ensure the trial vector stays within input bounds
        trial = np.clip(trial, min_bound, max_bound)
        
        # --- Selection ---
        # Evaluate trial vector
        trial_fitness = func(trial)
        
        # Greedy selection: if trial is better or equal, it replaces the target
        if trial_fitness <= fitness_scores[idx]:
            population[idx] = trial
            fitness_scores[idx] = trial_fitness
            
            # Update global best if we found a new minimum
            if trial_fitness < best:
                best = trial_fitness
        
        # Move to the next individual in the population (Cyclic)
        idx = (idx + 1) % pop_size

    # return fitness of the best found solution
    return best
