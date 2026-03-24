#Here is the fully operational algorithm implementing **Differential Evolution (DE)**.
#
#This algorithm is superior to Random Search as it uses the differences between existing solutions to guide the search toward the minimum, making it a robust method for continuous global optimization problems within a limited timeframe. It checks the elapsed time before every function evaluation to strictly respect `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timer
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Parameters
    # Population size: Standard is 10*dim. We set a minimum of 10 to ensure genetic diversity.
    pop_size = max(10, 10 * dim)
    mutation_factor = 0.8  # F: Controls the amplification of the differential variation
    crossover_prob = 0.7   # CR: Probability of crossover
    
    # Pre-process bounds into numpy arrays for vectorization
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = np.fabs(max_b - min_b)
    
    # Data structures
    population = np.zeros((pop_size, dim))
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')

    # --- Phase 1: Initialization ---
    # We populate and evaluate one by one to ensure we have a valid 'best'
    # even if max_time is extremely short.
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best

        # Generate random individual within bounds
        population[i] = min_b + np.random.rand(dim) * diff_b
        
        # Evaluate
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Phase 2: Evolution Loop ---
    while True:
        for i in range(pop_size):
            # Strict time check before every new evaluation
            if (datetime.now() - start_time) >= time_limit:
                return best

            # 1. Mutation: Select 3 distinct random individuals (a, b, c) distinct from i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # 2. Crossover: Create trial vector u
            # Mix mutant and current target vector based on crossover probability
            rand_cr = np.random.rand(dim)
            cross_mask = rand_cr < crossover_prob
            
            # Ensure at least one parameter comes from the mutant (standard DE requirement)
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, population[i])

            # 3. Constrain trial vector to bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection
            trial_val = func(trial)

            # Greedy selection: if trial is better or equal, it replaces the target
            if trial_val <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_val
                
                # Update global best found so far
                if trial_val < best:
                    best = trial_val

    return best
