#Here is the algorithm implementation using **Differential Evolution (DE)**. DE is a robust, population-based metaheuristic that is highly effective for global optimization of continuous functions without requiring gradient information. It is well-suited for finding the minimum value within a specific time limit as it progressively improves a population of candidate solutions.
#
#The code is self-contained and handles the time constraint by checking the elapsed time before every function evaluation.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    # Start the timer
    start_time = time.time()
    
    # --- Differential Evolution Hyperparameters ---
    # Population size: Sufficient to explore, but limited to ensure generations 
    # can occur within max_time. Capped at 50 for speed.
    pop_size = max(10, 10 * dim)
    pop_size = min(pop_size, 50) 
    
    mutation_factor = 0.8   # (F) Weight for differential mutation [0, 2]
    crossover_rate = 0.7    # (CR) Probability of crossover [0, 1]

    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness array
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')

    # Evaluate initial population
    # We iterate manually to check time constraints strictly between evaluations
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val

    # --- Main Optimization Loop ---
    while True:
        # Iterate through the population (Target Vector: x)
        for i in range(pop_size):
            # 1. Time Check
            if (time.time() - start_time) >= max_time:
                return best

            # 2. Mutation
            # Select 3 distinct random individuals (a, b, c) different from i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # 3. Crossover
            # Create trial vector (u) by mixing target (x) and mutant (v)
            rand_vals = np.random.rand(dim)
            # Ensure at least one parameter comes from mutant to introduce variation
            j_rand = np.random.randint(dim)
            mask = (rand_vals < crossover_rate)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, population[i])

            # 4. Boundary Constraint Handling
            # Clip values to stay within the defined search space
            trial = np.clip(trial, min_b, max_b)

            # 5. Selection
            # Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better, replace target
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best found so far
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
