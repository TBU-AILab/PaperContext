#Here is a fully operational, self-contained Python implementation using a **Differential Evolution** algorithm. 
#
#This algorithm is superior to random search because it uses the differences between existing solutions to guide the search direction, allowing it to converge on the minimum much faster while maintaining global exploration capabilities. It strictly respects the time limit by checking the clock before every expensive function evaluation.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Finds the minimum output of func using Differential Evolution within max_time.
    """
    start_time = time.time()
    
    # --- Algorithm Hyperparameters ---
    # Population size: Balance between diversity (high) and speed (low).
    # We constrain it between 10 and 50 to ensure the loop runs even with short time limits.
    pop_size = int(np.clip(dim * 10, 10, 50))
    mutation_factor = 0.8  # Scaling factor for mutation (F)
    crossover_prob = 0.7   # Probability of crossover (CR)

    # --- Initialization ---
    # Convert bounds to numpy arrays for vectorization
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Store fitness values
    fitness = np.full(pop_size, float('inf'))
    
    # Track the global best
    best_fitness = float('inf')
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        # Safety check: if initialization takes too long, return best found so far
        if (time.time() - start_time) >= max_time:
            return best_fitness if best_fitness != float('inf') else func(population[0])
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Differential Evolution Loop ---
    while True:
        # Check time at the start of the generation
        if (time.time() - start_time) >= max_time:
            return best_fitness

        for i in range(pop_size):
            # Check time before every individual evaluation to ensure strict adherence
            if (time.time() - start_time) >= max_time:
                return best_fitness

            # 1. Mutation: Target vector is population[i]
            # Select 3 distinct random individuals (a, b, c) different from i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Create mutant vector: V = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant to ensure it stays within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector
            # Select dimensions to crossover based on probability
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is changed (taken from mutant)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            f_trial = func(trial)

            if f_trial < fitness[i]:
                # Trial is better, replace the individual
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if necessary
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
