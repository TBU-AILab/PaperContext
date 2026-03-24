import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the set of input parameter values that lead to the minimum output value
    using the Differential Evolution (DE) algorithm within a limited time.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Algorithm Configuration ---
    # Differential Evolution parameters
    # Population size (NP): A balance between exploration and computation time.
    # We use a dynamic size based on dimension but capped to ensure iterations 
    # can occur if the function evaluation is slow.
    pop_size = max(5, 10 * dim)
    if pop_size > 50:
        pop_size = 50
        
    mutation_factor = 0.8  # F: Scaling factor for mutation (typically 0.5-0.9)
    crossover_prob = 0.9   # CR: Crossover probability (typically 0.8-0.9)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Store fitness values
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    
    # --- Evaluate Initial Population ---
    # We must check time even during initialization in case func is very slow
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            # If we timeout before finishing initialization, return best found so far
            return best_fitness if best_fitness != float('inf') else func(pop[i])
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- Main Differential Evolution Loop ---
    # Continue evolving until time runs out
    while True:
        # Check time at the start of generation
        if (datetime.now() - start_time) >= time_limit:
            break
            
        for i in range(pop_size):
            # Check time frequently (per individual) to adhere strictly to max_time
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness

            # 1. Mutation (Strategy: DE/rand/1)
            # Pick 3 random, distinct individuals other than current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Create mutant vector: V = X_a + F * (X_b - X_c)
            mutant = pop[a] + mutation_factor * (pop[b] - pop[c])
            
            # Bound constraints: Clip mutant vector to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover (Binomial)
            # Create trial vector by mixing target vector and mutant vector
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter comes from the mutant (promotes diversity)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
                
            trial_vector = np.where(cross_points, mutant, pop[i])
            
            # 3. Selection
            # Greedy selection: if trial is better, replace current
            trial_val = func(trial_vector)
            
            if trial_val < fitness[i]:
                fitness[i] = trial_val
                pop[i] = trial_vector
                
                # Update global best if needed
                if trial_val < best_fitness:
                    best_fitness = trial_val

    # return fitness of the best found solution
    return best_fitness
