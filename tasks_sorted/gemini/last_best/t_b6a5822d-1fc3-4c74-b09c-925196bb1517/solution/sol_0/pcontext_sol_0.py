#Here is a fully operational implementation using the **Differential Evolution** algorithm. 
#
#Differential Evolution is a robust metaheuristic used for global optimization. It is significantly more efficient than Random Search as it uses the differences between existing solutions to guide the search direction, allowing it to navigate out of local minima and converge on a global minimum efficiently.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Differential Evolution (DE) parameters
    # Population size: Adaptive to dimension, clamped to [10, 50] to balance
    # exploration vs. iteration speed within limited time.
    pop_size = max(10, min(50, 10 * dim)) 
    mutation_factor = 0.8   # F: Differential weight [0, 2]
    crossover_prob = 0.7    # CR: Crossover probability [0, 1]

    # Time tracking
    start = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population uniformly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness vector
    fitness = np.full(pop_size, float('inf'))
    best = float('inf')
    best_idx = -1

    # Evaluate initial population
    # We check time inside this loop to handle cases where func is very slow
    for i in range(pop_size):
        if datetime.now() - start >= time_limit:
            # If we time out during initialization, return best found so far
            return best
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best:
            best = val
            best_idx = i

    # Main DE Loop
    while True:
        for i in range(pop_size):
            # Check time constraint strictly before every evaluation
            if datetime.now() - start >= time_limit:
                return best

            # 1. Mutation
            # Select 3 distinct random individuals (a, b, c) different from current i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Bound constraints: Clip mutant to stay within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover
            # Binomial crossover: Construct trial vector
            cross_points = np.random.rand(dim) < crossover_prob
            # Ensure at least one parameter comes from the mutant
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection
            # Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better, replace current individual
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                
                # Update global best if trial is better than best found so far
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
