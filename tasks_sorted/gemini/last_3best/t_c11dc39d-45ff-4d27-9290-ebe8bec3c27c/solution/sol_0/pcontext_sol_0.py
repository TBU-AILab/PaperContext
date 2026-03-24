#Here is a fully operational, self-contained Python solution using the **Differential Evolution** algorithm. 
#
#Differential Evolution is a robust, population-based optimization algorithm that generally performs much better than random search or simple gradient descent, especially for functions that may have local minima or are not differentiable.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # pop_size: Number of candidate solutions. 15 is a balance between diversity and speed.
    pop_size = 15 
    mutation_factor = 0.8  # (F) Weighting factor for differential mutation [0, 2]
    crossover_prob = 0.7   # (CR) Probability of crossover [0, 1]
    
    # Pre-process bounds for efficient numpy operations
    bounds_np = np.array(bounds)
    lower_bounds = bounds_np[:, 0]
    upper_bounds = bounds_np[:, 1]
    bound_diff = upper_bounds - lower_bounds
    
    population = []
    fitness_scores = []
    best = float('inf')
    
    # --- Initialization Phase ---
    # Generate initial population one by one to safely handle very short time limits
    for _ in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best
        
        # Generate random individual within bounds
        individual = lower_bounds + np.random.rand(dim) * bound_diff
        
        # Evaluate fitness
        fitness = func(individual)
        
        population.append(individual)
        fitness_scores.append(fitness)
        
        if fitness < best:
            best = fitness

    # Convert lists to numpy arrays for vectorization in the loop
    population = np.array(population)
    fitness_scores = np.array(fitness_scores)
    
    # --- Evolution Loop ---
    while True:
        # Iterate through each individual in the population
        for i in range(pop_size):
            # strict time check before every evaluation
            if (datetime.now() - start_time) >= time_limit:
                return best
            
            # 1. Mutation
            # Select 3 distinct random individuals (a, b, c) different from current individual i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector: V = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant vector to stay within defined bounds
            mutant = np.clip(mutant, lower_bounds, upper_bounds)
            
            # 2. Crossover
            # Create trial vector by mixing current individual and mutant
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter changes to avoid evaluating the same point
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial_vector = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            trial_fitness = func(trial_vector)
            
            # Greedy selection: if trial is better, replace the existing individual
            if trial_fitness < fitness_scores[i]:
                fitness_scores[i] = trial_fitness
                population[i] = trial_vector
                
                # Update global best found so far
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
