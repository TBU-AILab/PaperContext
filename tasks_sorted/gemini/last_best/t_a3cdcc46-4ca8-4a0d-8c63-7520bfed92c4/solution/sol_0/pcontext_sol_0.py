#Here is a fully operational, self-contained Python algorithm using **Differential Evolution**. This approach is generally more effective than random search for continuous optimization problems as it learns from the population of solutions while remaining robust against local minima and requiring no gradients.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution within a time limit.
    """
    # Initialize timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Differential Evolution Hyperparameters
    # pop_size: Size of the population. 20 is a robust balance for speed vs diversity.
    pop_size = 20 
    mutation_factor = 0.8  # (F) Controls the amplification of differential variation
    crossover_prob = 0.7   # (CR) Probability of crossover
    
    # Pre-process bounds for efficient numpy operations
    # bounds is a list of [min, max], convert to array for vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # We generate random vectors within the bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Evaluate initial population
    fitness_scores = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    
    # We evaluate the initial population one by one to check time constraints frequently
    for i in range(pop_size):
        if datetime.now() >= end_time:
            return best_fitness
        
        score = func(population[i])
        fitness_scores[i] = score
        
        if score < best_fitness:
            best_fitness = score

    # Main Evolution Loop
    while True:
        # Loop through the population
        for i in range(pop_size):
            # Check time constraint strictly before every expensive function call
            if datetime.now() >= end_time:
                return best_fitness

            # 1. Mutation: Select 3 distinct random individuals (excluding current i)
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = random.sample(idxs, 3)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Clip mutant vector to ensure it stays within bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover: Create trial vector
            # Select random indices for crossover based on probability
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter is taken from mutant to avoid exact copy of parent
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector (u)
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection: Evaluate trial vector
            f_trial = func(trial)
            
            # Greedy selection: if trial is better, replace parent
            if f_trial < fitness_scores[i]:
                fitness_scores[i] = f_trial
                population[i] = trial
                
                # Update global best found so far
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
