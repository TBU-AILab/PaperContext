#Here is a fully operational, self-contained implementation using the **Differential Evolution** algorithm. 
#
#This algorithm is significantly more efficient than random search for finding global minima in continuous spaces. It evolves a population of candidate solutions by mixing them (mutation and crossover) to explore the search space and exploit promising areas, all while strictly adhering to the `max_time` constraint.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # --- Configuration for Differential Evolution ---
    # Population size: Balance between exploration (high size) and speed (low size).
    # We use a dynamic size capped at 50 to ensure the loop runs fast enough.
    pop_size = max(10, dim * 2)
    pop_size = min(pop_size, 50) 
    
    mutation_factor = 0.8  # Scaling factor for mutation (F)
    crossover_prob = 0.9   # Probability of crossover (CR)

    start = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Initialize best fitness found so far
    best = float('inf')

    # Convert bounds to numpy arrays for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # --- Initialization Phase ---
    # Create random initial population within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness_scores = np.full(pop_size, float('inf'))

    # Evaluate initial population
    for i in range(pop_size):
        # Time check inside initialization to prevent timeout on slow functions
        if datetime.now() - start >= time_limit:
            return best
        
        score = func(population[i])
        fitness_scores[i] = score
        
        if score < best:
            best = score

    # --- Evolution Loop ---
    while True:
        # Check time at the start of every generation
        if datetime.now() - start >= time_limit:
            return best

        for i in range(pop_size):
            # Check time frequently (before every function evaluation)
            if datetime.now() - start >= time_limit:
                return best

            # 1. Mutation: Select 3 distinct random agents (a, b, c) distinct from i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)

            # 2. Crossover: Create trial vector by mixing mutant and target
            rand_vals = np.random.rand(dim)
            cross_points = rand_vals < crossover_prob
            
            # Ensure at least one dimension is swapped to introduce variation
            j_rand = np.random.randint(0, dim)
            cross_points[j_rand] = True
            
            trial = np.where(cross_points, mutant, population[i])

            # 3. Boundary Constraint: Clip values to stay within bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection: Greedy approach
            trial_fitness = func(trial)

            # If trial is better, replace the individual in the population
            if trial_fitness < fitness_scores[i]:
                fitness_scores[i] = trial_fitness
                population[i] = trial
                
                # Update global best found solution immediately
                if trial_fitness < best:
                    best = trial_fitness

    # return fitness of the best found solution
    return best
