#Here is the complete, self-contained Python code implementing a Differential Evolution algorithm. This method is robust for continuous global optimization problems and typically outperforms simple random search by intelligently evolving the population of solutions.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using Differential Evolution (DE) within 'max_time'.
    """
    # 1. Initialization and Configuration
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Extract bound arrays for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # DE Hyperparameters
    # Population size: larger allows more exploration but fewer generations per second.
    # We use a dynamic size based on dimension, capped to ensure speed.
    pop_size = max(5, 10 * dim)
    if pop_size > 50: 
        pop_size = 50
        
    mutation_factor = 0.8  # F: Amplification of differential variation
    crossover_prob = 0.7   # CR: Probability of crossover
    
    # Initialize Population (pop_size x dim)
    # Random values scaled to the bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track fitness of the population
    fitnesses = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    best_idx = -1

    # 2. Initial Evaluation
    # Evaluate the initial population to establish a baseline best
    for i in range(pop_size):
        # Time check
        if datetime.now() >= end_time:
            return best_fitness
        
        val = func(population[i])
        fitnesses[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # 3. Optimization Loop (Evolution)
    # Strategy: DE/rand/1/bin (Random target vector, 1 difference vector, binomial crossover)
    while True:
        for i in range(pop_size):
            # Strict time check before every function evaluation
            if datetime.now() >= end_time:
                return best_fitness
            
            # --- Mutation ---
            # Select 3 distinct random indices (a, b, c) different from current index i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            x_a = population[a]
            x_b = population[b]
            x_c = population[c]
            
            # Generate mutant vector: v = a + F * (b - c)
            mutant = x_a + mutation_factor * (x_b - x_c)
            
            # Constrain mutant to bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover ---
            # Create trial vector by mixing mutant and current population vector
            # Perform binomial crossover
            rand_vals = np.random.rand(dim)
            mask = rand_vals < crossover_prob
            
            # Ensure at least one parameter comes from the mutant to ensure evolution
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial_vector = np.where(mask, mutant, population[i])
            
            # --- Selection ---
            # Evaluate trial vector
            f_trial = func(trial_vector)
            
            # Greedy selection: if trial is better, replace current
            if f_trial < fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial_vector
                
                # Update global best found so far
                if f_trial < best_fitness:
                    best_fitness = f_trial

    # Return fitness of the best found solution
    return best_fitness
