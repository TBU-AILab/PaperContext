#Here is a fully operational, self-contained Python implementation using a **Differential Evolution** algorithm. This approach is generally much more efficient than random search for global optimization, as it uses the differences between current solutions to navigate the search space intelligently, while strictly adhering to the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    best_fitness = float('inf')

    # --- Algorithm Configuration (Differential Evolution) ---
    # Population size: Kept relatively small (20) to ensure multiple generations 
    # can run even within tight time constraints.
    pop_size = 20
    F = 0.8       # Differential weight (scale factor)
    CR = 0.9      # Crossover probability

    # Prepare bounds as numpy arrays for vectorized operations
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # --- Initialization ---
    # Create random initial population: shape (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitnesses = np.full(pop_size, float('inf'))

    # Evaluate initial population
    for i in range(pop_size):
        # Time check before every expensive function call
        if datetime.now() - start_time >= limit:
            return best_fitness
        
        score = func(pop[i])
        fitnesses[i] = score
        
        if score < best_fitness:
            best_fitness = score

    # --- Main Loop (Evolution) ---
    # Strategy: DE/rand/1/bin
    while True:
        for i in range(pop_size):
            # Strict time check
            if datetime.now() - start_time >= limit:
                return best_fitness

            # 1. Mutation: pick 3 distinct random agents a, b, c, none equal to i
            candidates = [idx for idx in range(pop_size) if idx != i]
            # Fast random selection
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)

            # 2. Crossover: Binomial crossover
            rand_vals = np.random.rand(dim)
            # Create mask where crossover happens
            cross_mask = rand_vals < CR
            # Ensure at least one parameter is taken from mutant (standard DE rule)
            if not np.any(cross_mask):
                cross_mask[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_mask, mutant, pop[i])

            # 3. Boundary Handling: Clip trial vector to be within bounds
            trial = np.clip(trial, min_b, max_b)

            # 4. Selection: Evaluate trial vector
            score = func(trial)

            # Greedy selection: if trial is better, replace existing individual
            if score < fitnesses[i]:
                fitnesses[i] = score
                pop[i] = trial
                
                # Update global best if necessary
                if score < best_fitness:
                    best_fitness = score

    return best_fitness
