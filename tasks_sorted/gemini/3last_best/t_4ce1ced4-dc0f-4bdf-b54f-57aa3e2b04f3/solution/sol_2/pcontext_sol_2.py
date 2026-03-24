#The following algorithm implements a **Vectorized Self-Adaptive Differential Evolution (JADE-like variant)**.
#
#**Key Improvements:**
#1.  **Mutation Strategy (`DE/current-to-pbest/1`)**: Unlike the standard `DE/rand/1` (random exploration) used in previous iterations, this strategy directs the search by mutating individuals towards the best solutions currently found (top $p\%$). This significantly accelerates convergence while maintaining robustness.
#2.  **Self-Adaptation**: It employs the jDE mechanism where mutation factor $F$ and crossover rate $CR$ are encoded into individuals and evolve. This removes the need for manual hyperparameter tuning.
#3.  **Vectorization**: The generation of mutant vectors and crossover operations are fully vectorized using NumPy, maximizing the number of generations possible within the time limit.
#4.  **Restart Mechanism**: Detects population stagnation (when fitness variance drops) and restarts the population while preserving the elite solution, ensuring the algorithm doesn't get stuck in local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using Vectorized Adaptive Differential Evolution
    with 'current-to-pbest' mutation (JADE-style) and population restarts.
    """
    start_time = datetime.now()
    # Subtract a small safety buffer to ensure we return before external timeout
    time_limit = timedelta(seconds=max(0.01, max_time - 0.05))

    # ---------------- Hyperparameters ----------------
    # Population size: Balance between diversity and iteration speed.
    # A size of 10-15 * dim is standard, clipped to keep iterations fast.
    pop_size = int(max(20, min(15 * dim, 100)))
    
    # "p-best" rate: Controls greediness. Top 10% guide the mutation.
    p_best_rate = 0.10 
    
    # Restart threshold: If fitness spread < this, restart.
    restart_tol = 1e-8

    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b

    def init_population(n):
        return min_b + np.random.rand(n, dim) * diff_b

    population = init_population(pop_size)
    
    # Adaptive Parameters (jDE style)
    # Each individual has its own F and CR.
    # Initialize conservative values: F=0.5, CR=0.9
    F = 0.5 * np.ones(pop_size)
    CR = 0.9 * np.ones(pop_size)

    # Fitness tracking
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    best_sol = None # Keep track of the elite solution

    # Initial Evaluation
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            # If time runs out during initialization, return best found so far
            return best_fitness if best_sol is not None else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()

    # ---------------- Main Loop ----------------
    while True:
        # Time Check
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness

        # 1. Restart Check (Stagnation Detection)
        # If the population has converged (variance is near zero), we are likely stuck.
        if (np.max(fitness) - np.min(fitness)) < restart_tol:
            # Keep the elite, randomize the rest
            population = init_population(pop_size)
            population[0] = best_sol.copy()
            
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fitness
            
            # Reset adaptive parameters
            F = 0.5 * np.ones(pop_size)
            CR = 0.9 * np.ones(pop_size)
            
            # Re-evaluate new population (skipping index 0)
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_fitness
                val = func(population[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = population[i].copy()
            continue

        # 2. Preparation for Mutation
        # Sort population by fitness to identify "p-best"
        sorted_indices = np.argsort(fitness)
        
        # 3. Parameter Adaptation (jDE Logic)
        # Randomly reset F and CR for a subset of the population to maintain diversity in search logic
        # probabilities: tau_F = 0.1, tau_CR = 0.1
        mask_f = np.random.rand(pop_size) < 0.1
        mask_cr = np.random.rand(pop_size) < 0.1
        
        # F draws from U(0.1, 1.0)
        if np.any(mask_f):
            F[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
        # CR draws from U(0.0, 1.0)
        if np.any(mask_cr):
            CR[mask_cr] = np.random.rand(np.sum(mask_cr))

        # 4. Mutation: DE/current-to-pbest/1
        # Formula: V_i = X_i + F_i * (X_pbest - X_i) + F_i * (X_r1 - X_r2)
        
        # Select X_pbest: Randomly chosen from top p% for each individual
        num_p_best = max(1, int(pop_size * p_best_rate))
        top_indices = sorted_indices[:num_p_best]
        pbest_indices = np.random.choice(top_indices, pop_size)
        x_pbest = population[pbest_indices]

        # Select r1, r2: Random individuals
        # Note: In pure theory r1 != r2 != i, but for speed in Python we allow rare collisions.
        # The noise added is negligible compared to the vectorized speedup.
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1]
        x_r2 = population[r2]
        
        # Calculate mutant vectors (Vectorized)
        # Broadcast F to shape (pop_size, 1) for multiplication
        F_col = F[:, None]
        
        # The mutation has two components:
        # 1. Exploitation: Pull towards p-best (x_pbest - population)
        # 2. Exploration: Random difference (x_r1 - x_r2)
        mutant = population + F_col * (x_pbest - population) + F_col * (x_r1 - x_r2)
        
        # Clip to bounds
        mutant = np.clip(mutant, min_b, max_b)

        # 5. Crossover (Binomial)
        # Generate random matrix to compare against CR
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR[:, None]
        
        # Guarantee: At least one parameter comes from the mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        # Create trial population
        trial_pop = np.where(cross_mask, mutant, population)

        # 6. Selection & Evaluation
        # We must iterate to call func()
        for i in range(pop_size):
            # Strict time check inside the loop to ensure max_time isn't violated by a slow batch
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            f_trial = func(trial_pop[i])
            
            # Greedy selection: if trial is better or equal, it survives
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()

    return best_fitness
