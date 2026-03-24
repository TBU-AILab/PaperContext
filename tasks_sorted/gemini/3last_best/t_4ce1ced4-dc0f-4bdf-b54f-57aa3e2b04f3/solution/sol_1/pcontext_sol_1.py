#The following algorithm implements a **Self-Adaptive Differential Evolution (jDE) with Vectorized Operations and Population Restarts**.
#
#**Key Improvements over standard Differential Evolution:**
#1.  **Vectorization:** Instead of looping through the population to create mutants, it operates on the entire population matrix at once using Numpy. This significantly increases iteration speed (generations per second) in Python.
#2.  **Self-Adaptation (jDE):** The parameters $F$ (mutation factor) and $CR$ (crossover probability) are not fixed. Each individual carries its own $F$ and $CR$ values which evolve over time. This allows the algorithm to adapt to the specific landscape of the objective function (e.g., separable vs. non-separable).
#3.  **Restart Mechanism:** If the population converges (the spread of fitness values becomes very small) before the time limit, the algorithm triggers a "soft restart." It keeps the best solution found so far but re-initializes the rest of the population to explore new areas of the search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using Vectorized Self-Adaptive Differential Evolution (jDE)
    with a Restart Mechanism.
    """
    # ---------------- Setup ----------------
    start_time = datetime.now()
    # Safety buffer: stop 0.05s early to ensure return
    time_limit = timedelta(seconds=max(0.1, max_time - 0.05))
    
    # ---------------- Hyperparameters ----------------
    # Population size: Lower than standard DE to allow more generations in limited time,
    # but high enough to maintain diversity.
    pop_size = max(20, min(10 * dim, 100))
    
    # jDE Self-Adaptation parameters
    tau_F = 0.1
    tau_CR = 0.1
    
    # Convergence threshold for restart
    restart_tol = 1e-6

    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Function to generate random population within bounds
    def init_population(n):
        return min_b + np.random.rand(n, dim) * diff_b

    population = init_population(pop_size)
    
    # Initialize F and CR for each individual
    # F ~ U(0.1, 1.0), CR ~ U(0.0, 1.0)
    F = 0.1 + 0.9 * np.random.rand(pop_size)
    CR = np.random.rand(pop_size)
    
    # Initial Evaluation
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        # Initial check, though unlikely to timeout here
        if (datetime.now() - start_time) >= time_limit:
            return float('inf') 
        fitness[i] = func(population[i])
        
    # Track Global Best
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_sol = population[best_idx].copy()
    
    # ---------------- Main Optimization Loop ----------------
    while True:
        # 1. Time Check
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness
            
        # 2. Convergence Check & Restart Strategy
        # If population diversity is lost (all individuals trapped in same basin), restart.
        current_min = np.min(fitness)
        current_max = np.max(fitness)
        
        # If the spread of fitness is tiny, or we've just been stagnant too long
        # (Using a simple spread check here for efficiency)
        if (current_max - current_min) < restart_tol:
            # Preserve the single best, randomize the rest
            population = init_population(pop_size)
            population[0] = best_sol # Keep elite
            
            # Reset adaptive parameters
            F = 0.1 + 0.9 * np.random.rand(pop_size)
            CR = np.random.rand(pop_size)
            
            # Re-evaluate (except elite)
            fitness[0] = best_fitness
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_fitness
                fitness[i] = func(population[i])
            
            # Continue to next iteration immediately
            continue

        # 3. jDE Parameter Adaptation
        # Update F
        mask_F = np.random.rand(pop_size) < tau_F
        if np.any(mask_F):
            F[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
            
        # Update CR
        mask_CR = np.random.rand(pop_size) < tau_CR
        if np.any(mask_CR):
            CR[mask_CR] = np.random.rand(np.sum(mask_CR))
            
        # 4. Mutation (DE/rand/1) - Fully Vectorized
        # We select 3 random vectors for every individual. 
        # Using permutations is faster than selecting distinct sets per row strictly,
        # and statistically sufficient.
        idx_r1 = np.random.permutation(pop_size)
        idx_r2 = np.random.permutation(pop_size)
        idx_r3 = np.random.permutation(pop_size)
        
        # r1, r2, r3 must be different from 'target' (i). 
        # In a vectorized approach, collisions are rare enough to ignore for speed 
        # or accepted as "noise".
        
        # Mutant V = X_r1 + F * (X_r2 - X_r3)
        # Reshape F for broadcasting: (pop_size, 1)
        mutant = population[idx_r1] + F[:, None] * (population[idx_r2] - population[idx_r3])
        
        # Bound Constraints (Clipping)
        mutant = np.clip(mutant, min_b, max_b)
        
        # 5. Crossover (Binomial) - Vectorized
        rand_matrix = np.random.rand(pop_size, dim)
        # CR check
        cross_mask = rand_matrix < CR[:, None]
        # Ensure at least one dimension is changed (guarantee)
        j_rand = np.random.randint(0, dim, pop_size)
        # Use simple indexing to set the guarantee
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)
        
        # 6. Selection (Evaluate Trials)
        # We must loop for evaluation because func might not support 2D arrays (batching).
        # However, the generation math above was vectorized and fast.
        for i in range(pop_size):
            # Frequent time check is crucial
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
                
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    
    return best_fitness
