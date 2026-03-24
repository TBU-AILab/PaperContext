#The output value of the last generated algorithm is: 43.6522096413823
#
#The following algorithm implements **Adaptive Differential Evolution (ADE) with Latin Hypercube Initialization and Restart Mechanism**.
#
#**Improvements over previous version:**
#1.  **Latin Hypercube Sampling (LHS):** Instead of random initialization, LHS ensures the initial population is evenly distributed across the search space, reducing the chance of missing the global basin of attraction early on.
#2.  **Adaptive Parameters:** The mutation factor `F` and crossover rate `CR` are randomized per individual (dithered) within specific ranges. This prevents the population from stagnating due to fixed step sizes and allows different "search behaviors" (exploration vs. exploitation) to coexist.
#3.  **Strategy 'current-to-best/1/bin':** This mutation strategy converges faster than `rand/1/bin` by incorporating information from the best solution found so far.
#4.  **Restart Mechanism:** If the population converges (variance drops near zero) and stops improving, the algorithm saves the best solution and triggers a "soft restart" of the remaining population. This allows it to escape local optima and utilize the remaining time effectively.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Adaptive Differential Evolution with LHS Init and Restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Adaptive based on dimension, clamped for performance
    # Sufficient size is needed for DE to work, but too large slows down convergence in limited time.
    pop_size = int(max(10, min(100, 15 * dim)))
    
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization: Latin Hypercube Sampling (LHS) ---
    # LHS guarantees a stratified sample across all dimensions, superior to random uniform
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        # Divide dimension into N intervals
        edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
        # Sample uniformly from each interval
        samples = np.random.uniform(edges[:-1], edges[1:])
        # Shuffle to break correlation between dimensions
        np.random.shuffle(samples)
        population[:, d] = samples

    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    best_idx = -1
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Optimization Loop ---
    stall_generations = 0
    
    while True:
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        # --- Restart Mechanism ---
        # If population diversity is lost (low std dev) and no improvement (stall), restart.
        if stall_generations > 20 and np.std(fitness) < 1e-6:
            # Preserve the single best individual
            best_individual = population[best_idx].copy()
            
            # Re-initialize the rest of the population randomly to explore new areas
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_individual # Keep the champion
            
            # Reset fitness array
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            best_idx = 0
            
            # Evaluate new population (skip index 0)
            for i in range(1, pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_val
                val = func(population[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_idx = i
                    population[0] = population[best_idx] # Keep safe ref
            
            stall_generations = 0
            continue

        prev_best_val = best_val
        
        # Snapshot of best vector for mutation strategy
        x_best = population[best_idx].copy()
        
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            # --- Adaptive Parameters ---
            # Randomize F and CR slightly for each individual (Dithering)
            # F around 0.5-0.9 balances exploration/exploitation
            F = 0.4 + 0.5 * np.random.rand() 
            # CR around 0.8-1.0 encourages rotationally invariant search
            CR = 0.8 + 0.2 * np.random.rand()
            
            # Select random indices distinct from i
            # We need 2 distinct random vectors r1, r2
            idxs = np.random.randint(0, pop_size, 3)
            while i in idxs or idxs[0] == idxs[1]:
                idxs = np.random.randint(0, pop_size, 3)
            r1, r2 = idxs[0], idxs[1]
            
            # --- Mutation: current-to-best/1 ---
            # V = Xi + F*(Xbest - Xi) + F*(Xr1 - Xr2)
            # This guides the search towards the best known solution while maintaining diversity via difference vector
            x_i = population[i]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            
            # Handle Bounds (Clip)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover (Binomial) ---
            cross_mask = np.random.rand(dim) < CR
            # Ensure at least one dimension is changed
            if not np.any(cross_mask):
                cross_mask[np.random.randint(dim)] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # --- Selection ---
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i
                    # Greedy update of x_best for subsequent individuals in the same generation
                    x_best = trial 
        
        # Check for improvement
        if best_val < prev_best_val:
            stall_generations = 0
        else:
            stall_generations += 1

    return best_val
