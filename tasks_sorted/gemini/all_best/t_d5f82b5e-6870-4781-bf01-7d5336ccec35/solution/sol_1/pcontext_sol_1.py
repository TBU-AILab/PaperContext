The output value of the best generated algorithm is: 58.16144240805091

The best generated algorithm code:
#Here is a fully operational, self-contained Python implementation using an improved **Differential Evolution (DE)** strategy.
#
#Improvements over standard DE:
#1. **Strategy**: Uses 'current-to-best/1/bin' instead of 'rand/1/bin'. This utilizes the information of the best solution found so far to guide the search, significantly speeding up convergence for limited time budgets.
#2. **Restart Mechanism**: Detects population stagnation (convergence to a local optimum) by monitoring fitness standard deviation. If stagnated, it triggers a restart, preserving the best solution while re-initializing the rest to explore new areas.
#3. **Parameter Dithering**: Randomizes the scaling factor F per mutation to maintain diversity and prevent premature convergence.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the objective function using Differential Evolution with 
    current-to-best mutation and restart mechanism.
    """
    
    # --- Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Population Size
    # Adaptive size based on dimension, clamped to [20, 100] to ensure speed.
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # Crossover Probability
    CR = 0.9
    
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population: Shape (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_idx = -1
    best_val = float('inf')
    best_vec = None
    
    # Function to safely check time and return best
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # Evaluate Initial Population
    for i in range(pop_size):
        if check_time():
            return best_val if best_val != float('inf') else fitness[0]
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i
            best_vec = population[i].copy()

    # --- Main Optimization Loop ---
    while not check_time():
        
        # 1. Restart Mechanism
        # If the population has converged (variance is low), but time remains,
        # we might be stuck in a local optimum. Restart population.
        if np.std(fitness) < 1e-6:
            # Re-initialize population randomly
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            # Preserve the global best found so far
            population[0] = best_vec
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Evaluate new population (skip index 0 as it's the preserved best)
            for k in range(1, pop_size):
                if check_time(): return best_val
                val = func(population[k])
                fitness[k] = val
                if val < best_val:
                    best_val = val
                    best_vec = population[k].copy()
            # Continue to next iteration of main loop
            continue

        # 2. Evolutionary Cycle
        for i in range(pop_size):
            if check_time():
                return best_val

            # Target Vector
            x_i = population[i]
            
            # Select random indices distinct from i
            # We need r1 and r2
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Mutation: DE/current-to-best/1
            # V = X_i + F * (X_best - X_i) + F * (X_r1 - X_r2)
            # This balances exploitation (towards best) and exploration (diff r1-r2)
            
            # Dithering: Randomize F in [0.5, 1.0] for each vector
            F = 0.5 + 0.5 * np.random.rand()
            
            mutant = x_i + F * (best_vec - x_i) + F * (x_r1 - x_r2)
            
            # Crossover: Binomial
            rand_vals = np.random.rand(dim)
            mask = rand_vals < CR
            # Ensure at least one parameter comes from mutant
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, x_i)
            
            # Boundary Constraints: Clip
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            trial_val = func(trial)
            
            if trial_val <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_val
                
                if trial_val < best_val:
                    best_val = trial_val
                    best_vec = trial.copy()
                    
    return best_val