#Here is an implementation using the **Differential Evolution (DE)** algorithm. DE is a robust, derivative-free global optimization method that performs significantly better than random search or simple hill-climbing for continuous parameter spaces, especially within time constraints.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Differential Evolution Hyperparameters
    F = 0.8         # Mutation factor (typically 0.5-1.0)
    CR = 0.9        # Crossover probability (typically 0.7-1.0)
    
    # Population size: Standard is 10*dim, but constrained by max_time.
    # We use a balanced size to ensure we get through iterations quickly.
    # Min 10 to ensure diversity, Max 50 to avoid slow initialization.
    pop_size = max(10, min(50, 2 * dim))
    
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Pre-process bounds for efficient numpy operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = np.fabs(max_b - min_b)
    
    # Initialize best found value
    best = float('inf')
    
    # Initialize Population
    # We store population as a matrix (pop_size x dim)
    # and fitness as an array (pop_size)
    population = np.zeros((pop_size, dim))
    fitness_scores = np.full(pop_size, float('inf'))
    
    # 1. Initialization Phase
    for i in range(pop_size):
        # Time check inside loop to be responsive
        if (datetime.now() - start_time) >= limit:
            return best
            
        # Generate random individual within bounds
        ind = min_b + np.random.rand(dim) * diff_b
        population[i] = ind
        
        # Evaluate
        score = func(ind)
        fitness_scores[i] = score
        
        if score < best:
            best = score

    # 2. Evolution Phase
    while True:
        for i in range(pop_size):
            # Strict time check before every function evaluation
            if (datetime.now() - start_time) >= limit:
                return best
            
            # --- Mutation ---
            # Select 3 distinct random indices distinct from current i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # --- Crossover ---
            # Binomial crossover: select parameters from mutant or target
            rand_vals = np.random.rand(dim)
            # Ensure at least one parameter is taken from mutant (fixed random index j_rand)
            j_rand = np.random.randint(0, dim)
            
            mask = (rand_vals < CR)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, population[i])
            
            # --- Boundary Handling ---
            # Clip values to stay within provided bounds
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            # Greedy selection: if trial is better, replace target
            trial_fitness = func(trial)
            
            if trial_fitness < fitness_scores[i]:
                fitness_scores[i] = trial_fitness
                population[i] = trial
                
                # Update global best if found
                if trial_fitness < best:
                    best = trial_fitness
    
    return best
