#The following Python code implements an improved optimization algorithm based on **Differential Evolution (DE)** with a **"Current-to-Best" mutation strategy** and **Parameter Dithering**.
#
#### Key Improvements:
#1.  **Strategy**: Uses `DE/current-to-best/1/bin`, which guides individuals towards the best solution found so far while maintaining diversity. This typically converges faster than standard `DE/rand/1` for functions with time constraints.
#2.  **Adaptive Parameters (Dithering)**: The mutation factor $F$ and crossover probability $CR$ are randomized for each individual in every generation. This prevents the algorithm from getting stuck due to poor static parameter choices.
#3.  **Dynamic Population Size**: The population size scales with the dimension ($20 + 4 \times dim$), ensuring sufficient coverage for higher-dimensional problems without being too computationally expensive for the time limit.
#4.  **Vectorization**: The generation of trial vectors (mutation and crossover) is fully vectorized using NumPy, reducing interpreter overhead and leaving more time for function evaluations.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function `func` within `max_time` seconds using 
    an improved Differential Evolution algorithm with 'current-to-best' 
    mutation and parameter dithering.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Scales with dimension to balance exploration and speed.
    # A base of 20 plus 4 per dimension provides a robust spread.
    pop_size = 20 + 4 * dim
    
    # Pre-process bounds for efficient clipping
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff = max_b - min_b
    
    # --- Initialization ---
    # Initialize population uniformly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff
    fitness = np.full(pop_size, np.inf)
    
    best_val = np.inf
    best_vec = np.zeros(dim)
    
    # Evaluate Initial Population
    # Loop sequentially to ensure we respect the time limit strictly
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Check time before starting complex vector operations
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        # --- Parameter Dithering ---
        # Randomize F and CR for each individual to improve robustness (Jitter).
        # F in [0.5, 1.0], CR in [0.8, 1.0]
        # Shape (pop_size, 1) allows broadcasting against (pop_size, dim)
        F = 0.5 + 0.5 * np.random.rand(pop_size, 1)
        CR = 0.8 + 0.2 * np.random.rand(pop_size, 1)
        
        # --- Mutation: DE/current-to-best/1 ---
        # Strategy: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
        # This moves individuals towards the best while adding difference vector variation.
        
        # Select random indices r1, r2 for the whole population at once
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Compute mutant vectors (vectorized)
        # Note: best_vec is broadcasted to match population shape
        mutant = population + F * (best_vec - population) + F * (population[r1] - population[r2])
        
        # --- Crossover: Binomial ---
        # Generate random mask based on CR
        cross_points = np.random.rand(pop_size, dim) < CR
        
        # Ensure at least one dimension is taken from the mutant (avoid stagnation)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_points[np.arange(pop_size), j_rand] = True
        
        # Create trial population
        trial_pop = np.where(cross_points, mutant, population)
        
        # --- Bound Handling ---
        # Clip values to stay within specified bounds
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # --- Selection ---
        # Evaluate trials and update population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            # Greedy selection: if trial is better, overwrite current individual
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                # Update global best immediately
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()
                    
    # return fitness of the best found solution
    return best_val
