#The following Python code implements a robust **Restarted Differential Evolution (DE)** algorithm with the **current-to-best/1/bin** strategy.
#
#### Improvements Explained:
#1.  **Search Strategy**: It uses `current-to-best/1/bin` instead of `rand/1/bin`. This strategy directs the search towards the best solution found so far while maintaining exploration through random differences, resulting in significantly faster convergence.
#2.  **Adaptive Restarts**: The algorithm monitors the population's diversity (standard deviation of fitness). If the population converges to a local minimum, it automatically restarts with a fresh population to explore other areas of the search space, maximizing the utility of the available `max_time`.
#3.  **Dithering**: The mutation factor `F` is randomized per generation (Dithering) to prevent stagnation and improve the likelihood of escaping local optima.
#4.  **Vectorized Operations**: Mutant vector generation and crossover are vectorized using NumPy for efficiency, reducing the overhead outside of the objective function evaluations.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarted Differential Evolution with 
    current-to-best/1/bin strategy.
    """
    
    # --- Initialization ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Helper to check time constraint
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- DE Parameters ---
    # Population size: Adaptive based on dimension.
    # Clamped between 20 and 100 to balance exploration with iteration speed.
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # Crossover Probability (CR): 0.9 is generally effective for DE.
    CR = 0.9
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    global_best_val = float('inf')

    # --- Main Loop (Restarts) ---
    while True:
        # Check time before starting a new restart
        if check_time():
            return global_best_val

        # 1. Initialize Population
        # Uniform random distribution within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time():
                return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
        
        # Track the best individual in the *current* population
        # (Necessary for the current-to-best strategy)
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        
        # 2. Evolution Loop (Generations)
        while True:
            if check_time():
                return global_best_val

            # --- Mutation & Crossover ---
            
            # Dithering: Randomize F in [0.5, 1.0] to maintain diversity
            F = 0.5 + 0.5 * np.random.rand()

            # Strategy: current-to-best/1/bin
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            
            # Select random indices r1, r2
            # We allow r1/r2 collision for vectorization speed; impact is negligible.
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            x_best = population[best_idx]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Calculate mutant vectors
            # Term 1: Move towards best
            # Term 2: Random difference for exploration
            mutant = population + F * (x_best - population) + F * (x_r1 - x_r2)
            
            # Apply Bound Constraints
            mutant = np.clip(mutant, min_b, max_b)
            
            # Binomial Crossover
            # Generate mask: True where we take from mutant
            cross_points = np.random.rand(pop_size, dim) < CR
            
            # Enforce at least one parameter comes from mutant (standard DE rule)
            j_rand = np.random.randint(0, dim, pop_size)
            idxs = np.arange(pop_size)
            cross_points[idxs, j_rand] = True
            
            # Construct trial population
            trial_pop = np.where(cross_points, mutant, population)
            
            # --- Selection (Evaluation) ---
            for i in range(pop_size):
                if check_time():
                    return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update local best (for strategy)
                    if f_trial < best_val:
                        best_val = f_trial
                        best_idx = i
                        
                        # Update global best (for return)
                        if f_trial < global_best_val:
                            global_best_val = f_trial
            
            # --- Convergence Check ---
            # If population fitness variance is very small, we are stuck in a basin.
            # Trigger a restart to use remaining time elsewhere.
            if np.std(fitness) < 1e-8:
                break 

    return global_best_val
