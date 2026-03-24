#The output value of the best generated algorithm is: 15.355046934530776
#
# The best generated algorithm code:
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
#
#The following Python code implements a significantly improved algorithm: **Restart-based Self-Adaptive Differential Evolution (ADE-Restart)** with an **IPOP (Increasing Population)** strategy. 
#
##### Key Improvements over Previous Versions:
#1.  **Restart Mechanism**: Standard DE can get trapped in local optima (stagnation), indicated by the previous result of ~15.35. This algorithm detects convergence (when the population fitness variance drops) and automatically restarts the search to explore new basins of attraction.
#2.  **Elitism**: On restart, the global best solution found so far is injected into the new population, ensuring monotonicity (the result never gets worse).
#3.  **IPOP Strategy**: Each restart increases the population size (scaling by 1.5x). This allows the algorithm to start fast with a small population to find easy solutions quickly, then switch to larger populations to solve more complex, multimodal landscapes if time permits.
#4.  **DE/current-to-pbest/1 Strategy**: Replaces `current-to-best` with `current-to-pbest` (using the top 10% of individuals). This reduces greediness, preventing premature convergence while maintaining high selection pressure.
#5.  **Robust Parameter Dithering**: Uses Normal distributions for `F` and `CR` to explore a wider range of search behaviors per generation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the given function `func` within `max_time` seconds using 
    Restart-based Adaptive Differential Evolution (ADE) with IPOP strategy.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    global_best_vec = None
    
    # IPOP: Start with a moderate population size
    base_pop_size = 20 + 5 * dim
    restart_count = 0
    
    # --- Main Restart Loop ---
    while True:
        # Check time before starting a new restart phase
        if datetime.now() - start_time >= time_limit:
            return global_best_val
            
        # Increase population size for each restart (IPOP strategy)
        # to handle more complex landscapes if the first runs fail or converge too early.
        scaler = 1.5 ** restart_count
        pop_size = int(base_pop_size * scaler)
        
        # --- Initialization ---
        population = min_b + np.random.rand(pop_size, dim) * diff
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        start_idx = 0
        if global_best_vec is not None:
            population[0] = global_best_vec
            fitness[0] = global_best_val
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = population[i].copy()
                
        # --- Inner Optimization Loop ---
        while True:
            # Strict Time Check
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            # --- Convergence Detection ---
            # If population fitness variance is negligible, we are stuck in a local optimum.
            best_in_pop = np.min(fitness)
            worst_in_pop = np.max(fitness)
            
            # If the difference between best and worst is extremely small, trigger restart
            if (worst_in_pop - best_in_pop) < 1e-8:
                break 
            
            # --- DE Strategy: current-to-pbest/1 ---
            
            # 1. Identify p-best (top 10%, minimum 2)
            sorted_indices = np.argsort(fitness)
            num_pbest = max(2, int(0.1 * pop_size))
            top_indices = sorted_indices[:num_pbest]
            
            # 2. Adaptive Parameters (Dithering)
            # F: Normal distribution centered at 0.5 (good for convergence)
            F = np.random.normal(0.5, 0.3, (pop_size, 1))
            F = np.clip(F, 0.1, 1.0)
            
            # CR: Normal distribution centered at 0.9 (good for rotated problems)
            CR = np.random.normal(0.9, 0.1, (pop_size, 1))
            CR = np.clip(CR, 0.0, 1.0)
            
            # 3. Generate Mutants
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            pbest_idx = np.random.choice(top_indices, pop_size)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            x_pbest = population[pbest_idx]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            mutant = population + F * (x_pbest - population) + F * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            mask = np.random.rand(pop_size, dim) < CR
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask, mutant, population)
            
            # 5. Bound Handling
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 6. Selection
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return global_best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_vec = trial_pop[i].copy()
        
        # End of Inner Loop -> Converged. Prepare next restart.
        restart_count += 1
        
    return global_best_val
