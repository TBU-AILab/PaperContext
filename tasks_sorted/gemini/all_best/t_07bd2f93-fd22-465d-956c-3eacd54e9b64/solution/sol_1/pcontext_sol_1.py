#The following algorithm implements an improved **Differential Evolution (DE)** strategy. To improve upon standard DE, this implementation uses the **DE/current-to-best/1/bin** strategy, which generally converges faster by guiding the population towards the best solution found so far. Crucially, it incorporates a **Restart Mechanism with Elitism**: if the population converges (variance becomes low) before the time limit, the algorithm restarts with a new random population while preserving the best-found individual. This allows the algorithm to escape local minima and continue searching effectively for the remainder of the allocated time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using Differential Evolution with 'current-to-best' strategy
    and a restart mechanism to escape local minima.
    """
    # 1. Time Management
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    def is_timeout():
        return (datetime.now() - start_time) >= time_limit

    # 2. Hyperparameters
    # Population size: 15 * dim is a robust balance between diversity and speed.
    # We ensure a minimum size of 20.
    pop_size = max(20, 15 * dim)
    
    # 3. Initialization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Track global best across restarts
    global_best_val = float('inf')
    global_best_vec = None

    # 4. Main Loop (Restarts)
    # The algorithm runs in epochs. If an epoch converges, it restarts.
    while not is_timeout():
        
        # --- Initialize Population ---
        # Standard uniform random initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best from previous epochs into the new population
        # This ensures we never lose the best solution and helps refine it further.
        start_idx = 0
        if global_best_vec is not None:
            pop[0] = global_best_vec
            fitness[0] = global_best_val
            start_idx = 1 # Skip evaluating the preserved individual
        
        # Evaluate initial population
        for i in range(start_idx, pop_size):
            if is_timeout():
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
        
        # --- Evolution Loop ---
        converged = False
        while not converged and not is_timeout():
            
            # Identify the best individual in the current population
            # (Needed for current-to-best mutation)
            best_idx = np.argmin(fitness)
            current_best_vec = pop[best_idx]
            
            # Dynamic Parameters (Dithering)
            # Randomizing F slightly helps maintain diversity and prevents stagnation
            F = np.random.uniform(0.5, 0.9)
            CR = 0.9  # High crossover rate is good for current-to-best
            
            # Iterate through population
            for i in range(pop_size):
                if is_timeout():
                    return global_best_val
                
                # --- Mutation: DE/current-to-best/1 ---
                # V = Xi + F * (Xbest - Xi) + F * (Xr1 - Xr2)
                
                # Select distinct random indices r1, r2 != i
                r1, r2 = i, i
                while r1 == i:
                    r1 = np.random.randint(pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(pop_size)
                
                x_i = pop[i]
                x_r1 = pop[r1]
                x_r2 = pop[r2]
                
                # Compute mutant vector
                # The term F*(current_best_vec - x_i) guides towards the best
                # The term F*(x_r1 - x_r2) provides exploration
                mutant = x_i + F * (current_best_vec - x_i) + F * (x_r1 - x_r2)
                
                # --- Crossover: Binomial ---
                cross_points = np.random.rand(dim) < CR
                # Ensure at least one dimension is changed
                j_rand = np.random.randint(dim)
                cross_points[j_rand] = True
                
                trial = np.where(cross_points, mutant, x_i)
                
                # --- Bounds Handling ---
                trial = np.clip(trial, min_b, max_b)
                
                # --- Selection ---
                trial_val = func(trial)
                
                # Greedy selection
                if trial_val <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_val
                    
                    # Update global best
                    if trial_val < global_best_val:
                        global_best_val = trial_val
                        global_best_vec = trial.copy()
            
            # --- Convergence Check ---
            # If the fitness values in the population are very close, we are likely stuck.
            # We trigger a restart to explore new areas of the search space.
            if np.std(fitness) < 1e-6:
                converged = True

    return global_best_val
