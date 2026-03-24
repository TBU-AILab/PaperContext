#The output value is: 57.84797866648188
#
#The algorithm code is:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timer
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # ---------------------------------------------------------
    # Algorithm Hyperparameters
    # ---------------------------------------------------------
    # Population size: Adaptive based on dimension.
    # Capped between 20 and 60 to balance exploration and speed within limited time.
    # This size allows for 'current-to-best' strategy to work effectively without
    # exhausting time on function evaluations per generation.
    pop_size = int(np.clip(10 * dim, 20, 60))
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    global_best_val = float('inf')
    
    # ---------------------------------------------------------
    # Main Loop: Restart Strategy
    # ---------------------------------------------------------
    # We use restarts to escape local optima. 
    # If the population converges (low variance), we restart the search
    # with a new random population to explore other regions.
    while True:
        # Check time budget before starting a new population
        if datetime.now() - start_time >= time_limit:
            return global_best_val
            
        # Initialize Population (Uniform Random)
        # Shape: (pop_size, dim)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
                
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
        
        # Identify local best for the current population (needed for mutation)
        best_idx = np.argmin(fitness)
        local_best_pos = pop[best_idx].copy()
        local_best_val = fitness[best_idx]
        
        # -----------------------------------------------------
        # Evolutionary Loop (Generations)
        # -----------------------------------------------------
        while True:
            # Time Check
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            # Convergence Check
            # If standard deviation of fitness is very low, assume convergence and restart
            if np.std(fitness) < 1e-6:
                break 
            
            # -------------------------------------------------
            # Mutation: DE/current-to-best/1/bin
            # Equation: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This strategy converges faster than DE/rand/1
            # -------------------------------------------------
            
            # F (Scaling Factor): Randomized per individual [0.5, 1.0) (Dithering)
            # Dithering helps prevent stagnation
            F = 0.5 + 0.5 * np.random.rand(pop_size, 1)
            
            # CR (Crossover Probability)
            CR = 0.9 
            
            # Select random indices r1, r2 for difference vectors
            # We approximate distinctness for speed using simple random draws.
            idxs = np.random.randint(0, pop_size, size=(pop_size, 2))
            r1 = pop[idxs[:, 0]]
            r2 = pop[idxs[:, 1]]
            
            # Calculate Mutation Vectors
            # Broadcasting: local_best_pos (dim,) is broadcasted to (pop_size, dim)
            mutation_vec = pop + F * (local_best_pos - pop) + F * (r1 - r2)
            
            # -------------------------------------------------
            # Crossover (Binomial)
            # -------------------------------------------------
            # Generate mask based on CR
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < CR
            
            # Guarantee at least one parameter changes per individual
            j_rand = np.random.randint(0, dim, size=pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            # Create Trial Population
            trial_pop = np.where(cross_mask, mutation_vec, pop)
            
            # Enforce Bounds
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # -------------------------------------------------
            # Selection (Greedy)
            # -------------------------------------------------
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Update if trial is better or equal
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # Update global best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                    
                    # Update local best (for next generation's mutation)
                    if f_trial < local_best_val:
                        local_best_val = f_trial
                        local_best_pos = trial_pop[i].copy()
                        
    return global_best_val
