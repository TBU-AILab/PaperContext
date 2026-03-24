#The output value of Algorithm 1 (`14.27`) was significantly better than Algorithm 2 (`23.49`). The following proposal improves upon Algorithm 1 by incorporating a more advanced mutation strategy (**current-to-pbest**) inspired by the JADE/SHADE family of algorithms, while retaining the successful **Vectorized Restart** architecture.
#
#This hybrid approach combines the rapid convergence of `current-to-pbest` mutation with the global exploration capabilities of random restarts. This prevents the algorithm from wasting time in local optima (a weakness of standard DE) while converging faster than random mutation DE.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Restarting Differential Evolution (DE)
    with 'current-to-pbest' mutation strategy.
    
    Key Features:
    1. 'current-to-pbest/1' Mutation: Drives the population towards the best individuals 
       found so far, improving convergence speed significantly compared to 'rand/1'.
    2. Vectorization: Uses NumPy for bulk generation of mutants and trial vectors.
    3. Restarts: Detects population stagnation (low fitness variance) and restarts 
       the search to escape local optima.
    4. Adaptive-like Parameters: Randomizes F and CR per individual based on 
       successful distributions (JADE-like) to handle diverse landscapes.
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 
    # Scaled with dimension. Slightly larger than standard DE to support 
    # p-best selection, but clamped to ensure iteration speed.
    pop_size = int(np.clip(dim * 15, 30, 80))
    
    # Restart tolerance: if population fitness std dev drops below this, restart.
    restart_tol = 1e-7
    
    # p-best parameter: top percentage of population to guide mutation
    p_best_rate = 0.15 
    
    # Pre-process bounds for efficient broadcasting
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check time availability before starting a new population
        if datetime.now() >= end_time:
            return global_best_val
            
        # 1. Initialize Population
        # Random uniform initialization
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
        
        # 2. Evolutionary Cycle
        while True:
            # Strict Time Check
            if datetime.now() >= end_time:
                return global_best_val
            
            # Convergence Check (Restart Trigger)
            # If the population is clumped together (low variance), we are likely stuck.
            if np.std(fitness) < restart_tol:
                break # Break inner loop -> triggers restart in outer loop
            
            # --- Vectorized Mutation: current-to-pbest/1 ---
            # Strategy: V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # 1. Identify p-best individuals
            sorted_indices = np.argsort(fitness)
            num_top = max(2, int(pop_size * p_best_rate))
            top_indices = sorted_indices[:num_top]
            
            # For each individual, select a random 'pbest' from the top list
            pbest_idxs = np.random.choice(top_indices, pop_size)
            
            # 2. Select random individuals r1, r2
            # Using random integers is fast. Collisions (r1==r2 or r1==i) 
            # are ignored for performance as they add harmless stochastic noise.
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # 3. Generate Parameters (F and CR) per individual
            # Sample from distributions generally effective for DE (JADE settings)
            # F ~ Normal(0.5, 0.3), clipped to valid range
            F = np.random.normal(0.5, 0.3, pop_size)
            F = np.clip(F, 0.1, 1.2)
            
            # CR ~ Normal(0.9, 0.1), clipped
            CR = np.random.normal(0.9, 0.1, pop_size)
            CR = np.clip(CR, 0.0, 1.0)
            
            # Reshape for broadcasting
            F_col = F[:, np.newaxis]
            
            # 4. Compute Mutant Vector
            # Retrieve vectors based on indices
            x_pbest = pop[pbest_idxs]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Calculate mutation
            # Moves 'pop' towards 'pbest' with a deviation defined by (r1-r2)
            mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # Enforce Bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Vectorized Crossover ---
            # Binomial crossover
            CR_col = CR[:, np.newaxis]
            cross_mask = np.random.rand(pop_size, dim) < CR_col
            
            # Forced Gene: Ensure at least one dimension comes from the mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            # Create Trial Population
            trial_pop = np.where(cross_mask, mutant, pop)
            
            # --- Selection (Evaluation) ---
            for i in range(pop_size):
                if datetime.now() >= end_time:
                    return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection: Accept if better or equal
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial

    return global_best_val
