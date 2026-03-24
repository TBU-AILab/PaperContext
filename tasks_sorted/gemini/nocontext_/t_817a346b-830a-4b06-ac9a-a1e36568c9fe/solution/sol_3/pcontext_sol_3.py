#Here is a fully operational, self-contained Python algorithm based on **Differential Evolution (DE)**.
#
#Differential Evolution is chosen because it is a robust, gradient-free optimization method that handles local minima well and works efficiently within bound constraints, making it superior to simple random search for this template.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the input parameters that minimize the output of 'func' using 
    Differential Evolution within a specified time limit.
    """
    start = datetime.now()
    
    # ---------------------------------------------------------
    # Algorithm Hyperparameters
    # ---------------------------------------------------------
    # Population size: Scales with dimension. 
    # Minimum of 5 required for the mutation strategy (target + 3 random + safety).
    pop_size = max(5, 10 * dim) 
    
    # Differential weight [0, 2]
    mutation_factor = 0.8   
    
    # Crossover probability [0, 1]
    crossover_prob = 0.7    
    
    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track the best solution found
    best = float('inf')
    
    # Array to store fitness of current population
    fitnesses = np.full(pop_size, float('inf'))

    # ---------------------------------------------------------
    # Evaluate Initial Population
    # ---------------------------------------------------------
    # We loop manually to check time constraints during initialization
    for i in range(pop_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time):
            return best if best != float('inf') else float('inf')

        val = func(population[i])
        fitnesses[i] = val
        
        if val < best:
            best = val

    # ---------------------------------------------------------
    # Main Differential Evolution Loop
    # ---------------------------------------------------------
    while True:
        # Check overall time constraint at start of generation
        if (datetime.now() - start) >= timedelta(seconds=max_time):
            return best
            
        # Iterate through the population
        for i in range(pop_size):
            # Check time frequently (inside the loop) to strictly adhere to max_time
            if (datetime.now() - start) >= timedelta(seconds=max_time):
                return best

            # 1. Mutation
            # Select 3 distinct random indices distinct from current index i
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Create mutant vector: V = X_a + F * (X_b - X_c)
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            
            # Clip mutant vector to stay within search space bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover
            # Create trial vector by mixing target vector and mutant vector
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter changes (to avoid evaluating same point)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial_vector = np.where(cross_points, mutant, population[i])

            # 3. Selection
            # Greedy selection: if trial is better, it replaces the current individual
            try:
                f_trial = func(trial_vector)
            except Exception:
                # Fallback if func fails for specific parameters
                f_trial = float('inf')

            if f_trial <= fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial_vector
                
                # Update global best
                if f_trial < best:
                    best = f_trial

    return best
