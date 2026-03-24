import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) within a specified time limit.
    DE is a robust, gradient-free evolutionary algorithm suitable for real-valued global optimization.
    """
    
    # 1. Setup Timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # 2. Algorithm Hyperparameters
    # Population size: Critical for DE. 
    # Too small = premature convergence. Too large = slow generations.
    # We use a dynamic size clamped between 10 and 50 to ensure responsiveness.
    pop_size = int(10 * dim)
    pop_size = np.clip(pop_size, 10, 50)
    
    # Differential Evolution standard parameters (DE/rand/1/bin strategy)
    F = 0.8  # Mutation factor (typically 0.5-1.0)
    CR = 0.9 # Crossover probability (typically 0.7-1.0)

    # 3. Handle Bounds
    # Convert list of lists to numpy arrays for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # 4. Initialize Population
    # We generate the whole block, but evaluate individually to check time
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    # Best parameter set is not strictly required by return signature, 
    # but the algorithm needs to track the best scalar value.
    
    # Evaluate Initial Population
    for i in range(pop_size):
        # Time Check
        if datetime.now() >= end_time:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # 5. Main Optimization Loop (Generational)
    while datetime.now() < end_time:
        for i in range(pop_size):
            # Strict Time Check inside the inner loop to adhere to max_time constraints
            if datetime.now() >= end_time:
                return best_val

            # --- Mutation ---
            # Select 3 unique random indices distinct from current index i
            # Doing this in a loop is acceptable given Python overhead vs func cost
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            
            # Create mutant vector: V = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # --- Crossover ---
            # Binary crossover to create trial vector
            rand_vals = np.random.rand(dim)
            mask = rand_vals < CR
            
            # Ensure at least one parameter is changed (standard DE requirement)
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, population[i])
            
            # --- Boundary Constraint Handling ---
            # Clip values to stay within the defined search space
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            trial_fitness = func(trial)
            
            # Greedy selection: if trial is better, replace parent
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial
                
                # Update global best if improved
                if trial_fitness < best_val:
                    best_val = trial_fitness

    return best_val
