#Here is a fully operational, self-contained implementation using the **Differential Evolution** algorithm. 
#
#This algorithm is chosen because it is robust for continuous global optimization problems, handles non-differentiable or non-linear functions well, and requires no external libraries beyond `numpy`. It vectorizes the generation of candidate solutions to maximize the time spent evaluating the function `func`.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using Differential Evolution (DE) within a limited time.
    
    DE Strategy: Best-fit-found-so-far kept globally. 
    Scheme: DE/rand/1/bin (Random mutation, Binomial crossover).
    """
    
    # --- 1. Setup & Constants ---
    start_time = time.time()
    # Use a small safety buffer to ensure we return before the strict cutoff
    end_time = start_time + max_time - 0.05 
    
    # DE Parameters
    # Dynamic population size: Large enough for diversity, small enough for speed.
    # Clamped between 10 and 50 to ensure generations run quickly even in high dim.
    pop_size = max(10, min(50, 10 * dim))
    F = 0.8   # Mutation factor (Differential weight)
    CR = 0.9  # Crossover probability

    # Pre-process bounds for efficient numpy operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- 2. Initialization ---
    # Initialize population randomly within bounds
    # Shape: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Track fitness of population
    fitness = np.full(pop_size, float('inf'))
    
    # Global best tracking
    best_fitness = float('inf')

    # --- 3. Evaluate Initial Population ---
    for i in range(pop_size):
        # Strict time check inside evaluation loop
        if time.time() >= end_time:
            return best_fitness if best_fitness != float('inf') else float('inf')

        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val

    # --- 4. Main Optimization Loop ---
    while time.time() < end_time:
        
        # -- Mutation (Vectorized) --
        # Select 3 random distinct indices for every member of the population.
        # We use randint for speed; occasional collision (picking same index) is acceptable in DE.
        idxs = np.random.randint(0, pop_size, size=(pop_size, 3))
        
        a = pop[idxs[:, 0]]
        b = pop[idxs[:, 1]]
        c = pop[idxs[:, 2]]
        
        # DE/rand/1 mutation formula: v = a + F * (b - c)
        mutant = a + F * (b - c)
        
        # Clip mutants to stay within search space bounds
        mutant = np.clip(mutant, min_b, max_b)
        
        # -- Crossover (Vectorized) --
        # Create mask: True where we take from mutant, False where we keep original
        cross_points = np.random.rand(pop_size, dim) < CR
        
        # Construct trial population
        # trial[i][j] = mutant[i][j] if cross_points[i][j] else pop[i][j]
        trial_pop = np.where(cross_points, mutant, pop)

        # -- Selection (Evaluation Loop) --
        for i in range(pop_size):
            # Check time before every expensive function call
            if time.time() >= end_time:
                return best_fitness

            # Evaluate candidate
            trial_val = func(trial_pop[i])

            # Greedy selection: if trial is better, replace current
            if trial_val < fitness[i]:
                fitness[i] = trial_val
                pop[i] = trial_pop[i]
                
                # Update global best
                if trial_val < best_fitness:
                    best_fitness = trial_val

    return best_fitness
