#Here is a fully operational, self-contained Python implementation using **Differential Evolution**. This algorithm is robust for global optimization problems and works well without derivative information (black-box functions). It checks the time limit strictly after every function evaluation to ensure it returns within the allocated time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Differential Evolution within max_time.
    """
    # Initialize timing
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # Differential Evolution Parameters
    # pop_size: Number of candidate solutions. 15 * dim is a standard heuristic.
    pop_size = max(10, 15 * dim) 
    mutation_factor = 0.8  # F: Controls the amplification of the differential variation
    crossover_prob = 0.7   # CR: Probability of crossover
    
    # Pre-process bounds for faster numpy operations
    # Convert list of lists/tuples to numpy arrays
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # pop: shape (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    
    # Initialize fitness array
    fitness = np.full(pop_size, float('inf'))
    
    # Track the global best
    best_idx = 0
    best_fitness = float('inf')
    
    # --- Initialization Phase ---
    # Evaluate initial population one by one to ensure we don't exceed time 
    # if the function execution is very slow.
    for i in range(pop_size):
        # Check time constraint
        if (datetime.now() - start_time) >= limit:
            return best_fitness
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # --- Evolution Phase ---
    while True:
        for i in range(pop_size):
            # Check time constraint at every step
            if (datetime.now() - start_time) >= limit:
                return best_fitness

            # 1. Mutation: Select 3 distinct random indices distinct from current i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]
            
            # Create mutant vector: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # 2. Crossover: Create trial vector
            # Select random index to ensure at least one parameter comes from mutant
            j_rand = np.random.randint(dim)
            
            # Create mask for crossover
            # random values < CR OR index == j_rand
            cross_points = (np.random.rand(dim) < crossover_prob)
            cross_points[j_rand] = True
            
            # Select elements from mutant where cross_points is True, else from target (pop[i])
            trial = np.where(cross_points, mutant, pop[i])
            
            # 3. Boundary Constraint Handling (Clip to bounds)
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best if this trial is the new best
                if f_trial < best_fitness:
                    best_fitness = f_trial

    return best_fitness
