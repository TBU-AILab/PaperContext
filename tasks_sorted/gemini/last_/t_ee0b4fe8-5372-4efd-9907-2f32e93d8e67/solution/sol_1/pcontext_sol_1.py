#Here is a fully operational, self-contained Python implementation using a **Self-Adaptive Differential Evolution (jDE) algorithm with a 'current-to-pbest' mutation strategy and Restart mechanism**. 
#
#This approach improves upon standard Differential Evolution by:
#1.  **Adaptive Parameters**: Automatically tuning mutation ($F$) and crossover ($CR$) rates during execution to fit the function landscape.
#2.  **Greedy Mutation Strategy**: Using `current-to-pbest` directs the search towards the best known regions while maintaining diversity, converging faster than random search.
#3.  **LHS Initialization**: Using Latin Hypercube Sampling for the initial population ensures better coverage of the search space.
#4.  **Restart Mechanism**: If the population stagnates (converges to a local optimum) before time runs out, the algorithm resets the population (keeping the best solution) to explore new areas.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE)
    with 'current-to-pbest' mutation and a restart mechanism.
    """
    start_time = time.time()
    
    # Pre-process bounds for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Population size: Dynamic based on dimension, clamped to reasonable limits
    # to ensure enough generations run within limited time.
    pop_size = int(max(20, min(100, 15 * dim))) 
    
    # --- Initialization using Latin Hypercube Sampling (LHS) ---
    # LHS guarantees a more stratified distribution than pure random.
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        # Create intervals
        edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
        # Sample uniformly within intervals
        points = np.random.uniform(edges[:-1], edges[1:])
        # Shuffle to mix dimensions
        np.random.shuffle(points)
        pop[:, d] = points
        
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # Initial Evaluation Loop
    for i in range(pop_size):
        # Time check
        if (time.time() - start_time) >= max_time:
            # If time runs out during init, return best found so far
            return best_val if best_val != float('inf') else func(pop[i])
            
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # --- Self-Adaptive Parameters Initialization ---
    # F: Mutation factor, CR: Crossover probability
    # Each individual has its own F and CR which evolve.
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # --- Main Optimization Loop ---
    while True:
        # Strict Global Time Check
        if (time.time() - start_time) >= max_time:
            return best_val

        # Sort population to identify top individuals for p-best selection
        sorted_indices = np.argsort(fitness)
        
        # --- Restart Mechanism ---
        # If population diversity (standard deviation of fitness) drops too low,
        # we are likely stuck in a local optimum. Restart search, keeping the best found.
        if np.std(fitness) < 1e-6:
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Preserve the global best
            pop[0] = best_vec
            
            # Reset fitness array
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Re-evaluate new population (skipping index 0)
            for i in range(1, pop_size):
                if (time.time() - start_time) >= max_time:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            
            # Reset adaptive parameters
            F = np.full(pop_size, 0.5)
            CR = np.full(pop_size, 0.9)
            sorted_indices = np.argsort(fitness)

        # Determine indices for "p-best" (top 10% of population)
        p_limit = max(2, int(pop_size * 0.1))
        top_p_indices = sorted_indices[:p_limit]
        
        # Iterate over population
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
            
            # 1. Parameter Adaptation (jDE logic)
            # With small probability, reset F or CR to new random values
            if np.random.rand() < 0.1:
                F[i] = 0.1 + np.random.rand() * 0.9
            if np.random.rand() < 0.1:
                CR[i] = np.random.rand()
                
            # 2. Mutation: DE/current-to-pbest/1
            # Select random p-best from top p%
            idx_pbest = np.random.choice(top_p_indices)
            
            # Select r1, r2 distinct from i
            # Efficient sampling avoiding current index
            idxs = np.delete(np.arange(pop_size), i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            x_i = pop[i]
            x_pbest = pop[idx_pbest]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Mutation vector
            # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            mutant = x_i + F[i] * (x_pbest - x_i) + F[i] * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < CR[i]
            # Ensure at least one parameter comes from mutant (avoid cloning)
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True 
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # 4. Bound Handling (Clamping)
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()

    return best_val
