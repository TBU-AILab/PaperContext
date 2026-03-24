#The output value of the last generated algorithm (Hybrid DE + Local Search) was **79.3**, while the previous DE/rand/1 implementation achieved **18.65**. This indicates that the overhead of the local search (Coordinate Descent) was too high, or the greedy "best/1" strategy converged prematurely.
#
#To improve upon the **18.65** score, we need an algorithm that converges faster than DE/rand/1 but avoids the premature stagnation of DE/best/1.
#
#**Proposed Solution: Self-Adaptive DE/current-to-pbest/1 with Restarts**
#This improved algorithm utilizes the **DE/current-to-pbest/1** mutation strategy. This is a state-of-the-art strategy (used in JADE/SHADE variants) that directs the search towards the top $p\%$ of beneficial solutions ("p-best") while maintaining diversity through difference vectors. It balances exploration and exploitation better than purely random or purely greedy strategies. Additionally, we randomize parameters $F$ and $CR$ for each individual to robustly handle different landscape scaling without manual tuning.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using a Self-Adaptive Differential Evolution (DE) algorithm 
    with 'current-to-pbest' mutation and Restarts.
    
    Improvements:
    1. Strategy: DE/current-to-pbest/1/bin - Leads to faster convergence than DE/rand/1 
       by guiding search towards the top p% of solutions.
    2. Adaptive Parameters: F and CR are randomized per individual (JADE-style) to 
       automatically handle different function scalings.
    3. Restarts: If the population converges (low variance), the algorithm restarts 
       with Latin Hypercube Sampling to find new basins of attraction.
    """
    
    # --- Time Management ---
    start_time = time.time()
    # Reserve small buffer for safe return
    time_limit = max_time - 0.05
    
    # --- Pre-computation ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    diff = ub - lb
    
    # Global best tracker
    global_best_val = float('inf')
    
    # --- Configuration ---
    # Population size: clamped to ensure speed in high dimensions while allowing diversity
    # 'current-to-pbest' works well with slightly larger populations, but we need speed.
    pop_size = int(np.clip(20 * dim, 40, 100))
    
    # Greediness of the strategy (p-value for p-best)
    # Top 10% of solutions are considered 'best' guides
    p_val = 0.10 
    
    # --- Main Loop (Restarts) ---
    while True:
        # Check time before starting a new restart cycle
        if time.time() - start_time > time_limit:
            return global_best_val
        
        # 1. Initialization: Latin Hypercube Sampling (LHS)
        # Stratifies samples to cover the domain evenly
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            r = np.random.rand(pop_size)
            pop[:, d] = lb[d] + (perm + r) / pop_size * diff[d]
            
        # Evaluate Initial Population
        fitness = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            if time.time() - start_time > time_limit:
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # 2. Evolutionary Cycle
        while True:
            if time.time() - start_time > time_limit:
                return global_best_val
            
            # Convergence Check: Restart if population has collapsed
            if np.std(fitness) < 1e-6 or (np.max(fitness) - np.min(fitness)) < 1e-6:
                break
            
            # --- Prepare for current-to-pbest ---
            # Sort population by fitness to identify the p-best
            sorted_idx = np.argsort(fitness)
            
            # Select top p% indices
            top_count = max(2, int(p_val * pop_size))
            pbest_indices_pool = sorted_idx[:top_count]
            
            # --- Parameter Generation (Self-Adaptive approximate) ---
            # F: Cauchy(0.5, 0.3) approx via clipped Normal. Controls step size.
            F = np.random.normal(0.5, 0.3, size=(pop_size, 1))
            F = np.clip(F, 0.1, 1.0)
            
            # CR: Normal(0.9, 0.1). Controls crossover (high for convergence).
            CR = np.random.normal(0.9, 0.1, size=(pop_size, 1))
            CR = np.clip(CR, 0.0, 1.0)
            
            # --- Mutation: DE/current-to-pbest/1 ---
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Select random pbest for each individual
            idx_pbest = np.random.choice(pbest_indices_pool, size=pop_size)
            x_pbest = pop[idx_pbest]
            
            # Select r1 and r2 (random individuals)
            idx_r1 = np.random.randint(0, pop_size, size=pop_size)
            idx_r2 = np.random.randint(0, pop_size, size=pop_size)
            x_r1 = pop[idx_r1]
            x_r2 = pop[idx_r2]
            
            # Calculate mutation vectors (vectorized)
            # The term F*(x_pbest - pop) pulls towards best (Exploitation)
            # The term F*(x_r1 - x_r2) maintains diversity (Exploration)
            mutants = pop + F * (x_pbest - pop) + F * (x_r1 - x_r2)
            
            # Handle Bounds (Clip)
            mutants = np.clip(mutants, lb, ub)
            
            # --- Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < CR
            
            # Force at least one dimension from mutant (standard DE requirement)
            j_rand = np.random.randint(0, dim, size=pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trials = np.where(cross_mask, mutants, pop)
            
            # --- Selection ---
            for i in range(pop_size):
                if time.time() - start_time > time_limit:
                    return global_best_val
                
                f_trial = func(trials[i])
                
                # Greedy Selection: Accept if better or equal
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trials[i]
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        
    return global_best_val
