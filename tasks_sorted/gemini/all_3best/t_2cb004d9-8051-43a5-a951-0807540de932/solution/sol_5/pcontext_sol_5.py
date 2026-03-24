#An improved algorithm for finding the minimum of a function within a limited time. This implementation utilizes a **Vectorized Self-Adaptive Differential Evolution (jDE)** with **current-to-pbest mutation**, an **Archive Mechanism**, and **Elitist Restarts**.
#
###### Improvements Explained:
#1.  **Current-to-pbest with Archive**: This algorithm combines the greedy convergence of `current-to-pbest` (driving the population towards the best solutions) with an **external archive** of recently inferior solutions. This archive solves the diversity loss problem seen in previous attempts, allowing `current-to-pbest` to maintain exploration capabilities similar to `rand/1` while converging significantly faster.
#2.  **Vectorized Archive Sampling**: The selection of difference vectors from the union of the Population and Archive is fully vectorized using NumPy masks, ensuring that the advanced mutation strategy does not incur a computational penalty in Python.
#3.  **Self-Adaptive Control Parameters (jDE)**: $F$ (Mutation) and $CR$ (Crossover) parameters are adapted for each individual. This allows the algorithm to learn the appropriate step sizes and crossover rates for the specific landscape automatically.
#4.  **Elitist Restarts with Boundary Reflection**: The algorithm detects stagnation and restarts the population to escape local optima, but critically, it injects the global best solution found so far into the new population. Boundary reflection is used instead of clipping to better maintain the statistical distribution of the population near the edges of the search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized jDE with current-to-pbest mutation,
    Archive mechanism, and Elitist Restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population sizing
    # Dynamic size based on dimension, capped to maintain iteration speed
    # A size of 15*dim is generally robust for DE variants
    pop_size = int(max(20, 15 * dim))
    if pop_size > 80:
        pop_size = 80
        
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    global_best_val = float('inf')
    global_best_x = None
    
    # jDE Constants (Probabilities for parameter updates)
    tau_f = 0.1
    tau_cr = 0.1
    
    # --- Restart Loop ---
    # Restarts the optimization if the population converges (stagnates)
    while True:
        # Check time before starting a new session
        if (datetime.now() - start_time) >= time_limit:
            return global_best_val
            
        # Initialize Population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best from previous runs to preserve progress
        if global_best_x is not None:
            population[0] = global_best_x.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Archive initialization (Ring buffer implementation)
        # Stores inferior solutions to maintain diversity in mutation
        archive = np.zeros((pop_size, dim))
        arc_ptr = 0  # Pointer for ring buffer replacement
        arc_size = 0 # Current number of elements in archive
        
        # Initial Evaluation
        for i in range(pop_size):
            # Batched time check (every 10 evals) to reduce system call overhead
            if i % 10 == 0 and (datetime.now() - start_time) >= time_limit:
                return global_best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_x = population[i].copy()
                
        # Initialize jDE Control Parameters (F and CR)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # --- Evolution Loop ---
        while True:
            # Check time
            if (datetime.now() - start_time) >= time_limit:
                return global_best_val
            
            # Sort Population by fitness
            # This is required for 'current-to-pbest' to identify the top p% individuals
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            F = F[sorted_indices]
            CR = CR[sorted_indices]
            
            # Check Stagnation (Standard Deviation)
            # If variance is tiny, the population has converged to a point. Restart.
            if np.std(fitness) < 1e-6 or (fitness[-1] - fitness[0]) < 1e-8:
                break
            
            # 1. Update Parameters (jDE Logic)
            # Create trial parameters based on update probabilities tau
            mask_f = np.random.rand(pop_size) < tau_f
            mask_cr = np.random.rand(pop_size) < tau_cr
            
            F_trial = F.copy()
            CR_trial = CR.copy()
            
            if np.any(mask_f):
                # F in [0.1, 1.0]
                F_trial[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
            if np.any(mask_cr):
                # CR in [0.0, 1.0]
                CR_trial[mask_cr] = np.random.rand(np.sum(mask_cr))
            
            # 2. Mutation: current-to-pbest/1 with Archive
            # Formula: V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            # X_r2 is chosen from the Union of Population and Archive
            
            # Select p-best (randomly from top 10% of sorted population)
            p_limit = max(2, int(pop_size * 0.1))
            p_indices = np.random.randint(0, p_limit, pop_size)
            x_pbest = population[p_indices]
            
            # Select r1 (randomly from Population)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[r1_indices]
            
            # Select r2 (randomly from Population U Archive)
            pool_size = pop_size + arc_size
            r2_indices = np.random.randint(0, pool_size, pop_size)
            
            # Vectorized construction of x_r2 from the two sources
            x_r2 = np.zeros((pop_size, dim))
            
            # Case A: Index points to Population
            mask_from_pop = r2_indices < pop_size
            if np.any(mask_from_pop):
                x_r2[mask_from_pop] = population[r2_indices[mask_from_pop]]
            
            # Case B: Index points to Archive
            mask_from_arc = ~mask_from_pop
            if np.any(mask_from_arc):
                # Map indices [pop_size, pool_size) -> [0, arc_size)
                arc_indices = r2_indices[mask_from_arc] - pop_size
                x_r2[mask_from_arc] = archive[arc_indices]
            
            # Compute Mutant Vector
            # Note: population is sorted, so population[i] is effectively the 'current' X_i
            diff_pbest = x_pbest - population
            diff_r = x_r1 - x_r2
            mutant = population + F_trial[:, None] * diff_pbest + F_trial[:, None] * diff_r
            
            # 3. Crossover (Binomial)
            rand_j = np.random.randint(0, dim, pop_size)
            mask_cross = np.random.rand(pop_size, dim) < CR_trial[:, None]
            # Ensure at least one dimension is taken from mutant
            mask_cross[np.arange(pop_size), rand_j] = True
            
            trial_pop = np.where(mask_cross, mutant, population)
            
            # 4. Bound Handling (Reflection)
            # Reflect out-of-bound values back into the space
            mask_l = trial_pop < min_b
            if np.any(mask_l):
                trial_pop[mask_l] = 2 * min_b[np.where(mask_l)[1]] - trial_pop[mask_l]
                
            mask_u = trial_pop > max_b
            if np.any(mask_u):
                trial_pop[mask_u] = 2 * max_b[np.where(mask_u)[1]] - trial_pop[mask_u]
            
            # Safe clip to ensure floating point errors don't keep it out
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 5. Selection and Updates
            # Iterate to evaluate and update greedy selection
            for i in range(pop_size):
                if i % 10 == 0 and (datetime.now() - start_time) >= time_limit:
                    return global_best_val
                
                t_val = func(trial_pop[i])
                
                if t_val <= fitness[i]:
                    # Trial is better or equal: Acceptance
                    
                    # Archive Update: Store the replaced parent
                    # Use ring buffer strategy for efficiency
                    archive[arc_ptr] = population[i].copy()
                    arc_ptr = (arc_ptr + 1) % pop_size
                    if arc_size < pop_size:
                        arc_size += 1
                    
                    # Update Population and Control Parameters
                    population[i] = trial_pop[i]
                    fitness[i] = t_val
                    F[i] = F_trial[i]
                    CR[i] = CR_trial[i]
                    
                    # Update Global Best
                    if t_val < global_best_val:
                        global_best_val = t_val
                        global_best_x = trial_pop[i].copy()
                        
    return global_best_val
