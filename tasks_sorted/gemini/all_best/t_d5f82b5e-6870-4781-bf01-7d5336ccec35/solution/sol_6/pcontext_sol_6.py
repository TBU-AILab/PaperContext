The output value is: 18.293847510

The improved algorithm code is:
# The improved algorithm implements **JADE (Adaptive Differential Evolution with Optional External Archive)**.
#
# Key Improvements over previous attempts:
# 1.  **External Archive**: The algorithm maintains an archive of inferior solutions recently replaced by better ones. This preserves population diversity and directs the search away from local optima by providing "old" directions for difference vectors.
# 2.  **Midpoint Bound Correction**: Instead of simply clipping values to bounds (which piles points at the edge), or reflecting (which can trap), it uses a midpoint target `(bound + old_pos) / 2`. This maintains the statistical distribution better near boundaries.
# 3.  **Refined Restart Logic**: Monitors the "Peak-to-Peak" (max - min) fitness of the population. If the population collapses to a single point (stagnation), it triggers a hard restart of the non-elite individuals to search new areas.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    # Subtract a small buffer to ensure we return before timeout
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population Size:
    # A size of ~20*dim is robust for JADE-like strategies.
    # Clamped between 25 and 100 to balance exploration vs speed.
    pop_size = int(20 * dim)
    if pop_size < 25: pop_size = 25
    if pop_size > 100: pop_size = 100
    
    # Archive: Stores distinct solutions to maintain diversity
    archive = []
    max_archive_size = pop_size
    
    # jDE Self-Adaptation Probabilities
    tau_F = 0.1
    tau_CR = 0.1
    
    # Strategy: Current-to-pBest
    p_best_rate = 0.11 # Top 11%
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize Adaptive Parameters (F, CR) per individual
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    global_best_val = float('inf')
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return global_best_val
            
        val = func(population[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val
            
    # --- Main Loop ---
    while datetime.now() - start_time < time_limit:
        
        # Sort population by fitness (Best -> Worst)
        # This is required for p-best selection
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        F = F[sorted_indices]
        CR = CR[sorted_indices]
        
        # Create the selection pool for r2: Population U Archive
        # This gives access to history of search
        if len(archive) > 0:
            archive_np = np.array(archive)
            pool = np.vstack((population, archive_np))
        else:
            pool = population
        pool_size = len(pool)
        
        # Determine size of p-best group
        num_p_best = max(2, int(pop_size * p_best_rate))
        
        # Iterate over population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            # --- 1. Parameter Adaptation (jDE) ---
            # Randomly reset F and CR with probability tau
            if np.random.rand() < tau_F:
                F[i] = 0.1 + 0.9 * np.random.rand()
            if np.random.rand() < tau_CR:
                CR[i] = np.random.rand()
                
            # --- 2. Mutation: DE/current-to-pbest/1/bin with Archive ---
            # V = Xi + F*(Xpbest - Xi) + F*(Xr1 - Xr2)
            
            # Select Xpbest (from top p%)
            p_idx = np.random.randint(0, num_p_best)
            x_pbest = population[p_idx]
            
            # Select Xr1 (from population, distinct from i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select Xr2 (from Pool = Pop U Archive, distinct from i and r1)
            r2 = np.random.randint(0, pool_size)
            while r2 == i or (r2 < pop_size and r2 == r1):
                r2 = np.random.randint(0, pool_size)
            x_r2 = pool[r2]
            
            x_i = population[i]
            
            # Compute Mutant Vector
            mutant = x_i + F[i] * (x_pbest - x_i) + F[i] * (x_r1 - x_r2)
            
            # --- 3. Crossover ---
            mask = np.random.rand(dim) < CR[i]
            j_rand = np.random.randint(dim)
            mask[j_rand] = True # Ensure at least one parameter changes
            
            trial = np.where(mask, mutant, x_i)
            
            # --- 4. Bound Handling (Midpoint) ---
            # If a variable violates bounds, set it to the midpoint 
            # between the bound and the previous valid value.
            lower_viol = trial < min_b
            upper_viol = trial > max_b
            
            if np.any(lower_viol):
                trial[lower_viol] = (min_b[lower_viol] + x_i[lower_viol]) / 2.0
            if np.any(upper_viol):
                trial[upper_viol] = (max_b[upper_viol] + x_i[upper_viol]) / 2.0
                
            # --- 5. Evaluation ---
            f_trial = func(trial)
            
            # --- 6. Selection ---
            if f_trial <= fitness[i]:
                # Solution improved: Add old solution to archive
                archive.append(population[i].copy())
                if len(archive) > max_archive_size:
                    # Randomly remove from archive to maintain size
                    del archive[np.random.randint(0, len(archive))]
                
                fitness[i] = f_trial
                population[i] = trial
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
        
        # --- 7. Restart Mechanism ---
        # If population diversity (spread of fitness) is too low, we are stuck.
        # np.ptp calculates Range (Max - Min).
        if np.ptp(fitness) < 1e-6:
            # Keep the elite (index 0 is best due to sort), restart the rest
            archive = [] # Clear archive to reset history
            
            # Randomize positions
            population[1:] = min_b + np.random.rand(pop_size-1, dim) * diff_b
            fitness[1:] = np.full(pop_size-1, float('inf'))
            
            # Reset adaptive parameters
            F[1:] = 0.5
            CR[1:] = 0.9
            
            # Re-evaluate new individuals
            for k in range(1, pop_size):
                if datetime.now() - start_time >= time_limit:
                    return global_best_val
                val = func(population[k])
                fitness[k] = val
                if val < global_best_val:
                    global_best_val = val

    return global_best_val