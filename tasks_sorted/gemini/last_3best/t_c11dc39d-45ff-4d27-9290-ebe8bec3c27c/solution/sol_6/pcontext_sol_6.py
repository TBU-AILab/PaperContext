#The following algorithm implements a **Vectorized Restarting Differential Evolution** with a **"current-to-best/1"** mutation strategy.
#
#**Key Improvements:**
#1.  **Current-to-Best Mutation**: Unlike the standard `rand/1` used in Algorithm 1, this strategy (`V = X + F(X_best - X) + F(X_r1 - X_r2)`) actively guides the population towards the best solution found in the current basin. This results in significantly faster convergence, which is crucial for limited-time optimization.
#2.  **Stagnation Detection**: In addition to variance-based restarts (checking if the population has converged), this implementation tracks **stagnation** (generations without improvement). If the algorithm gets stuck in a local optimum but hasn't fully collapsed, it forces a restart to use the remaining time efficiently.
#3.  **Broad Parameter Dithering**: Randomizes mutation factor ($F$) and crossover rate ($CR$) over a wide range per generation. This allows the algorithm to dynamically switch between exploration (high F, high CR) and exploitation (low F, low CR) without complex adaptation logic.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Restarting Differential Evolution (VRDE)
    with 'current-to-best/1' mutation strategy.
    
    This approach balances fast convergence (exploiting best found solutions)
    with global search (restarts and random mutation components).
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size: 
    # Clamped between 20 and 60. Smaller populations converge faster, 
    # which is ideal when combined with a restart strategy.
    pop_size = int(np.clip(dim * 10, 20, 60))
    
    # Restart triggers
    restart_tol = 1e-7      # Restart if fitness standard deviation is below this
    max_stagnation = 30     # Restart if no improvement for this many generations

    # Pre-process bounds for efficient broadcasting
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Global best tracking
    global_best_val = float('inf')

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Strict time check
        if datetime.now() >= end_time:
            return global_best_val

        # 1. Initialize Population
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
        
        # Track local best for 'current-to-best' mutation
        best_idx = np.argmin(fitness)
        best_val_current = fitness[best_idx]
        stagnation_count = 0

        # 2. Evolutionary Cycle
        while True:
            # Time check
            if datetime.now() >= end_time:
                return global_best_val
            
            # --- Restart Check ---
            # Restart if population converged (low variance) or stagnated
            if np.std(fitness) < restart_tol or stagnation_count > max_stagnation:
                break # Break inner loop -> triggers restart in outer loop

            # --- Parameter Dithering ---
            # Randomize parameters per generation to handle various landscapes.
            # F (Mutation): [0.3, 0.9]. 
            #   Lower values (<0.5) favor exploitation (moving to best).
            #   Higher values favor exploration.
            F = 0.3 + 0.6 * np.random.rand() 
            
            # CR (Crossover): [0.1, 0.9].
            #   Allows both separable (low CR) and dependent (high CR) optimization.
            CR = 0.1 + 0.8 * np.random.rand()

            # --- Vectorized Mutation: current-to-best/1 ---
            # Formula: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This pulls every individual towards the best while maintaining diversity via difference vector.
            
            idxs = np.arange(pop_size)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Note: x_best is a vector (dim,). Subtraction broadcasts it to (pop_size, dim).
            x_best = pop[best_idx]
            
            # Compute mutant vectors
            mutant = pop + F * (x_best - pop) + F * (pop[r1] - pop[r2])
            
            # Enforce bounds
            mutant = np.clip(mutant, min_b, max_b)

            # --- Vectorized Crossover ---
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < CR
            
            # Forced Gene: Ensure at least one dimension is mutated
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[idxs, j_rand] = True
            
            # Create trial population
            trial_pop = np.where(cross_mask, mutant, pop)

            # --- Selection (Sequential Evaluation) ---
            improved_in_generation = False
            
            for i in range(pop_size):
                if datetime.now() >= end_time:
                    return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # Update local best (for mutation in next step)
                    if f_trial < best_val_current:
                        best_val_current = f_trial
                        best_idx = i
                        improved_in_generation = True
                        
                        # Update global best
                        if f_trial < global_best_val:
                            global_best_val = f_trial
            
            # Update stagnation counter
            if improved_in_generation:
                stagnation_count = 0
            else:
                stagnation_count += 1
                
    return global_best_val
