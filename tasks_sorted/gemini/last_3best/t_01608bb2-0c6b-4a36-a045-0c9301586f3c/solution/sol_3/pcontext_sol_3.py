#The following algorithm implements an improved **Self-Adaptive Differential Evolution (jDE)** enhanced with **Linear Population Size Reduction (LPSR)** and a **Bounce-Back Boundary Handling** strategy. 
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR)**: The population size starts large to encourage exploration and linearly decreases over time to a minimum size. This forces the algorithm to converge and exploit the best found areas as the deadline approaches.
#2.  **Canonical jDE Parameter Adaptation**: Implements the correct jDE logic where mutation ($F$) and crossover ($CR$) parameters are adapted only if the offspring is superior to the parent. This allows the algorithm to learn the optimal parameters for the specific function landscape.
#3.  **Bounce-Back Boundary Handling**: Instead of blindly clipping values to bounds (which concentrates search on edges), this method reflects violations back into the search space, preserving the momentum of the mutation.
#4.  **Restart with Memory**: When the population converges (variance becomes low), the algorithm restarts to find other basins of attraction, but it injects the global best solution into the new population to refine it further if possible.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial Population: Scaled with dimension, capped for performance
    # High initial size ensures adequate exploration of the search space
    pop_size_init = min(500, max(30, 20 * dim))
    pop_size_min = 5 # Minimum required for DE/rand/1 mutation
    
    # jDE Adaptation Probabilities
    tau_F = 0.1
    tau_CR = 0.1
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Main Optimization Loop (Restart Mechanism) ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Initialization for new Restart
        pop_size = pop_size_init
        
        # Initialize Randomly
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # jDE Control Parameters (Initialized Randomly)
        # F in [0.1, 0.9], CR in [0.0, 1.0]
        F = 0.1 + 0.8 * np.random.rand(pop_size)
        CR = np.random.rand(pop_size)
        
        # Inject global best from previous runs (Elitism)
        start_eval_idx = 0
        if best_sol is not None:
            population[0] = best_sol
            fitness[0] = best_val
            start_eval_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_sol = population[i].copy()
                
        # 2. Evolutionary Cycle
        while True:
            # Check Global Time
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= max_time:
                return best_val
                
            # --- Linear Population Size Reduction (LPSR) ---
            # Calculate target population size based on global elapsed time
            progress = elapsed / max_time
            target_pop = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
            target_pop = max(pop_size_min, target_pop)
            
            if pop_size > target_pop:
                # Sort by fitness and truncate the worst individuals
                sorted_indices = np.argsort(fitness)
                keep_indices = sorted_indices[:target_pop]
                
                population = population[keep_indices]
                fitness = fitness[keep_indices]
                F = F[keep_indices]
                CR = CR[keep_indices]
                pop_size = target_pop
                
            # --- Convergence Check (Trigger Restart) ---
            # If population variance is negligible, we are stuck. Restart.
            # Using standard deviation or range check.
            if np.std(fitness) < 1e-6 or (np.max(fitness) - np.min(fitness)) < 1e-6:
                break 
                
            # --- Evolution Step ---
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                
                # 1. Parameter Adaptation (jDE)
                # Generate trial F and CR
                if np.random.rand() < tau_F:
                    f_new = 0.1 + 0.9 * np.random.rand()
                else:
                    f_new = F[i]
                    
                if np.random.rand() < tau_CR:
                    cr_new = np.random.rand()
                else:
                    cr_new = CR[i]
                
                # 2. Mutation: DE/rand/1/bin
                # Select 3 distinct random individuals != i
                # Using a loop to ensure uniqueness is fast enough for small pop sizes
                idxs = []
                while len(idxs) < 3:
                    idx = np.random.randint(0, pop_size)
                    if idx != i and idx not in idxs:
                        idxs.append(idx)
                r1, r2, r3 = idxs
                
                # Mutant Vector: v = x_r1 + F * (x_r2 - x_r3)
                v = population[r1] + f_new * (population[r2] - population[r3])
                
                # 3. Boundary Handling: Bounce-Back (Reflection)
                # Reflect violations back into the bound instead of clipping
                # Lower Bound
                bad_low = v < min_b
                if np.any(bad_low):
                    v[bad_low] = min_b[bad_low] + (min_b[bad_low] - v[bad_low])
                    # Fix double bounce (if reflected value is still out)
                    v[bad_low] = np.maximum(v[bad_low], min_b[bad_low])
                    
                # Upper Bound
                bad_high = v > max_b
                if np.any(bad_high):
                    v[bad_high] = max_b[bad_high] - (v[bad_high] - max_b[bad_high])
                    # Fix double bounce
                    v[bad_high] = np.minimum(v[bad_high], max_b[bad_high])
                
                # 4. Crossover (Binomial)
                j_rand = np.random.randint(0, dim)
                mask = np.random.rand(dim) < cr_new
                mask[j_rand] = True # Ensure at least one parameter is from mutant
                
                trial = np.where(mask, v, population[i])
                
                # 5. Selection
                f_trial = func(trial)
                
                if f_trial <= fitness[i]:
                    # Trial is better or equal
                    population[i] = trial
                    fitness[i] = f_trial
                    # Adopt the successful parameters
                    F[i] = f_new
                    CR[i] = cr_new
                    
                    # Update global best
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = trial.copy()

    return best_val
