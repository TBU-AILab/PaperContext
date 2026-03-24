#The following algorithm implements a **Vectorized jDE (Self-Adaptive Differential Evolution)** enhanced with **LPSR (Linear Population Size Reduction)** and a **Current-to-pBest** mutation strategy.
#
#**Key Improvements:**
#1.  **Mutation Strategy (DE/current-to-pbest/1)**: Replaces the standard DE/rand/1 with the more advanced `current-to-pbest` strategy (found in state-of-the-art algorithms like L-SHADE). This directs the search towards the top $p\%$ of individuals, significantly speeding up convergence while maintaining diversity via the difference vector $(x_{r1} - x_{r2})$.
#2.  **Vectorized Operations**: The generation of candidate solutions, parameter adaptation, and boundary handling are fully vectorized using NumPy. This minimizes Python loop overhead, allowing for more generations within the `max_time`.
#3.  **Linear Population Size Reduction (LPSR)**: The population size is dynamically reduced from a large initial pool (for exploration) to a minimal set (for exploitation) based on the elapsed time. This ensures the algorithm focuses on refining the best solutions as the deadline approaches.
#4.  **Reflective Boundary Handling**: Uses a "bounce-back" method where particles hitting the bounds are reflected back into the search space rather than being clipped. This prevents population crowding at the edges.
#5.  **Restart with Elitism**: If the population converges (low variance), the algorithm restarts to explore new areas but carries over the global best solution to ensure non-regressive performance.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using Vectorized jDE with Current-to-pBest Mutation, 
    LPSR (Linear Population Size Reduction), and Restart Mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population Size: Adaptive based on dimension
    # Start large (~25*dim) for exploration, reduce linearly for exploitation
    pop_size_init = min(500, max(50, 25 * dim))
    pop_size_min = 4
    
    # jDE Control Parameters (Probabilities of update)
    tau_F = 0.1
    tau_CR = 0.1
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # Helper: Check for timeout
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- Main Optimization Loop (Restart Mechanism) ---
    while not check_timeout():
        
        # 1. Initialize Population for this run
        pop_size = pop_size_init
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous restarts into the new population
        start_eval_idx = 0
        if best_sol is not None:
            population[0] = best_sol.copy()
            fitness[0] = best_val
            start_eval_idx = 1
        
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_timeout(): return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = population[i].copy()
                
        # Initialize jDE Control Parameters
        # F ~ U(0.1, 0.9), CR ~ U(0.0, 1.0)
        F = 0.1 + 0.8 * np.random.rand(pop_size)
        CR = np.random.rand(pop_size)
        
        # --- Evolutionary Cycle ---
        while True:
            # Time Check
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= max_time:
                return best_val
                
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate target population size based on progress
            progress = elapsed / max_time
            target_pop = int(round(pop_size_init - (pop_size_init - pop_size_min) * progress))
            target_pop = max(pop_size_min, target_pop)
            
            if pop_size > target_pop:
                # Reduce population: Keep best individuals
                sorted_idxs = np.argsort(fitness)
                keep_idxs = sorted_idxs[:target_pop]
                
                population = population[keep_idxs]
                fitness = fitness[keep_idxs]
                F = F[keep_idxs]
                CR = CR[keep_idxs]
                pop_size = target_pop
            
            # 2. Convergence Check (Trigger Restart)
            # If population variance is negligible, restart to escape local optima
            if np.std(fitness) < 1e-8 or (np.max(fitness) - np.min(fitness)) < 1e-8:
                break
            
            # 3. Parameter Adaptation (jDE)
            # Vectorized generation of trial parameters
            # With probability tau, generate new F/CR, else keep old
            rand_F = np.random.rand(pop_size)
            rand_CR = np.random.rand(pop_size)
            
            F_trial = np.where(rand_F < tau_F, 0.1 + 0.9 * np.random.rand(pop_size), F)
            CR_trial = np.where(rand_CR < tau_CR, np.random.rand(pop_size), CR)
            
            # 4. Mutation Strategy: DE/current-to-pbest/1
            # Sort population to identify top p% individuals
            sorted_idxs = np.argsort(fitness)
            
            # p-best configuration (top 11% is a robust default)
            p_val = 0.11
            p_count = max(2, int(pop_size * p_val))
            
            # For each individual 'i', pick a random 'pbest' from the top p%
            pbest_rand_idxs = np.random.randint(0, p_count, pop_size)
            pbest_idxs = sorted_idxs[pbest_rand_idxs]
            
            # Pick r1, r2 distinct from i (and each other)
            # Strategy: Use shifting to avoid collision checks in loops
            idxs = np.arange(pop_size)
            
            # r1: distinct from i
            # Add random offset [1, pop_size-1] modulo pop_size
            offset1 = np.random.randint(1, pop_size, pop_size)
            r1 = (idxs + offset1) % pop_size
            
            # r2: distinct from i and r1
            # Generate candidates, then iteratively fix collisions (usually 0-2 iterations)
            r2 = np.random.randint(0, pop_size, pop_size)
            for _ in range(3):
                mask_collision = (r2 == idxs) | (r2 == r1)
                if not np.any(mask_collision):
                    break
                r2[mask_collision] = (r2[mask_collision] + 1) % pop_size
            
            # Compute Mutant Vectors: V = X_i + F*(X_pbest - X_i) + F*(X_r1 - X_r2)
            F_broad = F_trial[:, None] # Broadcast F for dimensions
            
            x_i = population
            x_pbest = population[pbest_idxs]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            diff_pbest = x_pbest - x_i
            diff_r = x_r1 - x_r2
            
            mutant = x_i + F_broad * diff_pbest + F_broad * diff_r
            
            # 5. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask = np.random.rand(pop_size, dim) < CR_trial[:, None]
            # Ensure at least one parameter is inherited from mutant
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, x_i)
            
            # 6. Bound Handling (Bounce-Back / Reflection)
            # If a value exceeds bounds, reflect it back instead of clipping
            # Lower bounds
            viol_l = trial_pop < min_b
            if np.any(viol_l):
                # Reflect: min + (min - v) = 2*min - v
                trial_pop[viol_l] = 2 * min_b[np.where(viol_l)[1]] - trial_pop[viol_l]
                # If still out (double bounce), clip to bound
                viol_l_2 = trial_pop < min_b
                trial_pop[viol_l_2] = min_b[np.where(viol_l_2)[1]]
                
            # Upper bounds
            viol_u = trial_pop > max_b
            if np.any(viol_u):
                # Reflect: max - (v - max) = 2*max - v
                trial_pop[viol_u] = 2 * max_b[np.where(viol_u)[1]] - trial_pop[viol_u]
                # If still out, clip to bound
                viol_u_2 = trial_pop > max_b
                trial_pop[viol_u_2] = max_b[np.where(viol_u_2)[1]]
            
            # 7. Selection (Greedy)
            # Evaluate trial vectors
            for i in range(pop_size):
                if check_timeout(): return best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    # Trial is better or equal: Replace parent
                    population[i] = trial_pop[i]
                    fitness[i] = f_trial
                    F[i] = F_trial[i]
                    CR[i] = CR_trial[i]
                    
                    # Update Global Best
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = trial_pop[i].copy()
                        
    return best_val
