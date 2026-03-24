#The algorithm below implements a **Vectorized Self-Adaptive Differential Evolution (jDE) with Elitist Restarts**.
#
#**Key Improvements:**
#1.  **Self-Adaptation (jDE)**: Instead of using fixed parameters or randomizing them blindly (dithering), this algorithm evolves the Mutation Factor ($F$) and Crossover Rate ($CR$) along with the population. Successful individuals pass their parameter values to the next generation, while unsuccessful parameters are discarded. This allows the algorithm to "learn" the landscape properties (e.g., whether to explore or exploit).
#2.  **Elitist Restarts**: To avoid getting stuck in local optima, the algorithm restarts the population when diversity (fitness standard deviation) drops. Uniquely, it **injects the global best solution** found so far into the new random population. This ensures previous progress is not lost and helps guide the new search.
#3.  **Vectorized Operations**: Following the success of the best-performing previous algorithm, this version strictly uses NumPy vectorization for mutation, crossover, and parameter updates to maximize the number of generations within the time limit.
#4.  **Optimized Population Sizing**: Uses a dynamic population size clamped between 20 and 50. This balance was observed to yield better results than larger populations (too slow) or smaller ones (stagnation) in the provided examples.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Self-Adaptive Differential Evolution (jDE)
    with Elitist Restarts.
    """
    # Initialize timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size: 
    # Scaled with dimension but clamped to [20, 50].
    # This range balances diversity with the speed of updates (iterations/sec).
    pop_size = int(np.clip(dim * 10, 20, 50))
    
    # jDE Control Parameters (probability of parameter update)
    tau_F = 0.1
    tau_CR = 0.1
    
    # Restart tolerance (if std dev of fitness drops below this, restart)
    restart_tol = 1e-6

    # Pre-process bounds for efficient broadcasting
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Global best tracking
    global_best_val = float('inf')
    global_best_pos = None

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Time check before expensive initialization
        if datetime.now() >= end_time:
            return global_best_val

        # 1. Initialize Population
        # Random initialization within bounds
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Initialize Adaptive Parameters
        # F starts at 0.5, CR at 0.9 (good default DE settings)
        pop_F = np.full(pop_size, 0.5)
        pop_CR = np.full(pop_size, 0.9)
        
        # Elitist Restart:
        # If we have a previous best, inject it into the new population at index 0.
        # This preserves the best found solution (exploitation) while the rest 
        # of the population explores.
        if global_best_pos is not None:
            pop[0] = global_best_pos.copy()
        
        fitness = np.full(pop_size, float('inf'))

        # Evaluate Initial Population
        # We start loop from 0 or 1 depending on if we injected a known solution
        start_eval_idx = 0
        if global_best_pos is not None:
            fitness[0] = global_best_val
            start_eval_idx = 1
        
        for i in range(start_eval_idx, pop_size):
            if datetime.now() >= end_time: return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_pos = pop[i].copy()

        # 2. Evolutionary Cycle
        while True:
            # Strict Time Check
            if datetime.now() >= end_time: return global_best_val
            
            # Convergence Check (Restart Trigger)
            # If population fitness variance is extremely low, we are stuck.
            if np.std(fitness) < restart_tol:
                break # Break inner loop to restart outer loop

            # --- jDE Parameter Adaptation (Vectorized) ---
            # Generate random masks to decide which individuals update their F/CR
            rand_F = np.random.rand(pop_size)
            rand_CR = np.random.rand(pop_size)
            
            mask_F = rand_F < tau_F
            mask_CR = rand_CR < tau_CR
            
            # Create trial parameter arrays based on current ones
            trial_F = pop_F.copy()
            trial_CR = pop_CR.copy()
            
            # Update F: 0.1 + 0.9 * rand()
            # This allows F to take values in [0.1, 1.0]
            if np.any(mask_F):
                trial_F[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
                
            # Update CR: rand()
            # This allows CR to take values in [0.0, 1.0]
            if np.any(mask_CR):
                trial_CR[mask_CR] = np.random.rand(np.sum(mask_CR))
            
            # --- Vectorized Mutation (DE/rand/1) ---
            # Generate random indices r1, r2, r3
            idxs = np.arange(pop_size)
            r1 = np.random.permutation(idxs)
            r2 = np.random.permutation(idxs)
            r3 = np.random.permutation(idxs)
            
            # Calculate Mutant Vector: V = X_r1 + F_trial * (X_r2 - X_r3)
            # Reshape F for broadcasting
            F_col = trial_F[:, np.newaxis]
            mutant = pop[r1] + F_col * (pop[r2] - pop[r3])
            
            # Bound Constraint Handling (Clipping)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Vectorized Crossover (Binomial) ---
            # Generate mask based on CR
            CR_col = trial_CR[:, np.newaxis]
            cross_mask = np.random.rand(pop_size, dim) < CR_col
            
            # Forced Gene: Ensure at least one dimension is taken from the mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[idxs, j_rand] = True
            
            # Create Trial Population
            trial_pop = np.where(cross_mask, mutant, pop)
            
            # --- Selection (Sequential Evaluation) ---
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # jDE Logic: Adoption
                    # If trial was successful, updated parameters are kept.
                    pop_F[i] = trial_F[i]
                    pop_CR[i] = trial_CR[i]
                    
                    # Update Global Best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_pos = trial_pop[i].copy()
                # If trial failed, we implicitly discard trial_pop[i] and
                # trial parameters (pop_F[i] remains old value).

    return global_best_val
