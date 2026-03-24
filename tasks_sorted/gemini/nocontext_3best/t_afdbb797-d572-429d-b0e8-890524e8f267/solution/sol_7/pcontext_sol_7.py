#The following Python code implements a **Self-Adaptive Ensemble Differential Evolution with Restart (SaDE-R)**.
#
#**Key Improvements over previous algorithms:**
#1.  **Ensemble Mutation Strategy**: Instead of relying on a single strategy (like `rand/1` in jDE or `current-to-pbest` in JADE), this algorithm concurrently uses both **`DE/rand/1`** (strong exploration) and **`DE/current-to-best/1`** (strong exploitation). It dynamically adapts the probability of choosing each strategy based on their recent success rates. This allows the algorithm to behave like a robust explorer in the early stages or on multimodal landscapes, and a fast converger on unimodal slopes.
#2.  **Dual Adaptation Layers**: 
#    *   **Strategy Level**: Adapts the mix of mutation types (`rand` vs `best`).
#    *   **Parameter Level**: Uses the lightweight **jDE** mechanism to self-adapt $F$ and $Cr$ for each individual.
#3.  **Stagnation-Based Restart**: In addition to variance-based convergence detection, this implementation tracks the "generations since last improvement". If the best solution in the current run doesn't improve for a fixed number of generations (stagnation), it triggers a restart. This is crucial for escaping wide, flat local optima where variance might remain high despite no progress.
#4.  **Reflective Bound Handling**: Ensures particles bounce back into the valid search space, preserving diversity better than simple clipping.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using Self-Adaptive Ensemble Differential Evolution 
    (SaDE) with Restart Mechanism.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Sufficiently large to support strategy learning, 
    # but small enough for fast iterations.
    pop_size = int(np.clip(10 * dim, 30, 80))
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking across all restarts
    global_best_val = float('inf')
    global_best_sol = None
    
    # --- Helper: Time Check ---
    def is_time_up():
        return (datetime.now() - start_time) >= limit

    # --- Main Loop (Restarts) ---
    while not is_time_up():
        
        # 1. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the global best solution into the new population (Soft Restart)
        start_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_val
            start_idx = 1
            
        # Evaluate initial population
        for i in range(start_idx, pop_size):
            if is_time_up(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_sol = pop[i].copy()
        
        # 2. Initialize Adaptive Parameters (jDE logic)
        F = np.random.uniform(0.1, 1.0, pop_size)
        CR = np.random.uniform(0.0, 1.0, pop_size)
        
        # 3. Strategy Adaptation State
        # Strategies: 0 -> DE/rand/1, 1 -> DE/current-to-best/1
        # strat_prob is the probability of selecting Strategy 0
        strat_prob = 0.5 
        strat_success = np.array([1.0, 1.0]) # Additive smoothing
        learning_rate = 0.05
        
        # Stagnation tracking for this run
        best_in_run = np.min(fitness)
        stag_count = 0
        stag_limit = 50  # Max generations without improvement before restart
        
        # --- Evolution Loop ---
        while not is_time_up():
            
            # --- A. Strategy Assignment ---
            # Randomly assign strategy 0 or 1 based on strat_prob
            strat_mask = np.random.rand(pop_size) < strat_prob
            # Ensure diversity in strategies for learning
            strat_mask[0] = True
            strat_mask[1] = False
            
            # --- B. Parameter Adaptation (jDE) ---
            # Update F and CR with probabilities tau1, tau2
            tau_f = 0.1
            tau_cr = 0.1
            
            rand_f = np.random.rand(pop_size)
            rand_cr = np.random.rand(pop_size)
            
            # F update: uniform [0.1, 1.0]
            mask_f = rand_f < tau_f
            if np.any(mask_f):
                F[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
                
            # CR update: uniform [0.0, 1.0]
            mask_cr = rand_cr < tau_cr
            if np.any(mask_cr):
                CR[mask_cr] = np.random.rand(np.sum(mask_cr))
            
            # --- C. Mutation Vector Generation ---
            # We vectorize the generation of both strategies efficiently
            
            # Index selection: distinct r1, r2, r3
            idxs = np.arange(pop_size)
            
            # Select r1 != i
            r1 = np.random.randint(0, pop_size, pop_size)
            conflict = (r1 == idxs)
            while np.any(conflict):
                r1[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
                conflict = (r1 == idxs)
            
            # Select r2 != i, r1
            r2 = np.random.randint(0, pop_size, pop_size)
            conflict = (r2 == idxs) | (r2 == r1)
            while np.any(conflict):
                r2[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
                conflict = (r2 == idxs) | (r2 == r1)
                
            # Select r3 != i, r1, r2 (needed for rand/1)
            r3 = np.random.randint(0, pop_size, pop_size)
            conflict = (r3 == idxs) | (r3 == r1) | (r3 == r2)
            while np.any(conflict):
                r3[conflict] = np.random.randint(0, pop_size, np.sum(conflict))
                conflict = (r3 == idxs) | (r3 == r1) | (r3 == r2)
            
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            x_r3 = pop[r3]
            
            # Identify current best in population (for current-to-best)
            idx_best = np.argmin(fitness)
            x_best = pop[idx_best]
            
            F_col = F[:, np.newaxis]
            
            # Strategy 0: DE/rand/1
            # v = x_r1 + F * (x_r2 - x_r3)
            mutant_0 = x_r1 + F_col * (x_r2 - x_r3)
            
            # Strategy 1: DE/current-to-best/1
            # v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            mutant_1 = pop + F_col * (x_best - pop) + F_col * (x_r1 - x_r2)
            
            # Combine based on strategy mask
            mutant = np.where(strat_mask[:, np.newaxis], mutant_0, mutant_1)
            
            # --- D. Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < CR[:, np.newaxis]
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[idxs, j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # --- E. Bound Handling (Reflection) ---
            # Lower bounds
            mask_l = trial < min_b
            if np.any(mask_l):
                trial[mask_l] = 2 * min_b[np.where(mask_l)[1]] - trial[mask_l]
                trial = np.maximum(trial, min_b)
            
            # Upper bounds
            mask_u = trial > max_b
            if np.any(mask_u):
                trial[mask_u] = 2 * max_b[np.where(mask_u)[1]] - trial[mask_u]
                trial = np.minimum(trial, max_b)
                
            # --- F. Selection & Adaptation ---
            wins_0 = 0
            wins_1 = 0
            
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    # Trial improved or equal
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    # Record success for strategy update
                    if strat_mask[i]:
                        wins_0 += 1
                    else:
                        wins_1 += 1
                        
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_sol = trial[i].copy()
            
            # Update Strategy Probabilities (Adaptive)
            # Accumulate success counts with decay (moving average)
            strat_success[0] = (1 - learning_rate) * strat_success[0] + learning_rate * wins_0
            strat_success[1] = (1 - learning_rate) * strat_success[1] + learning_rate * wins_1
            
            # Normalize to get probability for Strategy 0
            total_wins = strat_success[0] + strat_success[1]
            if total_wins > 0:
                # Clip to [0.05, 0.95] to ensure neither strategy dies out
                strat_prob = np.clip(strat_success[0] / total_wins, 0.05, 0.95)
            
            # --- G. Convergence / Restart Check ---
            # 1. Stagnation Check
            current_min = np.min(fitness)
            if current_min < best_in_run:
                best_in_run = current_min
                stag_count = 0
            else:
                stag_count += 1
                
            # 2. Variance Check
            std_dev = np.std(fitness)
            
            # Restart if variance is tiny (converged) OR stagnation limit reached
            if std_dev < 1e-8 or stag_count > stag_limit:
                break # Breaks inner loop to trigger restart

    return global_best_val
