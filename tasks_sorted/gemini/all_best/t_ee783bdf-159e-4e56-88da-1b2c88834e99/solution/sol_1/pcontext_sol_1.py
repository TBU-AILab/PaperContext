#The output value of the best generated algorithm is: 123.83193558020393. The previous algorithm used a standard Differential Evolution approach. To improve this, I propose an **Enhanced Adaptive Differential Evolution** algorithm. This version introduces three key improvements:
#1.  **Current-to-Best Strategy**: Instead of random mutation, vectors are guided towards the best solution found so far (`DE/current-to-best/1`), which significantly speeds up convergence.
#2.  **Adaptive Parameters**: The mutation factor `F` and crossover probability `CR` are randomized per individual (dithered) to prevent stagnation and adapt to different fitness landscape scales.
#3.  **Local Polishing**: When a new global best is found, a lightweight local search (stochastic hill climbing) is immediately triggered to refine the solution, exploiting the local basin of attraction.
#4.  **Restart Mechanism**: If the population fitness converges (low standard deviation), the algorithm triggers a soft restart to escape local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using an Enhanced Adaptive Differential Evolution strategy.
    Features: DE/current-to-best/1 strategy, parameter dithering, restart on stagnation,
    and opportunistic local search (polishing).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: trade-off between diversity and speed. 
    # Scaled by dimension but bounded to fit within time limits.
    pop_size = int(np.clip(15 * dim, 20, 80))
    
    # Parse bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # Uniform random initialization within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_idx = -1
    best_fit = float('inf')
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            # Return best found so far if time runs out during initialization
            return best_fit
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_idx = i
            
    # Handle edge case where no valid solution was found
    if best_idx == -1: return float('inf')

    # --- Main Optimization Loop ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Stagnation Check (Restart Mechanism)
        # If the population has converged (low variance), we might be stuck in a local optimum.
        # We keep the best solution and re-initialize the rest to search elsewhere.
        if pop_size > 5:
            fit_std = np.std(fitness)
            # Threshold for stagnation (small relative to problem scale or absolute zero)
            if fit_std < 1e-6:
                for k in range(pop_size):
                    if k == best_idx: continue
                    
                    # Randomize position
                    pop[k] = min_b + np.random.rand(dim) * diff_b
                    
                    # Evaluate immediately
                    if (datetime.now() - start_time) >= time_limit: return best_fit
                    val = func(pop[k])
                    fitness[k] = val
                    
                    if val < best_fit:
                        best_fit = val
                        best_idx = k
                continue # Skip to next generation

        # 2. Differential Evolution Step
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fit

            # --- Mutation: DE/current-to-best/1 ---
            # V = X_i + F * (X_best - X_i) + F * (X_r1 - X_r2)
            # This balances exploitation (move to best) and exploration (difference vector).
            
            # Select r1, r2 distinct from i
            while True:
                r1 = np.random.randint(pop_size)
                if r1 != i: break
            while True:
                r2 = np.random.randint(pop_size)
                if r2 != i and r2 != r1: break
            
            # Adaptive Parameters (Dithering)
            # Randomizing F and CR helps avoid getting stuck with bad fixed parameters
            F = np.random.uniform(0.5, 1.0)   # Scale factor
            CR = np.random.uniform(0.8, 1.0)  # Crossover probability (prefer high for DE)
            
            x_i = pop[i]
            x_best = pop[best_idx]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            cross_mask = np.random.rand(dim) < CR
            # Ensure at least one dimension is mutated
            cross_mask[np.random.randint(dim)] = True 
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # --- Bound Handling: Clip ---
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update Global Best
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_idx = i
                    
                    # 3. Local Polishing (Memetic Step)
                    # When a new global best is found, greedily refine it with small steps.
                    # This exploits the gradient-free "descent" logic.
                    curr_p = pop[i].copy()
                    curr_f = best_fit
                    
                    # Perform up to 5 steps of local hill climbing
                    for _ in range(5):
                        if (datetime.now() - start_time) >= time_limit: return best_fit
                        
                        # Perturbation: small gaussian noise (0.5% of domain range)
                        step = np.random.normal(0, 0.005, dim) * diff_b
                        cand = np.clip(curr_p + step, min_b, max_b)
                        f_cand = func(cand)
                        
                        if f_cand < curr_f:
                            # Improved: update and continue
                            curr_f = f_cand
                            curr_p = cand
                            
                            # Commit to population
                            pop[i] = curr_p
                            fitness[i] = curr_f
                            best_fit = curr_f
                            best_idx = i
                        else:
                            # No improvement: stop polishing
                            break

    return best_fit
