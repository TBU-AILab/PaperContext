#The output values indicate that **Algorithm 1 (Partial Restart + OBL)** achieved the best performance (~10.53), significantly outperforming Algorithm 2 (Gaussian LS, ~57.8) and Algorithm 6 (Current-to-best, ~49.3).
#
#**Analysis of Results:**
#*   **Algorithm 1** succeeded because it balanced exploration (DE/rand/1) with a robust restart mechanism that preserved elites.
#*   **Algorithm 6** (Current-to-best) likely failed because it converged too aggressively to local optima without sufficient exploration first.
#*   **Algorithm 2** failed because Gaussian Local Search was likely too computationally expensive or disruptive for the limited time budget.
#
#**Proposed Improvements:**
#To improve upon the best result (10.53), I propose a **Hybrid DE with Quasi-Opposition Based Learning (QOBL) and Dynamic Strategy Switching**.
#
#1.  **Quasi-Opposition Based Learning (QOBL)**: Instead of standard OBL (checking the opposite point $min+max-x$), QOBL checks a random point *between* the center of the search space and the opposite point. This is mathematically proven to be closer to the solution on average for black-box problems.
#2.  **Dynamic Strategy Switching**: Instead of sticking to one mutation strategy, the algorithm monitors stagnation.
#    *   **Phase 1 (Exploration)**: Uses `DE/rand/1` to search broadly.
#    *   **Phase 2 (Exploitation)**: If stagnation is detected (but not enough to restart), it switches to `DE/current-to-best/1`. This attempts to "snap" to the minimum of the current basin efficiently.
#    *   **Phase 3 (Restart)**: If stagnation continues, it triggers the **Partial Restart** (keeping elites and refilling with QOBL) to jump to a new basin.
#3.  **Reflective Boundaries**: Maintained from the best algorithm to preserve edge diversity.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Hybrid Differential Evolution with 
    Quasi-Opposition Based Learning (QOBL) and Dynamic Strategy Switching.
    
    This algorithm improves upon standard DE and simple restart strategies by:
    1. Using Quasi-Opposition Based Learning for initialization and restarts to 
       scan the search space more effectively than random or standard OBL.
    2. Switching from 'DE/rand/1' (Exploration) to 'DE/current-to-best/1' (Exploitation)
       when stagnation is detected, attempting to converge before giving up.
    3. Employing a Partial Restart mechanism (Elitism + QOBL Refill) to escape 
       local optima while preserving found genetic gains.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 
    # Scaled to dimension but capped to ensure sufficient generations within time limit.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # DE Parameters
    CR = 0.9
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    center_b = (min_b + max_b) / 2.0
    
    # Global Best Tracking
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # --- Helper: Time Check ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: QOBL Generation ---
    # Generates a pool of candidates using Quasi-Opposition Based Learning.
    # Logic: Generates Random points (P), calculates Opposite points (O), 
    # then samples Quasi-Opposite points (Q) between Center and O.
    # Returns P combined with Q.
    def generate_qobl_pool(needed=pop_size):
        # 1. Random Candidates
        p_rand = min_b + np.random.rand(needed, dim) * diff_b
        
        # 2. Opposite Candidates
        # x_opp = min + max - x
        p_opp = min_b + max_b - p_rand
        
        # 3. Quasi-Opposite Candidates
        # Randomly sample between Center and Opposite
        # x_qo = rand(center, x_opp)
        p_qo = np.zeros_like(p_rand)
        for i in range(needed):
            # Vectorized min/max for the uniform range logic
            lows = np.minimum(center_b, p_opp[i])
            highs = np.maximum(center_b, p_opp[i])
            p_qo[i] = np.random.uniform(lows, highs)
            
        # Clip to ensure bounds
        p_qo = np.clip(p_qo, min_b, max_b)
        
        # Combine Random and Quasi-Opposite
        return np.vstack((p_rand, p_qo))

    # --- Helper: Evaluate and Select ---
    # Evaluates a pool of candidates and returns the best 'count' individuals
    def evaluate_and_select(candidates, count):
        nonlocal global_best_fitness, global_best_sol
        
        cand_fit = []
        valid_cand = []
        
        for i in range(len(candidates)):
            if check_time(): break
            val = func(candidates[i])
            cand_fit.append(val)
            valid_cand.append(candidates[i])
            
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_sol = candidates[i].copy()
                
        if not cand_fit: 
            return np.array([]), np.array([])
        
        cand_fit = np.array(cand_fit)
        valid_cand = np.array(valid_cand)
        
        # Sort indices
        idx = np.argsort(cand_fit)
        
        # Select top 'count'
        # (Handle case where time ran out and we have fewer than count)
        n = min(count, len(idx))
        return valid_cand[idx[:n]], cand_fit[idx[:n]]

    # --- Initialization ---
    initial_pool = generate_qobl_pool(needed=pop_size)
    population, fitness = evaluate_and_select(initial_pool, pop_size)
    
    if len(population) == 0: return global_best_fitness # Immediate timeout
    
    # Run Statistics
    stagnation_counter = 0
    best_in_run = np.min(fitness)
    
    # --- Main Optimization Loop ---
    while True:
        if check_time(): return global_best_fitness
        
        # 1. Stagnation Check & Strategy Update
        current_best = np.min(fitness)
        if current_best < best_in_run - 1e-8:
            best_in_run = current_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # 2. Restart Mechanism
        # If stagnant for too long (e.g., 30 generations), trigger Partial Restart
        if stagnation_counter > 30:
            # Keep Elites (Top 20%)
            elite_count = max(1, int(pop_size * 0.2))
            sort_idx = np.argsort(fitness)
            elites = population[sort_idx[:elite_count]]
            elite_fit = fitness[sort_idx[:elite_count]]
            
            # Refill with QOBL
            needed = pop_size - elite_count
            new_pool = generate_qobl_pool(needed=needed)
            new_pop, new_fit = evaluate_and_select(new_pool, needed)
            
            if len(new_pop) > 0:
                population = np.vstack((elites, new_pop))
                fitness = np.concatenate((elite_fit, new_fit))
            
            # Reset counters
            stagnation_counter = 0
            best_in_run = np.min(fitness) if len(fitness) > 0 else float('inf')
            continue # Start fresh generation immediately
            
        # 3. Dynamic Strategy Selection
        # If moderate stagnation (15-30 gens), switch to Exploitation (Current-to-Best)
        # to try and snap to the minimum. Otherwise use Exploration (Rand/1).
        use_exploitation = (stagnation_counter > 15)
        
        # 4. Evolution Cycle
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Identify best for current-to-best strategy
        idx_best = np.argmin(fitness)
        x_best = population[idx_best]
        
        for i in range(len(population)):
            if check_time(): return global_best_fitness
            
            x_i = population[i]
            
            # Select indices distinct from i
            idxs = [x for x in range(len(population)) if x != i]
            if len(idxs) < 3: break
            
            if use_exploitation:
                # Strategy: DE/current-to-best/1/bin
                # Aggressive convergence attempts
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                x_r1, x_r2 = population[r1], population[r2]
                
                # F dithered [0.5, 0.9] for stability
                F = 0.5 + 0.4 * np.random.rand()
                
                v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            else:
                # Strategy: DE/rand/1/bin
                # Broad exploration
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
                
                # F dithered [0.3, 0.7] for higher variance
                F = 0.3 + 0.4 * np.random.rand()
                
                v = x_r1 + F * (x_r2 - x_r3)
                
            # Crossover (Binomial)
            mask = np.random.rand(dim) < CR
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            u = np.where(mask, v, x_i)
            
            # Boundary Handling: Reflection
            # Bounces particles back to preserve distribution at edges
            under_b = u < min_b
            if np.any(under_b):
                u[under_b] = 2 * min_b[under_b] - u[under_b]
            over_b = u > max_b
            if np.any(over_b):
                u[over_b] = 2 * max_b[over_b] - u[over_b]
            # Safety clip
            u = np.clip(u, min_b, max_b)
            
            # Selection
            f_u = func(u)
            
            if f_u <= fitness[i]:
                new_population[i] = u
                new_fitness[i] = f_u
                
                if f_u < global_best_fitness:
                    global_best_fitness = f_u
                    global_best_sol = u.copy()
                    
                    # Opportunistic update of x_best for current generation
                    # if we found something better during exploitation phase
                    if use_exploitation and f_u < fitness[idx_best]:
                        x_best = u
                        idx_best = i

        population = new_population
        fitness = new_fitness

    return global_best_fitness
