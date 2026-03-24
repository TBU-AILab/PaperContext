#Here is an improved algorithm. It implements **JADE (Adaptive Differential Evolution with Optional External Archive)** concepts, simplified for a single-file implementation.
#
#**Key Improvements over the previous algorithm:**
#1.  **Adaptive Parameters:** Instead of fixed Mutation ($F$) and Crossover ($CR$) factors, this algorithm adapts them based on the success history of the population. This allows it to switch between exploration (searching new areas) and exploitation (refining the best solution) automatically.
#2.  **Current-to-pBest Mutation:** It uses a specific mutation strategy that moves individuals towards the top $p\%$ of best solutions found so far, rather than just random ones. This significantly speeds up convergence.
#3.  **Vectorized Operations:** The mutation and crossover steps are vectorized using NumPy, reducing the overhead of Python loops and allowing more generations within the `max_time`.
#4.  **Restart Mechanism:** If the population converges (variance becomes too low) before time runs out, it triggers a restart to search different areas of the solution space while preserving the global best.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using a JADE-inspired Self-Adaptive Differential Evolution.
    
    Mechanisms:
    - 'current-to-pbest/1' mutation strategy for faster convergence.
    - Adaptive F and CR parameters based on successful mutations.
    - Vectorized operations for speed.
    - Population restart upon stagnation.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Bounds processing
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Population size: 
    # slightly larger than standard DE to utilize vectorization benefits
    # but capped to ensure enough generations run.
    pop_size = int(np.clip(15 * dim, 30, 100))
    
    # JADE Adaptive Parameters
    mu_cr = 0.5  # Initial mean for Crossover Probability
    mu_f = 0.5   # Initial mean for Mutation Factor
    c_adapt = 0.1 # Learning rate for parameter adaptation
    p_share = 0.05 # Top 5% individuals used for 'pbest'
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fit = float('inf')
    best_sol = None

    # Initial Evaluation
    # We evaluate in a loop because 'func' might not be vectorized
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_fit if best_fit != float('inf') else float('inf') # fallback
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_sol = pop[i].copy()

    # --- Main Loop ---
    while True:
        # Check time at the start of generation
        if time.time() - start_time >= max_time:
            return best_fit

        # 1. Sort population to easily find 'pbest' (top p%)
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Best is now at index 0 due to sort
        current_best = fitness[0]
        if current_best < best_fit:
            best_fit = current_best
            best_sol = pop[0].copy()

        # 2. Check for Stagnation / Restart
        # If population diversity (std dev of fitness) is extremely low, restart
        # keeping only the best solution.
        if np.std(fitness) < 1e-8:
            # Re-initialize population scattered across bounds
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol # Keep the champion
            
            # Re-evaluate new population (except 0 which is known)
            fitness[0] = best_fit
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time: return best_fit
                fitness[i] = func(pop[i])
            
            # Reset adaptive parameters
            mu_cr, mu_f = 0.5, 0.5
            continue # Start new generation

        # 3. Generate Parameter Values for this Generation (Vectorized)
        # CR_i ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr_i = np.random.normal(mu_cr, 0.1, pop_size)
        cr_i = np.clip(cr_i, 0, 1)
        
        # F_i ~ Cauchy(mu_f, 0.1). 
        # Approx Cauchy using Normal or Tan for simplicity in standard numpy
        # F_i = mu_f + 0.1 * tan(pi * (rand - 0.5))
        # Here we use clipped Normal for stability and speed, 
        # ensuring F > 0 to force mutation.
        f_i = np.random.normal(mu_f, 0.1, pop_size)
        f_i = np.clip(f_i, 0.1, 1.0) # F must be positive

        # 4. Mutation: DE/current-to-pbest/1
        # V_i = X_i + F_i * (X_pbest - X_i) + F_i * (X_r1 - X_r2)
        
        # Select pbest indices: random integer from top p_share %
        top_p_count = max(1, int(pop_size * p_share))
        pbest_idx = np.random.randint(0, top_p_count, pop_size)
        
        # Select r1, r2 indices: random distinct integers
        # (Simplified: random collision is rare/acceptable in vectorization for speed)
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        x_pbest = pop[pbest_idx]
        x_r1 = pop[r1]
        x_r2 = pop[r2]
        
        # Calculate Mutant Vectors (Vectorized)
        # Reshape f_i for broadcasting (pop_size, 1)
        f_col = f_i[:, None]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 5. Crossover
        # Mask: True where crossover happens
        mask = np.random.rand(pop_size, dim) < cr_i[:, None]
        
        # Ensure at least one parameter is changed for every individual (j_rand)
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutant, pop)
        
        # Bound Constraints (Clip)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 6. Selection & Adaptation
        successful_cr = []
        successful_f = []
        
        # We must iterate for evaluation as func is likely scalar
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_fit
                
            f_trial = func(trial_pop[i])
            
            if f_trial < fitness[i]:
                # Improvement found
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                # Record successful parameters
                successful_cr.append(cr_i[i])
                successful_f.append(f_i[i])
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_sol = trial_pop[i].copy()
                    
        # 7. Update Adaptive Parameters
        if len(successful_cr) > 0:
            # Update mu_cr (Weighted Arithmetic Mean logic simplified)
            mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(successful_cr)
            
            # Update mu_f (Lehmer Mean: sum(f^2) / sum(f))
            # Promotes larger F values slightly to avoid premature convergence
            sf = np.array(successful_f)
            mean_lehmer = np.sum(sf**2) / np.sum(sf)
            mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer

    return best_fit
