#To improve upon the previous algorithms, I propose an implementation of **Self-Adaptive Differential Evolution (JADE-like variant)** with a **Restart Mechanism**.
#
#**Key Improvements:**
#1.  **Adaptive Parameters ($F$ and $CR$)**: Instead of fixed values (which caused the previous algorithm to be either too slow or get stuck), this algorithm adapts the Mutation Factor ($F$) and Crossover Rate ($CR$) based on the success of previous generations. It learns whether the function landscape requires large jumps or fine-tuning.
#2.  **Current-to-pBest Strategy**: It uses the `current-to-pbest/1/bin` mutation strategy. This balances exploration (randomness) and exploitation (guiding towards the top 5% best solutions), converging faster than random search while maintaining diversity better than greedy strategies.
#3.  **Automatic Restarts**: If the population converges (variance becomes near-zero) or stagnates (no improvement for many generations), the algorithm triggers a restart to explore new basins of attraction within the remaining time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Establish strict timing to ensure we return within bounds
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # -----------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------
    # Population size: 15*dim is a robust starting point for DE
    pop_size = max(20, 15 * dim)
    
    # Adaptation parameters (JADE logic)
    mu_cr = 0.5    # Mean Crossover Rate
    mu_f = 0.5     # Mean Mutation Factor
    c_adapt = 0.1  # Learning rate for parameter updates
    p_best_rate = 0.05 # Top percentage for 'p-best' selection
    
    # Pre-process bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    bound_diff = ub - lb
    
    # Track the global best solution found across all restarts
    global_best_val = float('inf')
    
    # -----------------------------------------------------------
    # Main Optimization Loop (Restart Loop)
    # -----------------------------------------------------------
    while True:
        # Check time before starting a new population restart
        if datetime.now() >= end_time:
            return global_best_val
            
        # --- Initialization Phase ---
        pop = lb + np.random.rand(pop_size, dim) * bound_diff
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if datetime.now() >= end_time: return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Track stagnation for this restart
        last_best_fit = np.min(fitness)
        stagnation_count = 0
        
        # --- Evolution Phase ---
        while True:
            # Time check per generation
            if datetime.now() >= end_time:
                return global_best_val

            # 1. Parameter Adaptation Generation
            # CR ~ Normal(mu_cr, 0.1), clipped to [0, 1]
            cr_i = np.random.normal(mu_cr, 0.1, pop_size)
            cr_i = np.clip(cr_i, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1). 
            # Approx using standard_cauchy: F = mu + 0.1 * tan(pi*(rand-0.5)) or standard func
            f_i = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            
            # Robust clamping for F
            f_i = np.where(f_i >= 1.0, 1.0, f_i)
            # If F is too small (<=0), reset to a conservative default (e.g., 0.4) to avoid stagnation
            f_i = np.where(f_i <= 0.0, 0.4, f_i)
            
            # 2. Mutation: current-to-pbest/1/bin
            # Select p-best indices (top p% individuals)
            sorted_idx = np.argsort(fitness)
            p_limit = max(1, int(p_best_rate * pop_size))
            top_p_indices = sorted_idx[:p_limit]
            pbest_indices = np.random.choice(top_p_indices, pop_size)
            
            # Select r1, r2 random indices
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Retrieve vectors
            x = pop
            x_pbest = pop[pbest_indices]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Calculate Mutation Vectors V = X + F*(Xpbest - X) + F*(Xr1 - Xr2)
            F_col = f_i.reshape(-1, 1) # Reshape for broadcasting
            v = x + F_col * (x_pbest - x) + F_col * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            # Create mask where rand < CR
            mask = np.random.rand(pop_size, dim) < cr_i.reshape(-1, 1)
            # Ensure at least one dimension is mutated
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            # Create trial vectors
            u = np.where(mask, v, x)
            
            # 4. Boundary Constraint
            u = np.clip(u, lb, ub)
            
            # 5. Selection & Adaptation Update
            successful_cr = []
            successful_f = []
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_val
                
                new_val = func(u[i])
                
                # Greedy Selection
                if new_val < fitness[i]:
                    fitness[i] = new_val
                    pop[i] = u[i]
                    
                    # Store successful parameters
                    successful_cr.append(cr_i[i])
                    successful_f.append(f_i[i])
                    
                    if new_val < global_best_val:
                        global_best_val = new_val
            
            # Update adaptive means (mu_cr, mu_f) if there were successes
            if successful_cr:
                mean_scr = np.mean(successful_cr)
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * mean_scr
                
                sf = np.array(successful_f)
                # Lehmer Mean for F: sum(F^2) / sum(F)
                denom = np.sum(sf)
                if denom == 0: denom = 1e-10
                mean_lehmer = np.sum(sf**2) / denom
                mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer
            
            # 6. Convergence / Restart Check
            curr_best_fit = np.min(fitness)
            std_dev = np.std(fitness)
            
            # Restart if population has converged (variance is tiny)
            if std_dev < 1e-6:
                break
            
            # Restart if stuck in local minima (no improvement for many generations)
            if abs(curr_best_fit - last_best_fit) < 1e-8:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = curr_best_fit
            
            if stagnation_count > 40:
                break
                
    return global_best_val
