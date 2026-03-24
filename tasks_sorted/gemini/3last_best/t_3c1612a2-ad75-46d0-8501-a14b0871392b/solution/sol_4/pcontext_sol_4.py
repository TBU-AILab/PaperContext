The output value of the best generated algorithm is: 3.642278637031552

 The best generated algorithm code:
#The best generated algorithm code is improved below.
#
#### Algorithm Description
#The proposed algorithm is an implementation of **JADE (Adaptive Differential Evolution) with an External Archive and Restart Mechanism**.
#
#**Key Improvements:**
#1.  **External Archive**: An archive of inferior solutions is maintained to preserve diversity. The mutation strategy `current-to-pbest/1/bin` selects the second difference vector ($x_{r2}$) from the union of the population and the archive. This prevents the difference vectors from vanishing too quickly as the population converges, allowing for more robust exploration.
#2.  **Adaptive Parameters**: The mutation factor $F$ and crossover rate $CR$ are self-adapted for each individual based on a success history (Lehmer mean for $F$, arithmetic mean for $CR$).
#3.  **Restart Mechanism**: To effectively utilize the `max_time`, the algorithm monitors population variance and stagnation. If the search converges or stagnates, it restarts the population (keeping the global best) to search different areas of the landscape.
#
#### Code
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Establish strict timing to ensure we return within bounds
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # -----------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------
    # Population size: 12*dim is a balanced choice for speed/diversity
    pop_size = max(20, 12 * dim)
    
    # Archive size (same as population size)
    archive_size = pop_size 
    
    # Adaptation parameters (JADE)
    c_adapt = 0.1   # Learning rate
    p_best_rate = 0.05 # Top % for p-best selection
    
    # Pre-process bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    bound_width = ub - lb
    
    # Global best tracker
    global_best_val = float('inf')
    
    # -----------------------------------------------------------
    # Main Optimization Loop (Restart Loop)
    # -----------------------------------------------------------
    while True:
        # Check time before starting a new restart
        if datetime.now() >= end_time:
            return global_best_val
            
        # --- Initialization Phase ---
        pop = lb + np.random.rand(pop_size, dim) * bound_width
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if datetime.now() >= end_time: return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Initialize JADE state variables
        mu_cr = 0.5
        mu_f = 0.5
        
        # Archive initialization
        archive = np.empty((archive_size, dim))
        arc_count = 0
        
        # Stagnation tracking
        last_gen_best = np.min(fitness)
        stagnation_count = 0
        
        # --- Evolution Phase ---
        while True:
            if datetime.now() >= end_time: return global_best_val

            # 1. Parameter Adaptation Generation
            # CR ~ Normal(mu_cr, 0.1), clipped to [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            # Generated using tan formulation: F = mu + 0.1 * tan(pi * (rand - 0.5))
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Robust clamping for F
            f = np.where(f >= 1.0, 1.0, f)
            f = np.where(f <= 0.0, 0.4, f) # Clamp negative/zero F to 0.4
            
            # 2. Mutation: current-to-pbest/1/bin with Archive
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            
            # Select p-best indices
            sorted_idx = np.argsort(fitness)
            num_pbest = max(1, int(p_best_rate * pop_size))
            pbest_indices = np.random.choice(sorted_idx[:num_pbest], pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (random from pop)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_indices]
            
            # Select r2 (random from Pop U Archive)
            union_size = pop_size + arc_count
            r2_indices = np.random.randint(0, union_size, pop_size)
            
            x_r2 = np.empty((pop_size, dim))
            
            # Map indices to actual vectors
            from_pop = r2_indices < pop_size
            x_r2[from_pop] = pop[r2_indices[from_pop]]
            
            # For indices >= pop_size, they map to archive
            from_arc = ~from_pop
            if np.any(from_arc):
                arc_idx = r2_indices[from_arc] - pop_size
                x_r2[from_arc] = archive[arc_idx]
            
            # Compute Mutant Vectors
            F_col = f.reshape(-1, 1)
            v = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr.reshape(-1, 1)
            # Force at least one dimension
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            u = np.where(cross_mask, v, pop)
            
            # 4. Boundary Constraint
            u = np.clip(u, lb, ub)
            
            # 5. Selection and Adaptation Update
            successful_cr = []
            successful_f = []
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_val
                
                new_val = func(u[i])
                
                if new_val < fitness[i]:
                    # Archive Update: Add parent to archive
                    if arc_count < archive_size:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Randomly replace if full
                        replace_idx = np.random.randint(0, archive_size)
                        archive[replace_idx] = pop[i].copy()
                        
                    # Population Update
                    fitness[i] = new_val
                    pop[i] = u[i]
                    
                    # Store successful parameters
                    successful_cr.append(cr[i])
                    successful_f.append(f[i])
                    
                    if new_val < global_best_val:
                        global_best_val = new_val
            
            # Update adaptive means (mu_cr, mu_f)
            if successful_cr:
                mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(successful_cr)
                
                sf = np.array(successful_f)
                denom = np.sum(sf)
                if denom > 1e-10:
                    # Lehmer Mean for F
                    mean_lehmer = np.sum(sf**2) / denom
                    mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer
                else:
                    mu_f = (1 - c_adapt) * mu_f
            
            # 6. Convergence / Restart Check
            curr_best_fit = np.min(fitness)
            std_dev = np.std(fitness)
            
            if abs(curr_best_fit - last_gen_best) < 1e-8:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_gen_best = curr_best_fit
            
            # Restart triggers:
            # 1. Very low variance (converged)
            # 2. Stuck for too many generations
            if std_dev < 1e-6 or stagnation_count > 35:
                break

    return global_best_val