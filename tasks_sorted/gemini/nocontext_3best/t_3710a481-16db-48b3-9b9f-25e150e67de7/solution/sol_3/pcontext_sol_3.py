#The output values and codes for the best generated algorithms are as follows:
#1. output value is: 32.321142172045555
#2. output value is: 37.861043545364
#3. output value is: 87.40362767056685
#
#**Analysis of Previous Attempts:**
#- **Algorithm 1 (jDE with Restart)** performed the best (32.32). Its strength lies in the restart mechanism which helps escape local optima, and the simple self-adaptation of parameters.
#- **Algorithm 2 (jDE + DE/current-to-pbest/1 + Local Search)** performed slightly worse (37.86). While `current-to-pbest` is a powerful mutation strategy, it tends to converge faster. Without an external **Archive** of inferior solutions (a standard component of modern DE variants like SHADE), the population likely lost diversity too quickly, making the difference vectors small and stalling the search. The local search might have also consumed time inefficiently.
#- **Algorithm 3 (Standard DE)** performed poorly (87.40), confirming that adaptive parameters and advanced strategies are necessary.
#
#**Proposed Improvements:**
#I propose an algorithm based on **SHADE (Success-History based Adaptive Differential Evolution)**, simplified for this context but retaining its core strengths, combined with the successful **Restart Strategy** from Algorithm 1.
#1.  **Archive Implementation**: Unlike Algorithm 2, this version includes an **external archive** of recently replaced solutions. This is crucial for the `current-to-pbest` strategy to work effectively, as it maintains diversity in the mutation step even when the population converges.
#2.  **History-Based Adaptation**: Instead of random resets (jDE), we use a history memory ($M_F, M_{CR}$) that learns successful parameter values over time using a weighted Lehmer mean. This guides the search more efficiently.
#3.  **Robust Restart**: We keep the restart logic (resetting population but keeping the best solution) when the population standard deviation drops below a threshold.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # SHADE-inspired configuration
    # Population size: 15-20 * dim is robust.
    # We use max(30, 15 * dim) to ensure enough diversity in higher dimensions.
    pop_size = int(max(30, 15 * dim))
    
    # Archive size typically equals population size
    archive_size = pop_size
    
    # History Memory size for adaptive parameters
    H = 6
    
    # Pre-process bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Memory for Adaptive Parameters (M_F, M_CR)
    # Initialize with 0.5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0 # Index pointer for memory update
    
    # Archive to store inferior solutions (maintains diversity)
    archive = np.zeros((archive_size, dim))
    arc_count = 0
    
    # Track Global Best
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # --- Main Loop ---
    while True:
        # Check time constraint strictly
        if datetime.now() - start_time >= time_limit:
            return best_val
            
        # 1. Parameter Adaptation
        # -----------------------
        # Pick random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_f = mem_f[r_idx]
        mu_cr = mem_cr[r_idx]
        
        # Generate F using Cauchy distribution: Cauchy(mu_f, 0.1)
        F = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Handle F constraints: if F <= 0 resample; if F > 1 clip to 1.
        while True:
            mask_neg = F <= 0
            if not np.any(mask_neg):
                break
            F[mask_neg] = mu_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
        F = np.minimum(F, 1.0)
        
        # Generate CR using Normal distribution: Normal(mu_cr, 0.1)
        CR = np.random.normal(mu_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # 2. Mutation: DE/current-to-pbest/1 with Archive
        # -----------------------------------------------
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        
        # Select p-best from top 11% (robust setting)
        p_share = 0.11
        num_top = max(2, int(p_share * pop_size))
        top_indices = sorted_indices[:num_top]
        
        pbest_choices = np.random.choice(top_indices, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1 (from population, distinct from current)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Simple collision handling
        cols = (r1_indices == np.arange(pop_size))
        if np.any(cols):
            r1_indices[cols] = np.random.randint(0, pop_size, np.sum(cols))
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Population UNION Archive)
        if arc_count > 0:
            pool = np.vstack((pop, archive[:arc_count]))
        else:
            pool = pop
        
        r2_indices = np.random.randint(0, pool.shape[0], pop_size)
        x_r2 = pool[r2_indices]
        
        # Compute mutant vectors
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover
        # ------------
        rand_cross = np.random.rand(pop_size, dim)
        mask = rand_cross < CR[:, None]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutant, pop)
        
        # 4. Bound Constraints
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 5. Evaluation & Selection
        # -------------------------
        success_f = []
        success_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                # Improvement or neutral move
                
                # Add parent to archive before replacement
                if arc_count < archive_size:
                    archive[arc_count] = pop[i].copy()
                    arc_count += 1
                else:
                    # Replace random archive member
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = pop[i].copy()
                
                # Record data for memory update
                imp = fitness[i] - f_trial
                success_f.append(F[i])
                success_cr.append(CR[i])
                diff_fitness.append(imp)
                
                # Update Population
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()
                    
        # 6. Update Memory (History)
        # --------------------------
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            w = np.array(diff_fitness)
            
            # If all fitness differences are 0, use equal weights
            if np.sum(w) == 0:
                w = np.ones_like(w)
            
            # Normalize weights
            w = w / np.sum(w)
            
            # Weighted Lehmer Mean for F
            denom = np.sum(w * s_f)
            if denom == 0:
                 mean_f = 0.5
            else:
                 mean_f = np.sum(w * (s_f**2)) / denom
            
            # Weighted Arithmetic Mean for CR
            mean_cr = np.sum(w * s_cr)
            
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 7. Restart Strategy
        # -------------------
        # If population converges (low std dev), restart to explore new areas.
        if not np.isinf(fitness).any():
            std_dev = np.std(fitness)
            if std_dev < 1e-6:
                # Reset population to random values
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                # Inject the best solution found so far
                pop[0] = best_vec
                
                # Reset fitness
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_val
                
                # Reset Archive and Memory
                archive = np.zeros((archive_size, dim))
                arc_count = 0
                mem_f = np.full(H, 0.5)
                mem_cr = np.full(H, 0.5)

    return best_val
