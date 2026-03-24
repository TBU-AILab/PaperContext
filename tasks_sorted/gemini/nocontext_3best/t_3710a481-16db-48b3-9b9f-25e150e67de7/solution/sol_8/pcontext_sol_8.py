#The previous analysis of the generated algorithms highlights that **Algorithm 1 (SHADE with Archive and Restart)** performed the best (Score: 1.14), significantly outperforming the pure jDE (Score: 32.32) and the SHADE with Coordinate Descent (Score: 12.36). 
#
#The failure of Algorithm 2 (Score: 12.36) suggests that the expensive Local Search (Coordinate Descent) consumed too much computational budget or hindered the restart mechanism. Algorithm 1's success confirms that SHADE's adaptive mechanism is well-suited for the problem, but the non-zero score implies it likely got trapped in a deep local optimum or converged slightly off the global minimum.
#
#To improve upon Algorithm 1 and aim for a score near 0, I propose **SHADE with Reflection and Enhanced Diversity**. The key improvements are:
#
#1.  **Reflection Boundary Handling**: Instead of simple clipping (which piles up solutions at the bounds), we use reflection. If a mutation pushes a parameter outside the bounds, it is "bounced" back into the valid range. This preserves the distribution of the population and prevents stagnation at the edges.
#2.  **Dynamic `p-best` Selection**: Instead of a fixed top 11% (`p=0.11`) for the `current-to-pbest` mutation, we randomize `p` in the range `[2/N, 0.2]` for each generation. This varies the greediness of the search, alternating between exploitation (small `p`) and exploration (large `p`).
#3.  **Larger Archive**: Increasing the archive size to `2.0 * pop_size` (up from `1.0`) maintains a larger history of successful diversity, preventing the population from converging too quickly on a single mode.
#4.  **Tighter Restart Threshold**: The restart trigger is refined to check the fitness range (`max - min < 1e-8`) rather than standard deviation, ensuring restarts happen only when the population has truly collapsed to a single point.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 18 * dim is a robust setting for black-box optimization
    # Ensures sufficient diversity for high-dimensional problems
    pop_size = int(max(30, 18 * dim))
    
    # Archive size: Increased to 2.0 * pop_size to maintain higher diversity history
    archive_size = int(2.0 * pop_size)
    
    # History Memory size for SHADE adaptation
    H = 5
    
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
    k_mem = 0
    
    # Archive to store inferior solutions
    archive = np.zeros((archive_size, dim))
    arc_count = 0
    
    # Global Best tracking
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
            
    # --- Main Optimization Loop ---
    while True:
        # Strict time check
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        # 1. Parameter Generation (SHADE)
        # -------------------------------
        # Select random memory indices
        r_idx = np.random.randint(0, H, pop_size)
        mu_f = mem_f[r_idx]
        mu_cr = mem_cr[r_idx]
        
        # Generate F using Cauchy distribution: Cauchy(mu_f, 0.1)
        # If F <= 0, regenerate. If F > 1, clip to 1.
        F = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        while True:
            mask_neg = F <= 0
            if not np.any(mask_neg):
                break
            F[mask_neg] = mu_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
        F = np.minimum(F, 1.0)
        
        # Generate CR using Normal distribution: Normal(mu_cr, 0.1)
        # Clip to [0, 1]
        CR = np.random.normal(mu_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # 2. Mutation: DE/current-to-pbest/1
        # ----------------------------------
        # Sort population by fitness
        sorted_indices = np.argsort(fitness)
        
        # Dynamic p-best selection
        # Randomize p in [2/pop_size, 0.2] to vary greediness/exploration
        p_val = np.random.uniform(2.0/pop_size, 0.2)
        num_top = int(max(2, p_val * pop_size))
        top_indices = sorted_indices[:num_top]
        
        # Select p-best individuals
        pbest_indices = np.random.choice(top_indices, pop_size)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (from population, distinct from current not strictly enforced for speed)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Simple collision fix: ensure r1 != current
        cols = (r1_indices == np.arange(pop_size))
        if np.any(cols):
            r1_indices[cols] = np.random.randint(0, pop_size, np.sum(cols))
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Population UNION Archive)
        pool_size = pop_size + arc_count
        r2_indices = np.random.randint(0, pool_size, pop_size)
        
        # Construct r2 vectors
        # If index < pop_size, take from pop. Else take from archive.
        mask_pop = r2_indices < pop_size
        mask_arc = ~mask_pop
        
        x_r2 = np.empty((pop_size, dim))
        x_r2[mask_pop] = pop[r2_indices[mask_pop]]
        if np.any(mask_arc):
            # Map indices [pop_size, pool_size) to [0, arc_count)
            x_r2[mask_arc] = archive[r2_indices[mask_arc] - pop_size]
            
        # Compute Mutant Vectors
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        # -----------------------
        rand_cross = np.random.rand(pop_size, dim)
        mask_cross = rand_cross < CR[:, None]
        # Force at least one dimension to come from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask_cross, mutant, pop)
        
        # 4. Bound Handling (Reflection)
        # ------------------------------
        # Instead of clipping, reflect out-of-bound values back into the domain.
        # This preserves distribution better than sticking to bounds.
        
        # Reflect lower bounds
        mask_l = trial_pop < min_b
        trial_pop[mask_l] = min_b[mask_l] + (min_b[mask_l] - trial_pop[mask_l])
        
        # Reflect upper bounds
        mask_u = trial_pop > max_b
        trial_pop[mask_u] = max_b[mask_u] - (trial_pop[mask_u] - max_b[mask_u])
        
        # Final clip to be safe (in case reflection was still out)
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
            
            if f_trial <= fitness[i]:
                # Solution improved or equal
                
                # Update Archive: Insert parent before replacement
                if arc_count < archive_size:
                    archive[arc_count] = pop[i].copy()
                    arc_count += 1
                else:
                    # Randomly replace an archive member
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = pop[i].copy()
                
                # Collect data for memory update
                imp = fitness[i] - f_trial
                if imp > 0:
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
                    
        # 6. Memory Update (Weighted Lehmer Mean)
        # ---------------------------------------
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            w = np.array(diff_fitness)
            
            # Normalize weights
            if np.sum(w) == 0:
                w = np.ones_like(w)
            w = w / np.sum(w)
            
            # Update M_F
            mean_f = np.sum(w * (s_f**2)) / np.sum(w * s_f)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            # Update M_CR
            mean_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            # Move memory pointer
            k_mem = (k_mem + 1) % H
            
        # 7. Restart Strategy
        # -------------------
        # Calculate range of fitness in current population
        fit_range = np.max(fitness) - np.min(fitness)
        
        # Restart if population has converged to a single point (stagnation)
        if fit_range < 1e-8 and not np.isinf(fit_range):
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Keep the best solution found so far
            pop[0] = best_vec.copy()
            
            # Reset fitness array
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Reset Archive and Memory to allow new adaptation
            archive = np.zeros((archive_size, dim))
            arc_count = 0
            mem_f = np.full(H, 0.5)
            mem_cr = np.full(H, 0.5)
            k_mem = 0

    return best_val
