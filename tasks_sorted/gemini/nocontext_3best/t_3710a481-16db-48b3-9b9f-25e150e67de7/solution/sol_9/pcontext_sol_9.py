#The proposed algorithm is an enhanced version of the **SHADE (Success-History based Adaptive Differential Evolution)** algorithm. Based on the analysis of previous attempts, SHADE (Algorithm 1) performed significantly better than standard methods. This improved version introduces a **Hybrid Restart Strategy** that combines local exploitation (Gaussian sampling around the best solution) with global exploration (random re-initialization) to effectively escape local optima and refine the best solution.
#
#Key features:
#1.  **SHADE Adaptation**: Uses historical memory ($M_F, M_{CR}$) to adapt mutation factor $F$ and crossover rate $CR$, learning the most successful parameters for the specific landscape.
#2.  **External Archive**: Maintains diversity by preserving recently replaced inferior solutions, preventing premature convergence.
#3.  **Exploitative Restart**: When the population stagnates (detected by fitness range), instead of a full random reset, it keeps the best solution and spawns a "cloud" of mutated clones around it (30% of population) to perform a stochastic local search, while the rest explore the global space.
#4.  **Robust Configuration**: Population size set to `18 * dim` and randomized $p$-best selection to balance exploration and exploitation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    SHADE (Success-History based Adaptive Differential Evolution) 
    with Archive and Exploitative Restart Strategy.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: 18 * dim provides a good balance for SHADE.
    # Minimum 40 ensures statistical stability for adaptive parameters.
    pop_size = int(max(40, 18 * dim))
    
    # Archive size: Stores inferior solutions to maintain diversity.
    # 2x population size allows for a richer history of "good" directions.
    archive_size = int(pop_size * 2)
    
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
    
    # Memory for Adaptive Parameters (M_F, M_CR) initialized to 0.5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive
    archive = np.zeros((archive_size, dim))
    arc_count = 0
    
    # Global Best Tracking
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
        # Check time limit
        if datetime.now() - start_time >= time_limit:
            return best_val
            
        # 1. Parameter Generation
        # -----------------------
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_f = mem_f[r_idx]
        mu_cr = mem_cr[r_idx]
        
        # Generate F using Cauchy distribution
        # F = Cauchy(mu_f, 0.1)
        F = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Handle F constraints: if F <= 0 retry; if F > 1 clip to 1
        while True:
            mask_neg = F <= 0
            if not np.any(mask_neg):
                break
            # Resample only invalid ones
            F[mask_neg] = mu_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
        F = np.minimum(F, 1.0)
        
        # Generate CR using Normal distribution
        # CR = Normal(mu_cr, 0.1)
        CR = np.random.normal(mu_cr, 0.1)
        CR = np.clip(CR, 0.0, 1.0)
        
        # 2. Mutation (DE/current-to-pbest/1)
        # -----------------------------------
        # Sort population to identify top individuals
        sorted_indices = np.argsort(fitness)
        
        # Select p-best: random p in [0.05, 0.2] for dynamic selection pressure
        p_share = np.random.uniform(0.05, 0.2)
        num_top = max(2, int(p_share * pop_size))
        top_indices = sorted_indices[:num_top]
        
        pbest_choices = np.random.choice(top_indices, pop_size)
        x_pbest = pop[pbest_choices]
        
        # Select r1 (distinct from current)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Handle self-collision
        cols = (r1_indices == np.arange(pop_size))
        if np.any(cols):
            r1_indices[cols] = np.random.randint(0, pop_size, np.sum(cols))
        x_r1 = pop[r1_indices]
        
        # Select r2 (from Population UNION Archive) to maintain diversity
        if arc_count > 0:
            pool = np.vstack((pop, archive[:arc_count]))
        else:
            pool = pop
        
        r2_indices = np.random.randint(0, pool.shape[0], pop_size)
        x_r2 = pool[r2_indices]
        
        # Compute Mutant Vectors
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
        
        # 3. Crossover (Binomial)
        # -----------------------
        rand_cross = np.random.rand(pop_size, dim)
        mask = rand_cross < CR[:, None]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutant, pop)
        
        # 4. Bound Handling
        # -----------------
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
                # Improvement or Neutral: Update Population
                
                # Archive: Save the parent before replacement
                if arc_count < archive_size:
                    archive[arc_count] = pop[i].copy()
                    arc_count += 1
                else:
                    # Random replacement in full archive
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = pop[i].copy()
                
                # Record successful parameters
                imp = fitness[i] - f_trial
                # Only record if there is a real improvement (avoid zero-div later)
                if imp > 0:
                    success_f.append(F[i])
                    success_cr.append(CR[i])
                    diff_fitness.append(imp)
                
                # Update Individual
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()
                    
        # 6. Memory Update
        # ----------------
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            w = np.array(diff_fitness)
            
            # Weighted Lehmer Mean
            if np.sum(w) > 0:
                w = w / np.sum(w)
                
                # Mean for F (Lehmer)
                mean_f = np.sum(w * (s_f**2)) / np.sum(w * s_f)
                
                # Mean for CR (Arithmetic)
                mean_cr = np.sum(w * s_cr)
                
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                
                k_mem = (k_mem + 1) % H

        # 7. Hybrid Restart Strategy
        # --------------------------
        # Check convergence: if fitness range is very small, population is stagnant.
        if not np.isinf(fitness).any():
            fit_range = np.max(fitness) - np.min(fitness)
            
            if fit_range < 1e-6:
                # Strategy:
                # 1. Keep the global best.
                # 2. Generate a "cloud" around the best (Exploitation).
                # 3. Randomize the rest (Exploration).
                
                # Index 0: Best found so far
                pop[0] = best_vec.copy()
                fitness[0] = best_val
                
                # Indices 1 to N_exploit: Gaussian cloud around best
                # 30% of population acts as a stochastic local search
                n_exploit = int(0.3 * pop_size)
                sigma_exploit = 0.01 * diff_b # 1% of domain width standard deviation
                
                exploit_pop = best_vec + np.random.randn(n_exploit, dim) * sigma_exploit
                exploit_pop = np.clip(exploit_pop, min_b, max_b)
                pop[1:1+n_exploit] = exploit_pop
                
                # Indices N_exploit+1 to End: Pure random
                n_remain = pop_size - (1 + n_exploit)
                pop[1+n_exploit:] = min_b + np.random.rand(n_remain, dim) * diff_b
                
                # Reset fitness for new/modified individuals to force re-evaluation
                fitness[1:] = float('inf')
                
                # Reset Adaptive Mechanisms
                mem_f.fill(0.5)
                mem_cr.fill(0.5)
                archive.fill(0)
                arc_count = 0

    return best_val
