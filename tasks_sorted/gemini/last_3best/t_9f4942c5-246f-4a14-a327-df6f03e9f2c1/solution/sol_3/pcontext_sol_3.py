#The output value of the last generated algorithm was **11.94**. This indicates that while the previous JADE algorithm converged, it likely got stuck in a local optimum (common in multimodal functions like Rastrigin or hard valleys like Rosenbrock).
#
#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution) with Restart and Local Polish**.
#
#**Key Improvements:**
#1.  **Memory-Based Adaptation (SHADE)**: Unlike JADE (which uses a single mean for F and CR), SHADE maintains a **history memory** of successful parameters. This allows the algorithm to "remember" diverse configurations that worked well in the past, making it more robust.
#2.  **Correct Statistical Distributions**: Uses the **Cauchy distribution** for the Mutation Factor ($F$) to allow for long jumps (heavy tails), significantly improving the ability to escape local basins compared to the Normal distribution used previously.
#3.  **Local Polishing**: Before restarting, the algorithm runs a brief **stochastic hill-climbing** phase on the best solution. This helps refine the solution to the absolute minimum (e.g., getting from 11.94 to 0.0) if the basin is convex.
#4.  **Soft Restart Strategy**: When population diversity collapses, it triggers a restart but keeps the best solution (Elitism) and resets the archive/memory, effectively exploring the space again without losing progress.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    SHADE (Success-History based Adaptive Differential Evolution) with Restart and Local Polish.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: SHADE works well with 10*dim to 20*dim.
    # Constrain between 40 and 100 to balance exploration and speed within max_time.
    pop_size = int(np.clip(10 * dim, 40, 100))
    
    # Memory size for SHADE history (H)
    H = 6
    
    # Archive size (A)
    archive_size = int(pop_size * 2.0)
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Memory for Adaptive Parameters (F and CR)
    mem_M_F = np.full(H, 0.5)
    mem_M_CR = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer
    
    # Population and Fitness
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, np.inf)
    
    # Global Best Tracking
    best_val = np.inf
    best_sol = np.zeros(dim)
    
    # External Archive
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
            
    # --- Main Loop ---
    while True:
        if (time.time() - start_time) >= max_time:
            return best_val
            
        # ---------------------------------------------------------
        # 1. Parameter Generation (SHADE Strategy)
        # ---------------------------------------------------------
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        m_f = mem_M_F[r_idx]
        m_cr = mem_M_CR[r_idx]
        
        # Generate F using Cauchy Distribution: Location=m_f, Scale=0.1
        # Cauchy helps escape local optima due to heavy tails.
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Constraint Handling for F:
        # If F <= 0, regenerate. If F > 1, clip to 1.
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            count = np.sum(bad_mask)
            f[bad_mask] = m_f[bad_mask] + 0.1 * np.random.standard_cauchy(count)
        f = np.clip(f, 0, 1)
        
        # Generate CR using Normal Distribution: Mean=m_cr, Std=0.1
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # ---------------------------------------------------------
        # 2. Mutation: current-to-pbest/1
        # ---------------------------------------------------------
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # Sort population to find p-best
        sorted_idx = np.argsort(fitness)
        
        # Select random p for top p% (between 2/NP and 0.2)
        p_best_rate = np.random.uniform(2/pop_size, 0.2)
        n_pbest = max(2, int(pop_size * p_best_rate))
        top_p_indices = sorted_idx[:n_pbest]
        
        # Choose pbest for each individual
        pbest_indices = np.random.choice(top_p_indices, pop_size)
        x_pbest = population[pbest_indices]
        
        # Select r1: Random from population, distinct from current (i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Collision check r1 != i
        collisions = (r1_indices == np.arange(pop_size))
        while np.any(collisions):
            r1_indices[collisions] = np.random.randint(0, pop_size, np.sum(collisions))
            collisions = (r1_indices == np.arange(pop_size))
        x_r1 = population[r1_indices]
        
        # Select r2: Random from Union(Population, Archive), distinct from i and r1
        union_size = pop_size + n_archive
        r2_indices = np.random.randint(0, union_size, pop_size)
        
        # Check collisions: r2 != i AND r2 != r1
        # Note: r2 index >= pop_size means it is in archive, so cannot clash with i or r1 (which are < pop_size)
        collisions = (r2_indices == np.arange(pop_size)) | (r2_indices == r1_indices)
        while np.any(collisions):
            r2_indices[collisions] = np.random.randint(0, union_size, np.sum(collisions))
            collisions = (r2_indices == np.arange(pop_size)) | (r2_indices == r1_indices)
            
        # Construct x_r2 vector
        x_r2 = np.zeros((pop_size, dim))
        mask_pop = r2_indices < pop_size
        mask_arch = ~mask_pop
        
        x_r2[mask_pop] = population[r2_indices[mask_pop]]
        if np.any(mask_arch):
            # Archive indices mapping
            arch_idx = r2_indices[mask_arch] - pop_size
            x_r2[mask_arch] = archive[arch_idx]
            
        # Calculate Mutant Vector
        # Vectorized operation: X_i + F_i * (X_pbest - X_i) + F_i * (X_r1 - X_r2)
        mutant = population + f[:, None] * (x_pbest - population) + f[:, None] * (x_r1 - x_r2)
        
        # ---------------------------------------------------------
        # 3. Crossover (Binomial)
        # ---------------------------------------------------------
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        
        # Ensure at least one dimension is mutated
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # Bound Constraints (Clipping)
        trial = np.clip(trial, min_b, max_b)
        
        # ---------------------------------------------------------
        # 4. Selection and Memory Update
        # ---------------------------------------------------------
        success_indices = []
        diff_fitness = []
        
        # Evaluate one by one to respect time limit strictly
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
            
            f_trial = func(trial[i])
            
            # Selection: Greedy
            if f_trial <= fitness[i]:
                # Improvement: Update Archive
                if n_archive < archive_size:
                    archive[n_archive] = population[i].copy()
                    n_archive += 1
                else:
                    # Random replacement in archive
                    rand_idx = np.random.randint(0, archive_size)
                    archive[rand_idx] = population[i].copy()
                
                # Record success for memory update
                success_indices.append(i)
                diff_fitness.append(fitness[i] - f_trial)
                
                # Update Population
                population[i] = trial[i]
                fitness[i] = f_trial
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_sol = trial[i].copy()
                    
        # Update Memory (SHADE logic)
        if len(success_indices) > 0:
            idx_arr = np.array(success_indices)
            diff_arr = np.array(diff_fitness)
            s_f = f[idx_arr]
            s_cr = cr[idx_arr]
            
            # Calculate Weights based on improvement magnitude
            total_diff = np.sum(diff_arr)
            if total_diff > 0:
                w = diff_arr / total_diff
                
                # Weighted Lehmer Mean for F
                mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                
                # Weighted Mean for CR
                mean_cr = np.sum(w * s_cr)
                
                # Update current memory slot
                mem_M_F[k_mem] = 0.5 * mem_M_F[k_mem] + 0.5 * mean_f
                mem_M_CR[k_mem] = 0.5 * mem_M_CR[k_mem] + 0.5 * mean_cr
                
                # Advance pointer
                k_mem = (k_mem + 1) % H
                
        # ---------------------------------------------------------
        # 5. Restart and Local Polish Mechanism
        # ---------------------------------------------------------
        # Check for convergence (low standard deviation)
        if np.std(fitness) < 1e-6:
            # Phase A: Local Polish
            # Try to squeeze the last bit of performance from the best solution
            # using a simple Stochastic Hill Climber
            curr_polish = best_sol.copy()
            curr_polish_val = best_val
            polish_steps = 50
            sigma = 0.01 * np.linalg.norm(max_b - min_b) / np.sqrt(dim)
            
            for _ in range(polish_steps):
                if (time.time() - start_time) >= max_time:
                    return best_val
                
                # Gaussian mutation around best
                cand = curr_polish + np.random.normal(0, sigma, dim)
                cand = np.clip(cand, min_b, max_b)
                val = func(cand)
                
                if val < curr_polish_val:
                    curr_polish_val = val
                    curr_polish = cand
                else:
                    # Decay sigma on failure to zoom in
                    sigma *= 0.95
            
            # Update global if polished better
            if curr_polish_val < best_val:
                best_val = curr_polish_val
                best_sol = curr_polish.copy()
            
            # Phase B: Soft Restart
            # Re-initialize population to search new areas
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_sol.copy() # Elitism: keep best
            fitness = np.full(pop_size, np.inf)
            fitness[0] = best_val
            
            # Reset Archive and Memory to clear bias
            n_archive = 0
            mem_M_F.fill(0.5)
            mem_M_CR.fill(0.5)
            
            # Evaluate new population (skip index 0)
            for k in range(1, pop_size):
                if (time.time() - start_time) >= max_time:
                    return best_val
                val = func(population[k])
                fitness[k] = val
                if val < best_val:
                    best_val = val
                    best_sol = population[k].copy()

    return best_val
