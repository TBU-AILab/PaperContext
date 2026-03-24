#The following Python algorithm implements **SHADE (Success-History based Adaptive Differential Evolution) with a Restart Mechanism**.
#
#### Key Improvements
#1.  **SHADE Strategy**: Implements the state-of-the-art SHADE algorithm which uses a historical memory to adapt the control parameters ($F$ and $CR$) based on the success of previous generations. This allows the algorithm to learn the landscape properties much faster than static or simple adaptive DE.
#2.  **`current-to-pbest/1` Mutation**: Utilizes a mutation strategy that guides individuals toward the top $p\%$ of the best solutions found so far (exploitation) while using an external archive of inferior solutions to maintain direction diversity (exploration).
#3.  **External Archive**: Stores recently replaced solutions to provide diverse difference vectors for mutation, preventing premature convergence.
#4.  **Restart Mechanism**: Monitors population diversity (fitness standard deviation). If the population converges to a local optimum before the time limit, it saves the best solution (elite), restarts the rest of the population, and resets the historical memory to explore new basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using SHADE (Success-History based Adaptive DE) 
    with a Restart Mechanism within 'max_time'.
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Set a strict deadline with a small buffer for return operations
    deadline = start_time + timedelta(seconds=max_time - 0.05)
    
    # --- Configuration ---
    # Population size: Adaptive to dimension, clamped to ensure throughput
    # SHADE typically performs well with D-dependent population size
    pop_size = int(np.clip(20 * dim, 50, 150))
    
    # SHADE Memory Parameters
    # History size H
    H = 5
    mem_cr = np.full(H, 0.5) # Memory for Crossover Rate
    mem_f = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0                # Memory index pointer
    
    # Archive Configuration
    # Stores diverse solutions for mutation strategy
    archive_size = pop_size
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= deadline:
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while datetime.now() < deadline:
        
        # 1. Restart Mechanism (Convergence Detection)
        # If population diversity is lost (low std dev), restart to explore new areas
        if np.std(fitness) < 1e-6:
            # Preserve the Elite (Global Best)
            elite = best_sol.copy()
            elite_fit = best_fitness
            
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Inject Elite at index 0
            pop[0] = elite
            fitness[:] = float('inf')
            fitness[0] = elite_fit
            
            # Reset SHADE Memory and Archive
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            n_archive = 0
            
            # Evaluate new population (skipping elite)
            for i in range(1, pop_size):
                if datetime.now() >= deadline: return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            
            # Continue to next generation immediately
            continue

        # 2. Generate Adaptive Parameters (Vectorized)
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idx]
        mu_f = mem_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # Handle F Constraints
        # If F > 1, clamp to 1. If F <= 0, clamp to small value (force mutation)
        f = np.where(f > 1.0, 1.0, f)
        f = np.where(f <= 0.0, 0.1, f)
        
        # 3. Mutation Strategy: current-to-pbest/1
        # Sort population to identify p-best
        sort_idx = np.argsort(fitness)
        pop_sorted = pop[sort_idx]
        
        # Select p-best individuals (randomly from top p%)
        # p is typically ~0.1 (top 10%)
        top_p_count = max(2, int(pop_size * 0.1))
        pbest_indices = np.random.randint(0, top_p_count, pop_size)
        x_pbest = pop_sorted[pbest_indices]
        
        # Select r1 (random from pop, distinct from current i)
        # We ensure approximate distinctness via index manipulation
        r1_indices = np.random.randint(0, pop_size, pop_size)
        conflict_mask = (r1_indices == np.arange(pop_size))
        r1_indices[conflict_mask] = (r1_indices[conflict_mask] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (random from Union(Pop, Archive))
        if n_archive > 0:
            # Create a view of the union
            pop_union = np.vstack((pop, archive[:n_archive]))
        else:
            pop_union = pop
            
        r2_indices = np.random.randint(0, len(pop_union), pop_size)
        x_r2 = pop_union[r2_indices]
        
        # Compute Mutation Vectors
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        f_col = f[:, np.newaxis]
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = rand_vals < cr[:, np.newaxis]
        # Ensure at least one dimension is taken from mutant
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 5. Bound Constraints (Clipping)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 6. Selection and Adaptation Update
        successful_f = []
        successful_cr = []
        improvement_diffs = []
        
        for i in range(pop_size):
            if datetime.now() >= deadline:
                return best_fitness
            
            # Evaluate Candidate
            f_trial = func(trial_pop[i])
            
            # Greedy Selection
            if f_trial < fitness[i]:
                diff = fitness[i] - f_trial
                
                # Add old solution to archive
                if n_archive < archive_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    # Randomly replace an archive member
                    replace_idx = np.random.randint(0, archive_size)
                    archive[replace_idx] = pop[i].copy()
                
                # Accept new solution
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                # Record success for adaptation
                successful_f.append(f[i])
                successful_cr.append(cr[i])
                improvement_diffs.append(diff)
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    
        # 7. Update SHADE History Memory
        if len(successful_f) > 0:
            s_f = np.array(successful_f)
            s_cr = np.array(successful_cr)
            imp = np.array(improvement_diffs)
            
            # Calculate weights based on fitness improvement amount
            total_imp = np.sum(imp)
            weights = imp / total_imp if total_imp > 0 else np.ones_like(imp) / len(imp)
            
            # Update Mean CR (Weighted Arithmetic Mean)
            mean_cr = np.sum(weights * s_cr)
            
            # Update Mean F (Weighted Lehmer Mean)
            # mean_lehmer = sum(w * f^2) / sum(w * f)
            num = np.sum(weights * (s_f ** 2))
            den = np.sum(weights * s_f)
            mean_f = num / den if den > 0 else 0.5
            
            # Update Memory at pointer k
            mem_cr[k_mem] = mean_cr
            mem_f[k_mem] = mean_f
            k_mem = (k_mem + 1) % H

    return best_fitness
