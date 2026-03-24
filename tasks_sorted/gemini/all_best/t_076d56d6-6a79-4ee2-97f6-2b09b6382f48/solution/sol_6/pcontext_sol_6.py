#The following Python code implements **Restart-SHADE with Local Search (R-SHADE-LS)**. 
#
#### Key Improvements:
#1.  **SHADE Adaptation**: Instead of using fixed or random parameter distributions, the algorithm maintains a historical memory (`M_F`, `M_CR`) of successful `F` and `CR` values. This allows the algorithm to "learn" the best mutation and crossover rates for the specific function landscape (e.g., separable vs. non-separable).
#2.  **External Archive**: An external archive stores recently replaced inferior solutions. Using these in the `current-to-pbest/1` mutation strategy (`mutant = x + F*(pbest - x) + F*(r1 - r2_archive)`) significantly increases diversity and prevents premature convergence.
#3.  **IPOP (Increasing Population) Restart**: When the population converges (stagnation), the search restarts with a population size that grows exponentially ($P_{new} = P_{base} \times 2^{restarts}$). This strategy allows the algorithm to perform a fast, coarse search initially and then switch to a detailed, dense search if time permits.
#4.  **Local Search (LS)**: Before each restart, a coordinate-descent-based Local Search is applied to the global best solution. This helps to fine-tune the solution and extract maximum precision from the basin of attraction found by the global search.
#5.  **Efficiency**: The implementation uses vectorized NumPy operations for generating candidates and handling boundaries, ensuring maximum throughput of function evaluations within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Restart-SHADE with Local Search (R-SHADE-LS).
    Combines adaptive Differential Evolution with an archive, increasing population
    restarts, and a greedy local search for fine-tuning.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracking ---
    best_val = float('inf')
    best_vec = None
    
    # --- Configuration ---
    # Start with a conservative population size to get quick results
    # Pop size typically 10*dim, clamped to [20, 100] for the first run
    base_pop_size = max(20, min(10 * dim, 80))
    restart_count = 0
    
    # SHADE Memory parameters
    memory_size = 5
    
    # --- Main Restart Loop ---
    while True:
        # Check time before committing to a new restart
        if datetime.now() - start_time >= time_limit:
            return best_val
        
        # IPOP: Double population size for each restart to handle complex landscapes
        pop_size = int(base_pop_size * (2.0 ** restart_count))
        
        # --- Initialization ---
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best into new population
        start_eval_idx = 0
        if best_vec is not None:
            population[0] = best_vec.copy()
            fitness[0] = best_val
            start_eval_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            # Frequent time check during initialization
            if i % 50 == 0 and datetime.now() - start_time >= time_limit:
                return best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_vec = population[i].copy()
                
        # --- SHADE Memory Initialization ---
        M_CR = np.full(memory_size, 0.5)
        M_F = np.full(memory_size, 0.5)
        k_mem = 0
        
        # --- Archive Initialization ---
        # Stores decent solutions to maintain diversity
        archive = []
        
        # --- Evolution Loop ---
        while True:
            # Time check per generation
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            # --- Convergence Check ---
            # If population fitness range is tiny, we are stuck
            min_fit = np.min(fitness)
            max_fit = np.max(fitness)
            if (max_fit - min_fit) < 1e-8:
                break 
            
            # --- 1. Parameter Adaptation ---
            # Pick random memory index for each individual
            r_idx = np.random.randint(0, memory_size, pop_size)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]
            
            # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F: Cauchy(m_f, 0.1)
            u = np.random.rand(pop_size)
            f = m_f + 0.1 * np.tan(np.pi * (u - 0.5))
            
            # Handle F constraints: F > 0 (retry), F <= 1 (clip)
            # Vectorized retry logic
            bad_f_mask = f <= 0
            while np.any(bad_f_mask):
                cnt = np.sum(bad_f_mask)
                u_retry = np.random.rand(cnt)
                # Retry using the same m_f for consistency
                f[bad_f_mask] = m_f[bad_f_mask] + 0.1 * np.tan(np.pi * (u_retry - 0.5))
                bad_f_mask = f <= 0
            f = np.clip(f, 0.0, 1.0)
            
            # --- 2. Mutation: current-to-pbest/1 ---
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            
            # Select p-best (top 10% of population)
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            num_top = max(2, int(0.1 * pop_size))
            top_indices = sorted_indices[:num_top]
            
            # Randomly select one pbest for each individual
            pbest_idx = np.random.choice(top_indices, pop_size)
            x_pbest = population[pbest_idx]
            
            # Select r1 (random from population, try to distinguish from self)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Simple collision handling: if r1 == i, shift it
            collision = (r1_idx == np.arange(pop_size))
            r1_idx[collision] = (r1_idx[collision] + 1) % pop_size
            x_r1 = population[r1_idx]
            
            # Select r2 (random from Union(Population, Archive))
            if len(archive) > 0:
                archive_np = np.array(archive)
                # Indices: 0 to pop_size-1 -> Population
                #          pop_size to total-1 -> Archive
                total_candidates = pop_size + len(archive)
                r2_idx = np.random.randint(0, total_candidates, pop_size)
                
                # Construct x_r2 based on indices
                x_r2 = np.empty((pop_size, dim))
                
                # Mask for population sources
                mask_pop = r2_idx < pop_size
                x_r2[mask_pop] = population[r2_idx[mask_pop]]
                
                # Mask for archive sources
                mask_arc = ~mask_pop
                x_r2[mask_arc] = archive_np[r2_idx[mask_arc] - pop_size]
            else:
                r2_idx = np.random.randint(0, pop_size, pop_size)
                # Avoid collision with r1
                collision = (r2_idx == r1_idx) | (r2_idx == np.arange(pop_size))
                r2_idx[collision] = (r2_idx[collision] + 1) % pop_size
                x_r2 = population[r2_idx]
                
            # Compute Mutant Vector
            f_col = f.reshape(-1, 1)
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # --- 3. Crossover (Binomial) ---
            cr_col = cr.reshape(-1, 1)
            cross_mask = np.random.rand(pop_size, dim) < cr_col
            
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # --- 4. Bound Handling ---
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- 5. Selection & Memory Update ---
            succ_f = []
            succ_cr = []
            diff_fitness = []
            
            # Evaluate trial vectors
            for i in range(pop_size):
                # Check time frequently
                if i % 20 == 0 and datetime.now() - start_time >= time_limit:
                    return best_val
                
                f_trial = func(trial_pop[i])
                f_old = fitness[i]
                
                if f_trial < f_old:
                    # Successful Update
                    fitness_imp = f_old - f_trial
                    
                    # Add parent to archive
                    if len(archive) < pop_size:
                        archive.append(population[i].copy())
                    else:
                        # Replace random archive member
                        rem_idx = np.random.randint(0, len(archive))
                        archive[rem_idx] = population[i].copy()
                        
                    # Store successful parameters
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    diff_fitness.append(fitness_imp)
                    
                    # Update Population
                    population[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    # Update Global Best
                    if f_trial < best_val:
                        best_val = f_trial
                        best_vec = trial_pop[i].copy()
            
            # Maintain Archive Size (Strictly <= pop_size)
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update SHADE Memory (Weighted Lehmer Mean)
            if len(succ_f) > 0:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                w_diff = np.array(diff_fitness)
                
                # Normalize weights
                total_imp = np.sum(w_diff)
                if total_imp > 0:
                    weights = w_diff / total_imp
                    
                    # Mean CR (Weighted Arithmetic)
                    mean_cr = np.sum(weights * s_cr)
                    
                    # Mean F (Weighted Lehmer)
                    denom_f = np.sum(weights * s_f)
                    if denom_f > 0:
                        mean_f = np.sum(weights * (s_f ** 2)) / denom_f
                    else:
                        mean_f = 0.5
                        
                    # Update memory at pointer k
                    M_CR[k_mem] = np.clip(mean_cr, 0.0, 1.0)
                    M_F[k_mem] = np.clip(mean_f, 0.0, 1.0)
                    
                    k_mem = (k_mem + 1) % memory_size
        
        # --- End of Generation Loop (Converged) ---
        
        # --- Local Search (Fine Tuning) ---
        # Before restarting, perform a quick coordinate descent on the best solution
        # to squeeze out precision from the current basin.
        if best_vec is not None:
            # Step size relative to domain size
            step = diff_b * 0.05
            ls_iter = 0
            
            # Simple Coordinate Descent
            while np.max(step) > 1e-8 and ls_iter < 50:
                if datetime.now() - start_time >= time_limit:
                    return best_val
                
                improved = False
                # Try all dimensions in random order
                dims_perm = np.random.permutation(dim)
                
                for d in dims_perm:
                    if datetime.now() - start_time >= time_limit:
                        return best_val
                        
                    orig_val = best_vec[d]
                    
                    # Try negative direction
                    best_vec[d] = np.clip(orig_val - step[d], min_b[d], max_b[d])
                    val = func(best_vec)
                    
                    if val < best_val:
                        best_val = val
                        improved = True
                    else:
                        # Try positive direction
                        best_vec[d] = np.clip(orig_val + step[d], min_b[d], max_b[d])
                        val = func(best_vec)
                        
                        if val < best_val:
                            best_val = val
                            improved = True
                        else:
                            # Revert
                            best_vec[d] = orig_val
                
                if not improved:
                    step *= 0.5 # Shrink step size
                
                ls_iter += 1
                
        # Prepare for next restart with larger population
        restart_count += 1

    return best_val
