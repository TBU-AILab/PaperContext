#The following algorithm implements **SHADE (Success-History based Adaptive Differential Evolution) with Restarts and Local Search Polishing**.
#
#**Key Improvements over the previous JADE implementation:**
#1.  **External Archive:** It utilizes an external archive of inferior solutions. This allows the mutation operator to draw difference vectors from a wider pool of genetic material (historical data) without increasing the computational cost of evaluating a larger population. This significantly improves diversity preservation.
#2.  **Historical Memory (Success History):** Instead of randomizing parameters $F$ and $CR$ around fixed means, this algorithm learns. It maintains a memory bank of parameter pairs that successfully generated improved offspring. New parameters are generated based on this historical success, allowing the algorithm to adapt to the specific landscape of the function (e.g., separable vs. non-separable).
#3.  **Local Search Polishing:** Before restarting (when the population converges), the algorithm executes a fast, coordinate-wise local search (Polishing) on the best individual. This exploits the final basin of attraction to extract maximum precision.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History based Adaptive DE)
    with an external archive, restarts, and local search polishing.
    """
    start_time = time.time()
    
    # --- Helper: Bounds Management ---
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    diff_b = upper_b - lower_b
    
    # --- Global Best Tracking ---
    global_best_x = None
    global_best_fitness = float('inf')

    # --- SHADE Parameters ---
    # Memory size for historical parameter adaptation
    H_memory_size = 6
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check overall time remaining
        if (time.time() - start_time) > max_time - 0.1:
            return global_best_fitness

        # 1. Initialization for this Restart
        # Population size: usually 18*D is good for SHADE, bounded [20, 100] for balance
        pop_size = int(np.clip(18 * dim, 20, 100))
        
        # Initialize Population
        pop = lower_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_fitness
            
            f_val = func(pop[i])
            fitness[i] = f_val
            
            if f_val < global_best_fitness:
                global_best_fitness = f_val
                global_best_x = pop[i].copy()

        # Initialize Memory (M_CR and M_F)
        # 0.5 is a safe starting point
        M_CR = np.full(H_memory_size, 0.5)
        M_F = np.full(H_memory_size, 0.5)
        k_mem = 0  # Memory index counter

        # External Archive (stores replaced individuals)
        # Format: list of numpy arrays
        archive = []
        max_archive_size = pop_size

        # Convergence tracking
        stagnation_counter = 0
        
        # --- Evolution Loop ---
        while True:
            current_time = time.time()
            if (current_time - start_time) >= max_time:
                return global_best_fitness

            # Sort population for p-best selection
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # 2. Parameter Adaptation
            # Generate CR and F for each individual based on Memory
            # Pick random memory index for each individual
            r_idx = np.random.randint(0, H_memory_size, size=pop_size)
            m_cr = M_CR[r_idx]
            m_f = M_F[r_idx]

            # Generate CR: Normal distribution around memory, clipped [0, 1]
            # If CR is close to 0, we clamp to 0 (no crossover), but usually we want some mixing.
            # SHADE specific: if CR < 0 -> 0; if CR > 1 -> 1. 
            # Ideally Normal(m_cr, 0.1)
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # Generate F: Cauchy distribution around memory
            # Cauchy = loc + scale * standard_cauchy
            # If F > 1 -> 1. If F <= 0 -> regenerate.
            F = m_f + 0.1 * np.random.standard_cauchy(size=pop_size)
            
            # Repair F values
            retry_mask = F <= 0
            while np.any(retry_mask):
                F[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(size=np.sum(retry_mask))
                retry_mask = F <= 0
            F = np.clip(F, 0.0, 1.0) # Actually standard SHADE clips 1.0, but regeneration logic handles <= 0

            # 3. Mutation: current-to-pbest/1 with Archive
            # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            
            # p-best selection (top 5% to 20% random)
            p_share = np.random.uniform(0.05, 0.2)
            p_num = max(1, int(pop_size * p_share))
            pbest_indices = np.random.randint(0, p_num, size=pop_size)
            x_pbest = pop[pbest_indices]

            # r1 selection (random from pop, distinct from i)
            # Simplified: random from pop (diversity usually handles the "distinct" requirement approx)
            r1_indices = np.random.randint(0, pop_size, size=pop_size)
            x_r1 = pop[r1_indices]

            # r2 selection (Union of Pop and Archive)
            # Create the union pool
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((pop, archive_np))
            else:
                union_pop = pop
            
            r2_indices = np.random.randint(0, len(union_pop), size=pop_size)
            x_r2 = union_pop[r2_indices]

            # Calculate mutant vectors (vectorized)
            F_col = F[:, np.newaxis]
            # x_target is simply the current population (since we sorted, pop[i] is the target)
            x_target = pop
            
            diff_1 = x_pbest - x_target
            diff_2 = x_r1 - x_r2
            mutant = x_target + F_col * diff_1 + F_col * diff_2
            
            # Boundary handling (bounce back or clip) - Clipping is safer for standard templates
            mutant = np.clip(mutant, lower_b, upper_b)

            # 4. Crossover (Binomial)
            cross_mask = np.random.rand(pop_size, dim) < CR[:, np.newaxis]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, size=pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, pop)

            # 5. Selection and Memory Update
            success_F = []
            success_CR = []
            diff_fitness = []
            
            improved_any = False
            
            for i in range(pop_size):
                if (time.time() - start_time) >= max_time:
                    return global_best_fitness
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    # Improvement or neutral move
                    if f_trial < fitness[i]:
                        # Store improvement magnitude for weighted memory update
                        diff_fitness.append(fitness[i] - f_trial)
                        success_F.append(F[i])
                        success_CR.append(CR[i])
                        
                        # Add old solution to archive
                        archive.append(pop[i].copy())
                        improved_any = True
                    
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial
                        global_best_x = trial_pop[i].copy()
                        # Reset stagnation if global best improves
                        stagnation_counter = 0 
            
            # Maintain Archive Size
            while len(archive) > max_archive_size:
                # Remove random elements to keep size constant
                rem_idx = np.random.randint(0, len(archive))
                archive.pop(rem_idx)

            # 6. Update Memory (Weighted Lehmer Mean)
            if len(success_F) > 0:
                success_F = np.array(success_F)
                success_CR = np.array(success_CR)
                diff_fitness = np.array(diff_fitness)
                
                # Weights based on fitness improvement
                total_diff = np.sum(diff_fitness)
                if total_diff > 0:
                    weights = diff_fitness / total_diff
                    
                    # Lehmer mean for F
                    mean_f_lehmer = np.sum(weights * (success_F ** 2)) / np.sum(weights * success_F)
                    M_F[k_mem] = mean_f_lehmer
                    
                    # Weighted mean for CR
                    mean_cr = np.sum(weights * success_CR)
                    M_CR[k_mem] = mean_cr
                    
                    # Increment memory pointer
                    k_mem = (k_mem + 1) % H_memory_size

            # 7. Check Convergence / Restart
            # Calculate population diversity
            std_dev = np.mean(np.std(pop, axis=0))
            current_range = np.mean(np.max(pop, axis=0) - np.min(pop, axis=0))
            
            if not improved_any:
                stagnation_counter += 1
            
            # Restart triggers:
            # 1. Very low diversity
            # 2. Too many generations without any local improvement (stagnation)
            # 3. Small range relative to domain
            domain_range = np.mean(diff_b)
            
            if stagnation_counter > 25 or std_dev < 1e-8 or current_range < 1e-8 * domain_range:
                # --- Polishing Phase ---
                # Before destroying the population, perform a quick Local Search (Coordinate Descent)
                # on the best individual found in this restart.
                best_idx = np.argmin(fitness)
                polish_x = pop[best_idx].copy()
                polish_f = fitness[best_idx]
                
                # Simple adaptive coordinate descent
                step_size = current_range * 0.5 if current_range > 1e-12 else 0.01 * domain_range
                
                # Limited budget for polishing
                polish_iter = 0
                max_polish_iter = 2 * dim 
                
                while step_size > 1e-9 and polish_iter < max_polish_iter:
                    improved_polish = False
                    for d in range(dim):
                        if (time.time() - start_time) >= max_time:
                            return global_best_fitness
                        
                        # Try positive step
                        old_val = polish_x[d]
                        polish_x[d] = np.clip(old_val + step_size, lower_b[d], upper_b[d])
                        f_new = func(polish_x)
                        
                        if f_new < polish_f:
                            polish_f = f_new
                            if polish_f < global_best_fitness:
                                global_best_fitness = polish_f
                            improved_polish = True
                        else:
                            # Try negative step
                            polish_x[d] = np.clip(old_val - step_size, lower_b[d], upper_b[d])
                            f_new = func(polish_x)
                            if f_new < polish_f:
                                polish_f = f_new
                                if polish_f < global_best_fitness:
                                    global_best_fitness = polish_f
                                improved_polish = True
                            else:
                                # Revert
                                polish_x[d] = old_val
                        polish_iter += 1
                    
                    if not improved_polish:
                        step_size /= 2.0
                
                # Break inner loop to trigger restart
                break

    return global_best_fitness
