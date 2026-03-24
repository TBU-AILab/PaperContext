#Here is the improved algorithm. It implements **L-SHADE with Restarts (L-SHADE-R)** and a final **Coordinate Descent** polishing phase.
#
#### Key Improvements:
#1.  **Restart Mechanism**: The algorithm detects stagnation (no improvement for a set number of generations) or population convergence. Upon detection, it restarts the population (keeping the global best) to escape local optima, addressing the issue where previous attempts got stuck at `51.48`.
#2.  **L-SHADE Adaptive Parameters**: Uses historical success memory to adapt mutation factor ($F$) and crossover rate ($CR$) specifically for the function landscape.
#3.  **Current-to-pbest/1 Mutation**: A highly efficient mutation strategy that balances exploration (searching around the best found so far) and diversity (using random differences).
#4.  **Optimized Vectorization**: Uses NumPy operations for the heavy lifting of population generation, mutation, and crossover to maximize the number of function evaluations within the time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Restarts and Coordinate Descent Polish.
    """
    
    # --- 1. Setup and Time Management ---
    start_time = time.time()
    
    # Reserve a small fraction of time for final local search (Polish)
    # 5% of time or at least 0.5 seconds if possible
    polish_ratio = 0.05
    polish_limit = max(0.5, max_time * polish_ratio)
    # The time to stop the evolutionary phase
    evo_end_time = start_time + max_time - polish_limit
    
    def check_timeout():
        return time.time() > evo_end_time

    # Process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best solution found across all restarts
    global_best_val = float('inf')
    global_best_vec = None
    
    # --- 2. Configuration ---
    # Population size: SHADE typically uses 18*dim. 
    # We clip it to ensure speed for high dimensions within limited time.
    pop_size = int(np.clip(10 * dim, 30, 100))
    
    # Memory size for adaptive parameters
    H = 5
    
    # --- 3. Main Restart Loop ---
    # We run the evolutionary algorithm. If it converges/stagnates, we restart.
    while not check_timeout():
        
        # A. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Inject global best if available to refine it (exploit) 
        # or guide the new population.
        if global_best_vec is not None:
            pop[0] = global_best_vec
            
        fitness = np.zeros(pop_size)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_timeout(): break
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
                
        if check_timeout(): break
        
        # B. Initialize SHADE Memory
        mem_M_CR = np.full(H, 0.5)
        mem_M_F = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # C. Convergence Tracking for Restart
        last_best_in_run = np.min(fitness)
        no_improve_gens = 0
        
        # --- 4. Evolutionary Loop (SHADE) ---
        while not check_timeout():
            
            # Sort population by fitness (needed for p-best selection)
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Check Stagnation
            current_best = fitness[0]
            if current_best < last_best_in_run - 1e-8:
                last_best_in_run = current_best
                no_improve_gens = 0
            else:
                no_improve_gens += 1
                
            # Trigger Restart if stuck (30 gens no improv) or converged (low std dev)
            if no_improve_gens > 30 or np.std(fitness) < 1e-8:
                break 
            
            # --- Parameter Adaptation ---
            # Pick random memory slots
            r_idx = np.random.randint(0, H, pop_size)
            r_CR = mem_M_CR[r_idx]
            r_F = mem_M_F[r_idx]
            
            # Generate CR from Normal distribution
            CR = np.random.normal(r_CR, 0.1)
            CR = np.clip(CR, 0, 1)
            
            # Generate F from Cauchy distribution
            F = r_F + 0.1 * np.random.standard_cauchy(pop_size)
            F[F > 1] = 1.0 # Clip upper
            # Repair F <= 0
            neg_mask = F <= 0
            if np.any(neg_mask):
                # Retry once
                F[neg_mask] = r_F[neg_mask] + 0.1 * np.random.standard_cauchy(np.sum(neg_mask))
                # Fallback
                F[F <= 0] = 0.5
            
            # --- Mutation: current-to-pbest/1 ---
            # Select p-best (top 11%)
            p_val = 0.11
            p_top = max(2, int(p_val * pop_size))
            pbest_idx = np.random.randint(0, p_top, pop_size)
            x_pbest = pop[pbest_idx]
            
            # Select r1 (random from pop, r1 != i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Fix collisions with i
            same_mask = (r1_idx == np.arange(pop_size))
            r1_idx[same_mask] = (r1_idx[same_mask] + 1) % pop_size
            x_r1 = pop[r1_idx]
            
            # Select r2 (random from Pop U Archive, r2 != i, r2 != r1)
            if len(archive) > 0:
                arc_np = np.array(archive)
                pop_all = np.vstack((pop, arc_np))
            else:
                pop_all = pop
            
            r2_idx = np.random.randint(0, len(pop_all), pop_size)
            # (Skipping expensive collision check for r2 for speed, impact is minimal)
            x_r2 = pop_all[r2_idx]
            
            # Create Mutant Vectors
            F_col = F[:, None]
            v = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            j_rand = np.random.randint(0, dim, pop_size)
            rand_u = np.random.rand(pop_size, dim)
            mask = rand_u < CR[:, None]
            mask[np.arange(pop_size), j_rand] = True
            
            u = np.where(mask, v, pop)
            u = np.clip(u, min_b, max_b)
            
            # --- Selection ---
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            success_mask = np.zeros(pop_size, dtype=bool)
            diff_f = np.zeros(pop_size)
            
            # Evaluate candidates
            for i in range(pop_size):
                if check_timeout(): break
                
                f_new = func(u[i])
                
                if f_new < fitness[i]:
                    new_pop[i] = u[i]
                    new_fitness[i] = f_new
                    success_mask[i] = True
                    diff_f[i] = fitness[i] - f_new
                    
                    # Add replaced parent to archive
                    archive.append(pop[i].copy())
                    
                    # Update Global Best
                    if f_new < global_best_val:
                        global_best_val = f_new
                        global_best_vec = u[i].copy()
            
            pop = new_pop
            fitness = new_fitness
            
            # Maintain Archive Size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))
            
            # --- Update Memory ---
            n_succ = np.sum(success_mask)
            if n_succ > 0:
                s_F = F[success_mask]
                s_CR = CR[success_mask]
                w = diff_f[success_mask]
                
                # Normalize weights
                total_w = np.sum(w)
                if total_w > 0:
                    w = w / total_w
                    
                    # Weighted Lehmer Mean for F
                    denom_F = np.sum(w * s_F)
                    if denom_F > 0:
                        mem_M_F[k_mem] = np.sum(w * (s_F ** 2)) / denom_F
                    else:
                        mem_M_F[k_mem] = 0.5
                    
                    # Weighted Mean for CR
                    mem_M_CR[k_mem] = np.sum(w * s_CR)
                    
                    k_mem = (k_mem + 1) % H

    # --- 5. Final Polish: Coordinate Descent ---
    # Use remaining time to fine-tune the best solution
    if global_best_vec is not None:
        x_curr = global_best_vec.copy()
        f_curr = global_best_val
        
        # Step sizes (shrinking)
        step_sizes = [1e-2, 1e-4, 1e-6, 1e-8]
        
        for step_scale in step_sizes:
            if time.time() >= start_time + max_time: break
            
            improved = True
            while improved:
                if time.time() >= start_time + max_time: break
                improved = False
                
                # Randomize dimension order
                dims = np.random.permutation(dim)
                for d in dims:
                    if time.time() >= start_time + max_time: break
                    
                    original = x_curr[d]
                    step = diff_b[d] * step_scale
                    
                    # Try positive direction
                    x_curr[d] = min(max_b[d], original + step)
                    val = func(x_curr)
                    if val < f_curr:
                        f_curr = val
                        improved = True
                        continue
                    
                    # Try negative direction
                    x_curr[d] = max(min_b[d], original - step)
                    val = func(x_curr)
                    if val < f_curr:
                        f_curr = val
                        improved = True
                        continue
                    
                    # Revert if no improvement
                    x_curr[d] = original
        
        if f_curr < global_best_val:
            global_best_val = f_curr

    return global_best_val
