#The following algorithm implements an improved version of **L-SHADE (Success-History based Adaptive Differential Evolution)** with **Linear Population Size Reduction (LPSR)** and a **Final Polish (Local Search)** phase.
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR)**: The population size dynamically decreases from a large initial size (for exploration) to a small size (for exploitation) based on the elapsed time. This focuses computational resources on the best individuals as the algorithm converges.
#2.  **Weighted Lehmer Mean**: The adaptive memory ($M_F, M_{CR}$) is updated using a weighted mean based on fitness improvements, ensuring that parameters generating larger improvements have more influence.
#3.  **Final Polish (Local Search)**: When the time budget is nearly exhausted, the algorithm switches to a (1+1)-ES based Gaussian local search around the global best solution. This helps refine the solution to high precision, which DE sometimes struggles with in the final stages.
#4.  **Robust Time Management**: The algorithm constantly estimates the evaluation cost to perfectly pace the population reduction and ensures the time limit is respected strictly.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Linear Population Size Reduction (LPSR)
    and a final Gaussian Local Search polish.
    """
    # --- Initialization & Timing ---
    start_time = datetime.now()
    
    def get_remaining_seconds():
        return max_time - (datetime.now() - start_time).total_seconds()

    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Global Best Tracking
    best_fitness = float('inf')
    best_sol = None

    # --- Algorithm Configuration ---
    # Initial Population: High for exploration, scaled by dimension
    # Capped at 200 to ensure efficiency on high dimensions within time limits
    pop_size_init = int(np.clip(25 * dim, 50, 200))
    pop_size_min = 4
    
    # Archive parameters
    archive_rate = 2.0  # Archive size relative to population size
    
    # SHADE Memory parameters
    H = 5  # History size
    
    # Evaluation statistics
    eval_count = 0
    avg_eval_time = 0.0
    
    # --- Main Optimization Loop (Restarts) ---
    # We allow restarts if the population converges early, 
    # but LPSR usually spans the whole duration.
    while get_remaining_seconds() > 0.05:
        
        # Reset parameters for this run
        current_pop_size = pop_size_init
        
        # Initialize Memory (M_CR, M_F) to 0.5
        mem_cr = np.full(H, 0.5)
        mem_f = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Population
        pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
        fitness = np.full(current_pop_size, float('inf'))
        
        # Elitism: Inject best solution found so far (if any)
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol
            fitness[0] = best_fitness
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, current_pop_size):
            if get_remaining_seconds() <= 0: return best_fitness
            
            val = func(pop[i])
            eval_count += 1
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_sol = pop[i].copy()
                
            # Calibrate time estimate
            if eval_count % 10 == 0 or eval_count == 5:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_eval_time = elapsed / eval_count
        
        # Initialize Archive
        archive = []
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # --- L-SHADE Generation Loop ---
        while get_remaining_seconds() > 0.05:
            
            # 1. Linear Population Size Reduction (LPSR) & Time Mgmt
            elapsed = (datetime.now() - start_time).total_seconds()
            if eval_count > 0:
                avg_eval_time = elapsed / eval_count
            
            # Estimate total evaluations possible within max_time
            safe_avg = avg_eval_time if avg_eval_time > 1e-9 else 1e-9
            max_evals_total = int(max_time / safe_avg)
            
            # Calculate progress ratio (0.0 to 1.0)
            progress = min(1.0, eval_count / max_evals_total) if max_evals_total > 0 else 1.0
            
            # Calculate target population size based on progress
            target_pop = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            target_pop = max(pop_size_min, target_pop)
            
            # Reduce Population
            if current_pop_size > target_pop:
                current_pop_size = target_pop
                # Since pop is sorted at end of loop, simply truncate the worst
                pop = pop[:current_pop_size]
                fitness = fitness[:current_pop_size]
                
                # Resize Archive
                target_arc = int(current_pop_size * archive_rate)
                if len(archive) > target_arc:
                    # Randomly remove elements to fit target size
                    del_cnt = len(archive) - target_arc
                    indices_to_del = np.random.choice(len(archive), del_cnt, replace=False)
                    # Rebuild archive
                    archive = [arr for i, arr in enumerate(archive) if i not in indices_to_del]

            # 2. Parameter Adaptation
            # Select memory slots
            r_idx = np.random.randint(0, H, current_pop_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]
            
            # Generate CR ~ Normal(mu, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # Generate F ~ Cauchy(mu, 0.1)
            f = mu_f + 0.1 * np.random.standard_cauchy(current_pop_size)
            f = np.clip(f, 0.0, 1.0) # Truncate to [0, 1]
            f[f <= 0] = 0.05 # Clamp near-zero to small value
            
            # 3. Mutation: current-to-pbest/1
            # Dynamic 'p' for p-best selection: Decreases from 0.2 to 0.05 as we progress
            p_val = 0.2 * (1.0 - progress) + 0.05
            p_count = int(max(2, p_val * current_pop_size))
            
            # Select pbest vectors (from top p_count individuals)
            pbest_indices = np.random.randint(0, p_count, current_pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1 (distinct from i)
            idxs = np.arange(current_pop_size)
            r1 = np.random.randint(0, current_pop_size - 1, current_pop_size)
            r1 = np.where(r1 >= idxs, r1 + 1, r1) # Shift to avoid self
            x_r1 = pop[r1]
            
            # Select r2 (distinct from i, r1) from Union(Population, Archive)
            if len(archive) > 0:
                arc_np = np.array(archive)
                union_pop = np.vstack((pop, arc_np))
            else:
                union_pop = pop
            
            # Random selection for r2 (collision prob is low, ignored for speed)
            r2 = np.random.randint(0, len(union_pop), current_pop_size)
            x_r2 = union_pop[r2]
            
            # Calculate Mutant Vectors
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            rand_cr = np.random.rand(current_pop_size, dim)
            cross_mask = rand_cr < cr[:, None]
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, current_pop_size)
            cross_mask[idxs, j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 5. Selection & Evaluation
            success_f = []
            success_cr = []
            success_diff = []
            
            for i in range(current_pop_size):
                if get_remaining_seconds() <= 0: return best_fitness
                
                new_val = func(trial[i])
                eval_count += 1
                
                if new_val < fitness[i]:
                    # Successful update
                    success_diff.append(fitness[i] - new_val)
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    
                    # Update population
                    fitness[i] = new_val
                    pop[i] = trial[i]
                    
                    if new_val < best_fitness:
                        best_fitness = new_val
                        best_sol = trial[i].copy()
            
            # 6. Update History Memory (Weighted Lehmer Mean)
            if len(success_diff) > 0:
                s_diff = np.array(success_diff)
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                
                # Calculate weights based on fitness improvement
                weights = s_diff / np.sum(s_diff)
                
                # Weighted Lehmer Mean for F
                denom = np.sum(weights * s_f)
                if denom > 0:
                    mean_f = np.sum(weights * (s_f ** 2)) / denom
                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
                
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(weights * s_cr)
                mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                
                k_mem = (k_mem + 1) % H
            
            # 7. Sort Population (Crucial for pbest and truncation)
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # 8. Check Polish Condition
            # If time is running out (last 5% or <1s), break to Polish phase
            remaining = get_remaining_seconds()
            if remaining < max(1.0, max_time * 0.05):
                break
            
            # Convergence check for restart
            if np.std(fitness) < 1e-9:
                break
        
        # Break outer loop if strictly in polish time
        if get_remaining_seconds() < max(1.0, max_time * 0.05):
            break

    # --- Final Polish (Gaussian Local Search) ---
    # Use the remaining seconds to fine-tune the best solution found.
    # Simple (1+1) Evolution Strategy with adaptive step size.
    if best_sol is not None:
        sigma = 1e-2 * diff_b # Initial step size (1% of domain)
        current_sol = best_sol.copy()
        current_val = best_fitness
        
        while get_remaining_seconds() > 0:
            # Generate candidate: best + noise
            step = np.random.normal(0, 1, dim) * sigma
            candidate = current_sol + step
            candidate = np.clip(candidate, min_b, max_b)
            
            # Check time strictly
            if get_remaining_seconds() <= 0: return best_fitness
            
            val = func(candidate)
            
            if val < current_val:
                current_val = val
                current_sol = candidate
                if val < best_fitness:
                    best_fitness = val
                    best_sol = candidate
                
                # Success: slightly increase step size (exploit direction)
                sigma *= 1.2
            else:
                # Failure: decrease step size (refine)
                sigma *= 0.5
                
    return best_fitness
