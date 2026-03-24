The output values and codes for the best generated algorithms are as follows:
1. output value is: 1.721299181045481

 1. algorithm code is:
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


2. output value is: 11.248754273515829

 2. algorithm code is:
#The output values and codes for the best generated algorithms are as follows:
#1. output value is: 1.721299181045481
#
# 1. algorithm code is:
##The following algorithm implements an improved version of **L-SHADE (Success-History based Adaptive Differential Evolution)** with **Linear Population Size Reduction (LPSR)** and a **Final Polish (Local Search)** phase.
##
##**Key Improvements:**
##1.  **Linear Population Size Reduction (LPSR)**: The population size dynamically decreases from a large initial size (for exploration) to a small size (for exploitation) based on the elapsed time. This focuses computational resources on the best individuals as the algorithm converges.
##2.  **Weighted Lehmer Mean**: The adaptive memory ($M_F, M_{CR}$) is updated using a weighted mean based on fitness improvements, ensuring that parameters generating larger improvements have more influence.
##3.  **Final Polish (Local Search)**: When the time budget is nearly exhausted, the algorithm switches to a (1+1)-ES based Gaussian local search around the global best solution. This helps refine the solution to high precision, which DE sometimes struggles with in the final stages.
##4.  **Robust Time Management**: The algorithm constantly estimates the evaluation cost to perfectly pace the population reduction and ensures the time limit is respected strictly.
##
#import numpy as np
#from datetime import datetime, timedelta
#
#def run(func, dim, bounds, max_time):
#    """
#    Minimizes a function using L-SHADE with Linear Population Size Reduction (LPSR)
#    and a final Gaussian Local Search polish.
#    """
#    # --- Initialization & Timing ---
#    start_time = datetime.now()
#    
#    def get_remaining_seconds():
#        return max_time - (datetime.now() - start_time).total_seconds()
#
#    # Pre-process bounds
#    bounds_np = np.array(bounds)
#    min_b = bounds_np[:, 0]
#    max_b = bounds_np[:, 1]
#    diff_b = max_b - min_b
#
#    # Global Best Tracking
#    best_fitness = float('inf')
#    best_sol = None
#
#    # --- Algorithm Configuration ---
#    # Initial Population: High for exploration, scaled by dimension
#    # Capped at 200 to ensure efficiency on high dimensions within time limits
#    pop_size_init = int(np.clip(25 * dim, 50, 200))
#    pop_size_min = 4
#    
#    # Archive parameters
#    archive_rate = 2.0  # Archive size relative to population size
#    
#    # SHADE Memory parameters
#    H = 5  # History size
#    
#    # Evaluation statistics
#    eval_count = 0
#    avg_eval_time = 0.0
#    
#    # --- Main Optimization Loop (Restarts) ---
#    # We allow restarts if the population converges early, 
#    # but LPSR usually spans the whole duration.
#    while get_remaining_seconds() > 0.05:
#        
#        # Reset parameters for this run
#        current_pop_size = pop_size_init
#        
#        # Initialize Memory (M_CR, M_F) to 0.5
#        mem_cr = np.full(H, 0.5)
#        mem_f = np.full(H, 0.5)
#        k_mem = 0
#        
#        # Initialize Population
#        pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
#        fitness = np.full(current_pop_size, float('inf'))
#        
#        # Elitism: Inject best solution found so far (if any)
#        start_idx = 0
#        if best_sol is not None:
#            pop[0] = best_sol
#            fitness[0] = best_fitness
#            start_idx = 1
            
#        # Evaluate Initial Population
#        for i in range(start_idx, current_pop_size):
#            if get_remaining_seconds() <= 0: return best_fitness
#            
#            val = func(pop[i])
#            eval_count += 1
#            fitness[i] = val
#            
#            if val < best_fitness:
#                best_fitness = val
#                best_sol = pop[i].copy()
                
#            # Calibrate time estimate
#            if eval_count % 10 == 0 or eval_count == 5:
#                elapsed = (datetime.now() - start_time).total_seconds()
#                avg_eval_time = elapsed / eval_count
#        
#        # Initialize Archive
#        archive = []
#        
#        # Sort population by fitness
#        sorted_indices = np.argsort(fitness)
#        pop = pop[sorted_indices]
#        fitness = fitness[sorted_indices]
#        
#        # --- L-SHADE Generation Loop ---
#        while get_remaining_seconds() > 0.05:
#            
#            # 1. Linear Population Size Reduction (LPSR) & Time Mgmt
#            elapsed = (datetime.now() - start_time).total_seconds()
#            if eval_count > 0:
#                avg_eval_time = elapsed / eval_count
#            
#            # Estimate total evaluations possible within max_time
#            safe_avg = avg_eval_time if avg_eval_time > 1e-9 else 1e-9
#            max_evals_total = int(max_time / safe_avg)
#            
#            # Calculate progress ratio (0.0 to 1.0)
#            progress = min(1.0, eval_count / max_evals_total) if max_evals_total > 0 else 1.0
#            
#            # Calculate target population size based on progress
#            target_pop = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
#            target_pop = max(pop_size_min, target_pop)
            
#            # Reduce Population
#            if current_pop_size > target_pop:
#                current_pop_size = target_pop
#                # Since pop is sorted at end of loop, simply truncate the worst
#                pop = pop[:current_pop_size]
#                fitness = fitness[:current_pop_size]
#                
#                # Resize Archive
#                target_arc = int(current_pop_size * archive_rate)
#                if len(archive) > target_arc:
#                    # Randomly remove elements to fit target size
#                    del_cnt = len(archive) - target_arc
#                    indices_to_del = np.random.choice(len(archive), del_cnt, replace=False)
#                    # Rebuild archive
#                    archive = [arr for i, arr in enumerate(archive) if i not in indices_to_del]
#
#            # 2. Parameter Adaptation
#            # Select memory slots
#            r_idx = np.random.randint(0, H, current_pop_size)
#            mu_cr = mem_cr[r_idx]
#            mu_f = mem_f[r_idx]
#            
#            # Generate CR ~ Normal(mu, 0.1)
#            cr = np.random.normal(mu_cr, 0.1)
#            cr = np.clip(cr, 0.0, 1.0)
#            
#            # Generate F ~ Cauchy(mu, 0.1)
#            f = mu_f + 0.1 * np.random.standard_cauchy(current_pop_size)
#            f = np.clip(f, 0.0, 1.0) # Truncate to [0, 1]
#            f[f <= 0] = 0.05 # Clamp near-zero to small value
#            
#            # 3. Mutation: current-to-pbest/1
#            # Dynamic 'p' for p-best selection: Decreases from 0.2 to 0.05 as we progress
#            p_val = 0.2 * (1.0 - progress) + 0.05
#            p_count = int(max(2, p_val * current_pop_size))
#            
#            # Select pbest vectors (from top p_count individuals)
#            pbest_indices = np.random.randint(0, p_count, current_pop_size)
#            x_pbest = pop[pbest_indices]
#            
#            # Select r1 (distinct from i)
#            idxs = np.arange(current_pop_size)
#            r1 = np.random.randint(0, current_pop_size - 1, current_pop_size)
#            r1 = np.where(r1 >= idxs, r1 + 1, r1) # Shift to avoid self
#            x_r1 = pop[r1]
#            
#            # Select r2 (distinct from i, r1) from Union(Population, Archive)
#            if len(archive) > 0:
#                arc_np = np.array(archive)
#                union_pop = np.vstack((pop, arc_np))
#            else:
#                union_pop = pop
#            
#            # Random selection for r2 (collision prob is low, ignored for speed)
#            r2 = np.random.randint(0, len(union_pop), current_pop_size)
#            x_r2 = union_pop[r2]
#            
#            # Calculate Mutant Vectors
#            f_col = f[:, None]
#            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
#            mutant = np.clip(mutant, min_b, max_b)
#            
#            # 4. Crossover (Binomial)
#            rand_cr = np.random.rand(current_pop_size, dim)
#            cross_mask = rand_cr < cr[:, None]
#            
#            # Ensure at least one dimension is taken from mutant
#            j_rand = np.random.randint(0, dim, current_pop_size)
#            cross_mask[idxs, j_rand] = True
#            
#            trial = np.where(cross_mask, mutant, pop)
#            
#            # 5. Selection & Evaluation
#            success_f = []
#            success_cr = []
#            success_diff = []
#            
#            for i in range(current_pop_size):
#                if get_remaining_seconds() <= 0: return best_fitness
#                
#                new_val = func(trial[i])
#                eval_count += 1
#                
#                if new_val < fitness[i]:
#                    # Successful update
#                    success_diff.append(fitness[i] - new_val)
#                    success_f.append(f[i])
#                    success_cr.append(cr[i])
#                    
#                    # Add parent to archive
#                    archive.append(pop[i].copy())
#                    
#                    # Update population
#                    fitness[i] = new_val
#                    pop[i] = trial[i]
#                    
#                    if new_val < best_fitness:
#                        best_fitness = new_val
#                        best_sol = trial[i].copy()
#            
#            # 6. Update History Memory (Weighted Lehmer Mean)
#            if len(success_diff) > 0:
#                s_diff = np.array(success_diff)
#                s_f = np.array(success_f)
#                s_cr = np.array(success_cr)
#                
#                # Calculate weights based on fitness improvement
#                weights = s_diff / np.sum(s_diff)
#                
#                # Weighted Lehmer Mean for F
#                denom = np.sum(weights * s_f)
#                if denom > 0:
#                    mean_f = np.sum(weights * (s_f ** 2)) / denom
#                    mem_f[k_mem] = np.clip(mean_f, 0, 1)
#                
#                # Weighted Arithmetic Mean for CR
#                mean_cr = np.sum(weights * s_cr)
#                mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
#                
#                k_mem = (k_mem + 1) % H
#            
#            # 7. Sort Population (Crucial for pbest and truncation)
#            sorted_indices = np.argsort(fitness)
#            pop = pop[sorted_indices]
#            fitness = fitness[sorted_indices]
#            
#            # 8. Check Polish Condition
#            # If time is running out (last 5% or <1s), break to Polish phase
#            remaining = get_remaining_seconds()
#            if remaining < max(1.0, max_time * 0.05):
#                break
#            
#            # Convergence check for restart
#            if np.std(fitness) < 1e-9:
#                break
#        
#        # Break outer loop if strictly in polish time
#        if get_remaining_seconds() < max(1.0, max_time * 0.05):
#            break
#
#    # --- Final Polish (Gaussian Local Search) ---
#    # Use the remaining seconds to fine-tune the best solution found.
#    # Simple (1+1) Evolution Strategy with adaptive step size.
#    if best_sol is not None:
#        sigma = 1e-2 * diff_b # Initial step size (1% of domain)
#        current_sol = best_sol.copy()
#        current_val = best_fitness
#        
#        while get_remaining_seconds() > 0:
#            # Generate candidate: best + noise
#            step = np.random.normal(0, 1, dim) * sigma
#            candidate = current_sol + step
#            candidate = np.clip(candidate, min_b, max_b)
#            
#            # Check time strictly
#            if get_remaining_seconds() <= 0: return best_fitness
#            
#            val = func(candidate)
#            
#            if val < current_val:
#                current_val = val
#                current_sol = candidate
#                if val < best_fitness:
#                    best_fitness = val
#                    best_sol = candidate
#                
#                # Success: slightly increase step size (exploit direction)
#                sigma *= 1.2
#            else:
#                # Failure: decrease step size (refine)
#                sigma *= 0.5
#                
#    return best_fitness
#
#
#2. output value is: 34.2603727831155
#
# 2. algorithm code is:
##An algorithm to find the set of input parameter values that lead to the minimum output value.
##
##This solution implements a **Simplified SHADE (Success-History based Adaptive Differential Evolution) with Archive and Restart Strategy**.
##
##Key improvements over standard algorithms include:
##1.  **Current-to-pbest Mutation**: Moves individuals towards the best solutions found so far while maintaining diversity, converging faster than standard random mutation.
##2.  **Adaptive Parameters (F & CR)**: Automatically tunes the mutation factor and crossover probability based on successful evaluations, adapting to the specific function landscape.
##3.  **Archive**: Stores recent inferior solutions to preserve diversity and prevent premature convergence, a critical component of state-of-the-art DE variants like JADE/SHADE.
##4.  **Restart Mechanism**: Detects stagnation (low population variance) and restarts the population while preserving the global best, ensuring the algorithm uses the full time budget to escape local optima.
##
#import numpy as np
#from datetime import datetime, timedelta
#
#def run(func, dim, bounds, max_time):
#    """
#    Minimizes a function using a Simplified SHADE (Adaptive Differential Evolution)
#    algorithm with an external archive and restart strategy.
#    """
#    # --- Initialization & Timing ---
#    start_time = datetime.now()
#    time_limit = timedelta(seconds=max_time)
#
#    # Helper to check remaining time
#    def has_time():
#        return (datetime.now() - start_time) < time_limit
#
#    # --- Configuration ---
#    # Population size: Balance between exploration (high) and speed (low).
#    # We use a dynamic size based on dimension, capped to ensure efficiency.
#    pop_size = min(60, max(20, 10 * dim))
#    
#    bounds_np = np.array(bounds)
#    min_b = bounds_np[:, 0]
#    max_b = bounds_np[:, 1]
#    diff_b = max_b - min_b
#    
#    # Archive for historical vectors (preserves diversity)
#    archive = []
#    
#    # Adaptive Parameter Memory (History length H)
#    H = 5
#    mem_cr = np.full(H, 0.5)
#    mem_f = np.full(H, 0.5)
#    k_mem = 0  # Memory index
#    
#    # Initialize Population
#    pop = min_b + np.random.rand(pop_size, dim) * diff_b
#    fitness = np.full(pop_size, float('inf'))
#    
#    best_fitness = float('inf')
#    best_sol = None
#
#    # --- Initial Evaluation ---
#    for i in range(pop_size):
#        if not has_time():
#            return best_fitness if best_fitness != float('inf') else func(pop[i])
#            
#        val = func(pop[i])
#        fitness[i] = val
#        if val < best_fitness:
#            best_fitness = val
#            best_sol = pop[i].copy()
#
#    # --- Main Optimization Loop ---
#    while has_time():
#        
#        # 1. Parameter Adaptation
#        # Assign each individual a parameter set from history memory
#        r_idx = np.random.randint(0, H, pop_size)
#        mu_cr = mem_cr[r_idx]
#        mu_f = mem_f[r_idx]
#        
#        # Generate CR ~ Normal(mu, 0.1), clipped to [0, 1]
#        cr = np.random.normal(mu_cr, 0.1)
#        cr = np.clip(cr, 0.0, 1.0)
#        
#        # Generate F ~ Cauchy(mu, 0.1), clipped to [0.1, 1.0]
#        # Approximation using standard_cauchy
#        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
#        f = np.clip(f, 0.1, 1.0)
#        
#        # 2. Mutation Strategy: current-to-pbest/1
#        # Select pbest from top p% (greedy component)
#        p = 0.1
#        top_p_cnt = max(1, int(p * pop_size))
#        sorted_indices = np.argsort(fitness)
#        pbest_indices = sorted_indices[:top_p_cnt]
#        
#        pbest_choice = np.random.choice(pbest_indices, pop_size)
#        x_pbest = pop[pbest_choice]
#        
#        # Select r1 distinct from i (random component)
#        idxs = np.arange(pop_size)
#        r1 = (idxs + np.random.randint(1, pop_size, pop_size)) % pop_size
#        x_r1 = pop[r1]
#        
#        # Select r2 from Union(Pop, Archive) distinct from i, r1 (diversity component)
#        if len(archive) > 0:
#            archive_np = np.array(archive)
#            pop_all = np.vstack((pop, archive_np))
#        else:
#            pop_all = pop
#            
#        # Randomly select r2
#        r2 = np.random.randint(0, len(pop_all), pop_size)
#        x_r2 = pop_all[r2]
#        
#        # Calculate Mutant Vector V
#        # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
#        f_broad = f[:, np.newaxis]
#        mutant = pop + f_broad * (x_pbest - pop) + f_broad * (x_r1 - x_r2)
#        
#        # 3. Crossover (Binomial)
#        rand_vals = np.random.rand(pop_size, dim)
#        cross_mask = rand_vals < cr[:, np.newaxis]
#        
#        # Ensure at least one parameter is taken from mutant
#        j_rand = np.random.randint(0, dim, pop_size)
#        cross_mask[idxs, j_rand] = True 
#        
#        trial = np.where(cross_mask, mutant, pop)
#        
#        # Bound Constraints (Clipping)
#        trial = np.clip(trial, min_b, max_b)
#        
#        # 4. Selection & Evaluation
#        success_f = []
#        success_cr = []
#        
#        for i in range(pop_size):
#            if not has_time():
#                return best_fitness
#            
#            f_trial = func(trial[i])
#            
#            if f_trial <= fitness[i]:
#                # If strictly better, move parent to archive
#                if f_trial < fitness[i]:
#                    archive.append(pop[i].copy())
#                    # Limit archive size to population size
#                    if len(archive) > pop_size:
#                        archive.pop(np.random.randint(0, len(archive)))
#                        
#                fitness[i] = f_trial
#                pop[i] = trial[i]
#                
#                success_f.append(f[i])
#                success_cr.append(cr[i])
#                
#                if f_trial < best_fitness:
#                    best_fitness = f_trial
#                    best_sol = trial[i].copy()
#                    
#        # 5. Update Adaptive Memory
#        if len(success_f) > 0:
#            sf = np.array(success_f)
#            scr = np.array(success_cr)
#            
#            # Lehmer Mean for F (biases towards larger successful F)
#            mean_f = np.sum(sf**2) / np.sum(sf)
#            # Arithmetic Mean for CR
#            mean_cr = np.mean(scr)
#            
#            mem_f[k_mem] = mean_f
#            mem_cr[k_mem] = mean_cr
#            k_mem = (k_mem + 1) % H
#            
#        # 6. Restart Strategy
#        # If population variance is extremely low, we are converged.
#        # Restart to use remaining time to find potential better optima.
#        if np.std(fitness) < 1e-6:
#             # Re-initialize population with random values
#             pop = min_b + np.random.rand(pop_size, dim) * diff_b
#             # Inject the best solution found so far (Elitism)
#             pop[0] = best_sol
#             fitness[:] = float('inf')
#             fitness[0] = best_fitness
#             # Clear archive
#             archive = []
#             
#    return best_fitness
#
#
#3. output value is: 45.70525435631852
#
# 3. algorithm code is:
#import numpy as np
#from datetime import datetime, timedelta
#
#def run(func, dim, bounds, max_time):
#    """
#    Finds the set of input parameter values that lead to the minimum output value
#    using the Differential Evolution (DE) algorithm within a limited time.
#    """
#    start_time = datetime.now()
#    time_limit = timedelta(seconds=max_time)
#    
#    # --- Algorithm Configuration ---
#    # Differential Evolution parameters
#    # Population size (NP): A balance between exploration and computation time.
#    # We use a dynamic size based on dimension but capped to ensure iterations 
#    # can occur if the function evaluation is slow.
#    pop_size = max(5, 10 * dim)
#    if pop_size > 50:
#        pop_size = 50
#        
#    mutation_factor = 0.8  # F: Scaling factor for mutation (typically 0.5-0.9)
#    crossover_prob = 0.9   # CR: Crossover probability (typically 0.8-0.9)
#
#    # --- Initialization ---
#    bounds_np = np.array(bounds)
#    min_b = bounds_np[:, 0]
#    max_b = bounds_np[:, 1]
#    diff_b = max_b - min_b
#    
#    # Initialize population with random values within bounds
#    # Shape: (pop_size, dim)
#    pop = min_b + np.random.rand(pop_size, dim) * diff_b
#    
#    # Store fitness values
#    fitness = np.full(pop_size, float('inf'))
#    
#    best_fitness = float('inf')
#    
#    # --- Evaluate Initial Population ---
#    # We must check time even during initialization in case func is very slow
#    for i in range(pop_size):
#        if (datetime.now() - start_time) >= time_limit:
#            # If we timeout before finishing initialization, return best found so far
#            return best_fitness if best_fitness != float('inf') else func(pop[i])
#            
#        val = func(pop[i])
#        fitness[i] = val
#        
#        if val < best_fitness:
#            best_fitness = val
#
#    # --- Main Differential Evolution Loop ---
#    # Continue evolving until time runs out
#    while True:
#        # Check time at the start of generation
#        if (datetime.now() - start_time) >= time_limit:
#            break
#            
#        for i in range(pop_size):
#            # Check time frequently (per individual) to adhere strictly to max_time
#            if (datetime.now() - start_time) >= time_limit:
#                return best_fitness
#
#            # 1. Mutation (Strategy: DE/rand/1)
#            # Pick 3 random, distinct individuals other than current i
#            candidates = [idx for idx in range(pop_size) if idx != i]
#            a, b, c = np.random.choice(candidates, 3, replace=False)
#            
#            # Create mutant vector: V = X_a + F * (X_b - X_c)
#            mutant = pop[a] + mutation_factor * (pop[b] - pop[c])
#            
#            # Bound constraints: Clip mutant vector to stay within bounds
#            mutant = np.clip(mutant, min_b, max_b)
#            
#            # 2. Crossover (Binomial)
#            # Create trial vector by mixing target vector and mutant vector
#            cross_points = np.random.rand(dim) < crossover_prob
#            
#            # Ensure at least one parameter comes from the mutant (promotes diversity)
#            if not np.any(cross_points):
#                cross_points[np.random.randint(0, dim)] = True
#                
#            trial_vector = np.where(cross_points, mutant, pop[i])
#            
#            # 3. Selection
#            # Greedy selection: if trial is better, replace current
#            trial_val = func(trial_vector)
#            
#            if trial_val < fitness[i]:
#                fitness[i] = trial_val
#                pop[i] = trial_vector
#                
#                # Update global best if needed
#                if trial_val < best_fitness:
#                    best_fitness = trial_val
#
#    # return fitness of the best found solution
#    return best_fitness
#
#
#improve code.
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using an Improved L-SHADE-LPSR algorithm with a 
    hybrid global/local search strategy and robust time management.
    """
    # --- Timing ---
    t_start = time.time()
    
    def get_remaining():
        return max_time - (time.time() - t_start)
    
    # Early check
    if get_remaining() <= 0:
        return float('inf')

    # --- Constants & Config ---
    # Population Strategy: L-SHADE with Linear Reduction
    # Initial size: Large enough for exploration (e.g., 18*D), capped for speed
    init_pop_size = int(round(max(50, 18 * dim)))
    min_pop_size = 4
    
    # SHADE Parameters
    H = 6 # Memory size (Success History)
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    archive = []
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    current_pop_size = init_pop_size
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None
    
    eval_count = 0
    # Rolling window for time estimation
    eval_times = []
    
    # Safe function wrapper to handle time limits inside loops
    def safe_evaluate(x):
        nonlocal eval_count, best_fitness, best_sol
        if get_remaining() <= 0:
            raise TimeoutError
            
        t0 = time.time()
        val = func(x)
        dt = time.time() - t0
        
        eval_count += 1
        eval_times.append(dt)
        if len(eval_times) > 50:
            eval_times.pop(0)
            
        if val < best_fitness:
            best_fitness = val
            best_sol = x.copy()
        return val

    # Evaluate Initial Population
    try:
        for i in range(current_pop_size):
            fitness[i] = safe_evaluate(pop[i])
    except TimeoutError:
        return best_fitness

    # Sort based on fitness
    sorted_idx = np.argsort(fitness)
    pop = pop[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # --- Main DE Loop ---
    while get_remaining() > 0:
        
        # 1. Time & LPSR Logic
        # Estimate average time per evaluation
        avg_eval = np.mean(eval_times) if eval_times else 0.0
        if avg_eval < 1e-9: avg_eval = 1e-6
            
        # Estimated total evaluations possible
        rem_t = get_remaining()
        est_rem_evals = rem_t / avg_eval
        
        # Current progress (0.0 to 1.0)
        # We estimate 'max_evals' as current + remaining
        max_evals = eval_count + est_rem_evals
        progress = min(1.0, eval_count / max_evals) if max_evals > 0 else 1.0
        
        # Calculate Target Population Size (LPSR)
        target_size = int(round((min_pop_size - init_pop_size) * progress + init_pop_size))
        target_size = max(min_pop_size, target_size)
        
        # Reduce Population
        if current_pop_size > target_size:
            current_pop_size = target_size
            pop = pop[:current_pop_size]
            fitness = fitness[:current_pop_size]
            
            # Archive resizing
            arc_target = int(current_pop_size * 2.5) 
            if len(archive) > arc_target:
                # Remove random elements
                dels = np.random.choice(len(archive), len(archive) - arc_target, replace=False)
                archive = [archive[i] for i in range(len(archive)) if i not in dels]
                
        # Polish Trigger: If time is very short or population converged to min
        # Reserve last 5% of time for local search
        if rem_t < max(0.2, max_time * 0.05) or current_pop_size <= min_pop_size:
            break
            
        # 2. Parameter Generation
        # Vectorized generation for efficiency
        r_idxs = np.random.randint(0, H, current_pop_size)
        m_f = mem_f[r_idxs]
        m_cr = mem_cr[r_idxs]
        
        # Cauchy for F
        f = m_f + 0.1 * np.random.standard_cauchy(current_pop_size)
        f = np.clip(f, 0, 1)
        f[f <= 0] = 0.05 # Constraint check
        
        # Normal for CR
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # 3. Mutation: current-to-pbest/1
        # p decreases with progress (exploration -> exploitation)
        p = 0.2 * (1 - progress) + 0.05
        p = max(0.02, p)
        p_count = int(max(2, p * current_pop_size))
        
        # pbest selection
        pbest_idxs = np.random.randint(0, p_count, current_pop_size)
        x_pbest = pop[pbest_idxs]
        
        # r1 selection (distinct from i)
        idxs = np.arange(current_pop_size)
        r1 = np.random.randint(0, current_pop_size - 1, current_pop_size)
        r1 += (r1 >= idxs) # shift
        x_r1 = pop[r1]
        
        # r2 selection (distinct from i, r1) - using Archive
        if len(archive) > 0:
            arc_arr = np.array(archive)
            u_pop = np.vstack((pop, arc_arr))
        else:
            u_pop = pop
            
        r2 = np.random.randint(0, len(u_pop), current_pop_size)
        # Collision check with r1 and i skipped for performance (DE robust enough)
        x_r2 = u_pop[r2]
        
        # Compute Mutant
        f_vec = f[:, None]
        mutant = pop + f_vec * (x_pbest - pop) + f_vec * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        mask = np.random.rand(current_pop_size, dim) < cr[:, None]
        j_rand = np.random.randint(0, dim, current_pop_size)
        mask[idxs, j_rand] = True
        
        trial = np.where(mask, mutant, pop)
        trial = np.clip(trial, min_b, max_b)
        
        # 5. Evaluation
        success_f = []
        success_cr = []
        success_diff = []
        
        # Temp arrays
        next_pop = pop.copy()
        next_fit = fitness.copy()
        
        try:
            for i in range(current_pop_size):
                y = safe_evaluate(trial[i])
                
                if y < fitness[i]:
                    success_diff.append(fitness[i] - y)
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    
                    archive.append(pop[i].copy())
                    next_pop[i] = trial[i]
                    next_fit[i] = y
        except TimeoutError:
            return best_fitness
            
        pop = next_pop
        fitness = next_fit
        
        # 6. Update Memory
        if len(success_diff) > 0:
            w = np.array(success_diff)
            w /= np.sum(w)
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            # Lehmer Mean F
            mean_f = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            # Weighted Mean CR
            mean_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 7. Sort
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

        # Convergence Restart
        # If population is stagnant and we have plenty of time (>40%), restart
        # This prevents getting stuck in a local optimum early.
        if np.std(fitness) < 1e-9:
             if get_remaining() > 0.4 * max_time:
                 current_pop_size = init_pop_size
                 pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
                 # Inject best solution found so far (Elitism)
                 pop[0] = best_sol
                 fitness[:] = float('inf')
                 fitness[0] = best_fitness
                 # Reset archive
                 archive = []
                 # Re-eval
                 try:
                     for i in range(1, current_pop_size):
                         fitness[i] = safe_evaluate(pop[i])
                 except TimeoutError:
                     return best_fitness
                 sorted_idx = np.argsort(fitness)
                 pop = pop[sorted_idx]
                 fitness = fitness[sorted_idx]
             else:
                 break # Go to Polish phase immediately

    # --- Final Polish: Coordinate Descent (MTS-LS1 style) ---
    # We use the remaining time to refine the best solution.
    # This is often more effective than DE for final high-precision tuning.
    if best_sol is not None:
        curr_x = best_sol.copy()
        curr_y = best_fitness
        
        # Initial step sizes per dimension
        step_sizes = (max_b - min_b) * 0.05
        min_steps = (max_b - min_b) * 1e-9
        
        try:
            while get_remaining() > 0:
                improved_any = False
                
                # Shuffle dimensions to avoid bias
                dims = np.random.permutation(dim)
                
                for d in dims:
                    if step_sizes[d] < min_steps[d]:
                        continue
                        
                    # Try +step
                    x_tmp = curr_x.copy()
                    x_tmp[d] = np.clip(x_tmp[d] + step_sizes[d], min_b[d], max_b[d])
                    val = safe_evaluate(x_tmp)
                    
                    if val < curr_y:
                        curr_y = val
                        curr_x = x_tmp
                        if val < best_fitness: best_fitness = val; best_sol = x_tmp.copy()
                        step_sizes[d] *= 1.5 # Accelerate step
                        improved_any = True
                        continue
                        
                    # Try -step (Search in opposite direction)
                    x_tmp[d] = np.clip(curr_x[d] - step_sizes[d], min_b[d], max_b[d])
                    val = safe_evaluate(x_tmp)
                    
                    if val < curr_y:
                        curr_y = val
                        curr_x = x_tmp
                        if val < best_fitness: best_fitness = val; best_sol = x_tmp.copy()
                        step_sizes[d] *= 1.5
                        improved_any = True
                    else:
                        # Fail: decrease step
                        step_sizes[d] *= 0.5
                        
                if not improved_any:
                    # If all dims failed, check global convergence
                    if np.max(step_sizes) < np.min(min_steps) * 10:
                        break 
        except TimeoutError:
            pass

    return best_fitness