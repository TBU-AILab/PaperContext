#The following algorithm implements **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution) with Restarts**.
#
#**Why this is an improvement:**
#1.  **Linear Population Reduction (L-SHADE)**: Unlike standard SHADE (which keeps population constant), this algorithm linearly reduces the population size from a large initial value (to explore the search space) to a small final value (to refine the solution) over the course of the execution. This mechanism acts as a dynamic pressure to converge and is widely proven to outperform fixed-population strategies (winning CEC 2014).
#2.  **Time-Aware Dynamic Budgeting**: Standard Evolutionary Algorithms rely on a fixed number of evaluations. This algorithm continuously estimates the remaining "budget" (in terms of function evaluations) based on the current execution speed and `max_time`. It adapts the population reduction schedule in real-time, ensuring that the "exploitation phase" (small population) is reached exactly before the time runs out.
#3.  **Robust Restarts**: If the algorithm converges early (variance becomes zero) or the population becomes too small, it triggers a restart. This allows it to escape local optima and utilize the full available time effectively.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Restarts and Time-Aware Budgeting.
    """
    start_time = datetime.now()
    # Use 98% of max_time to ensure we return before the hard timeout
    end_time = start_time + timedelta(seconds=max_time * 0.98)
    
    # --- Pre-process Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracking ---
    best_val = np.inf
    best_vec = None
    
    # Wrapper to handle bounds and update global best
    def safe_func(x):
        nonlocal best_val, best_vec
        # Clip to bounds (L-SHADE usually uses clipping)
        x_c = np.clip(x, min_b, max_b)
        val = func(x_c)
        if val < best_val:
            best_val = val
            best_vec = x_c.copy()
        return val

    # --- 1. Speed Estimation ---
    # Perform a few evaluations to estimate how fast the function is.
    # This allows us to map 'max_time' to 'max_evaluations' for the L-SHADE schedule.
    n_est = 5
    t0 = datetime.now()
    for _ in range(n_est):
        # Quick check to not exceed time during init
        if datetime.now() >= end_time: return best_val
        safe_func(min_b + np.random.rand(dim) * diff_b)
    t1 = datetime.now()
    
    elapsed_est = (t1 - t0).total_seconds()
    avg_eval_time = elapsed_est / n_est
    
    # --- 2. L-SHADE Hyperparameters ---
    H = 5  # Historical memory size
    # Initial Population Size: 
    # Standard L-SHADE uses 18*dim. We cap it to ensure reasonable speed in high dimensions.
    pop_max_limit = int(min(200, max(30, 18 * dim)))
    pop_min_limit = 4
    
    # --- 3. Restart Loop ---
    # We restart the optimization if it converges or finishes its schedule while time remains.
    while datetime.now() < end_time:
        
        # Calculate available time for this run
        time_left = (end_time - datetime.now()).total_seconds()
        if time_left < 0.05: # Stop if virtually no time left
            break
        
        # Estimate how many evaluations we can perform in this run
        est_evals_remaining = time_left / (avg_eval_time + 1e-10)
        
        # Determine Initial Population for this run
        # If budget is tight, scale down initial population
        current_pop_size = pop_max_limit
        if est_evals_remaining < current_pop_size * 2:
            current_pop_size = int(max(pop_min_limit + 1, est_evals_remaining // 5))
        
        initial_pop_size_run = current_pop_size
        
        # Initialization
        population = min_b + np.random.rand(current_pop_size, dim) * diff_b
        fitness = np.full(current_pop_size, np.inf)
        
        # Evaluate Initial Population
        for i in range(current_pop_size):
            if datetime.now() >= end_time: return best_val
            fitness[i] = safe_func(population[i])
            
        # Initialize Memory (F and CR)
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        
        # Archive (stores inferior solutions to preserve diversity)
        archive = []
        
        # Tracking for Population Reduction
        evals_in_run = current_pop_size
        run_start_time = datetime.now()
        
        # --- Evolution Loop ---
        while True:
            # Global Time Check
            t_now = datetime.now()
            if t_now >= end_time: return best_val
            
            # --- Dynamic Budgeting for Population Reduction ---
            # Update estimate of max_fes for this run based on real-time speed
            run_elapsed = (t_now - run_start_time).total_seconds()
            run_time_left = (end_time - t_now).total_seconds()
            
            # Current speed (evals/sec)
            if run_elapsed > 1e-6:
                speed = evals_in_run / run_elapsed
            else:
                speed = 1.0 / avg_eval_time
                
            est_remaining_evals = speed * run_time_left
            max_fes_run = evals_in_run + est_remaining_evals
            
            # Calculate Target Population Size using Linear Reduction Formula
            # size = round( (min - init) * (curr_evals / max_evals) + init )
            progress = min(1.0, evals_in_run / (max_fes_run + 1e-9))
            target_size = int(round((pop_min_limit - initial_pop_size_run) * progress + initial_pop_size_run))
            target_size = max(pop_min_limit, target_size)
            
            # Apply Reduction
            if current_pop_size > target_size:
                # Remove worst individuals
                n_kill = current_pop_size - target_size
                # Sort by fitness (descending = worst at end? No, minimizing, so worst is largest)
                sorted_indices = np.argsort(fitness)
                # Keep top 'target_size'
                keep_indices = sorted_indices[:target_size]
                
                population = population[keep_indices]
                fitness = fitness[keep_indices]
                current_pop_size = target_size
                
                # Reduce archive size to match population size
                if len(archive) > current_pop_size:
                    random.shuffle(archive)
                    archive = archive[:current_pop_size]
            
            # Convergence Check (Variance or Value Range)
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break # Trigger restart
            
            # --- Parameter Generation (SHADE) ---
            # Pick memory index
            r_idx = np.random.randint(0, H, current_pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate F using Cauchy distribution
            # Cauchy(loc, scale) => loc + scale * tan(pi * (rand - 0.5))
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(current_pop_size) - 0.5))
            
            # Check F constraints
            retry_mask = f <= 0
            while np.any(retry_mask):
                f[retry_mask] = m_f[retry_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(retry_mask)) - 0.5))
                retry_mask = f <= 0
            f = np.minimum(f, 1.0)
            
            # Generate CR using Normal distribution
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # --- Mutation: current-to-pbest/1 ---
            # Sort for pbest selection
            sorted_indices = np.argsort(fitness)
            
            # Select pbest from top p% (p is random in [2/N, 0.2])
            p_val = np.random.uniform(2.0/current_pop_size, 0.2)
            top_cnt = max(2, int(p_val * current_pop_size))
            top_indices = sorted_indices[:top_cnt]
            
            pbest_idx = np.random.choice(top_indices, current_pop_size)
            x_pbest = population[pbest_idx]
            
            # Select r1 (distinct from i)
            r1_idx = np.random.randint(0, current_pop_size, current_pop_size)
            # Fix collisions
            cols = (r1_idx == np.arange(current_pop_size))
            while np.any(cols):
                r1_idx[cols] = np.random.randint(0, current_pop_size, np.sum(cols))
                cols = (r1_idx == np.arange(current_pop_size))
            x_r1 = population[r1_idx]
            
            # Select r2 (distinct from i and r1) from Population U Archive
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((population, archive_np))
            else:
                union_pop = population
            
            r2_idx = np.random.randint(0, len(union_pop), current_pop_size)
            cols = (r2_idx == np.arange(current_pop_size)) | (r2_idx == r1_idx)
            while np.any(cols):
                r2_idx[cols] = np.random.randint(0, len(union_pop), np.sum(cols))
                cols = (r2_idx == np.arange(current_pop_size)) | (r2_idx == r1_idx)
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant Vector
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            cross_mask = np.random.rand(current_pop_size, dim) < cr[:, np.newaxis]
            j_rand = np.random.randint(0, dim, current_pop_size)
            cross_mask[np.arange(current_pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, population)
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            new_fitness = np.empty(current_pop_size)
            
            for i in range(current_pop_size):
                if datetime.now() >= end_time: return best_val
                new_fitness[i] = safe_func(trial[i])
                evals_in_run += 1
            
            # Identify Improvements
            improved_mask = new_fitness < fitness
            improved_indices = np.where(improved_mask)[0]
            
            # Update Archive with replaced parents
            for i in improved_indices:
                archive.append(population[i].copy())
            
            # Maintain Archive Size (<= current_pop_size)
            while len(archive) > current_pop_size:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update Memory (History Adaptation)
            diffs = fitness[improved_mask] - new_fitness[improved_mask]
            if len(diffs) > 0:
                # Calculate weighted means based on fitness improvement
                weights = diffs / np.sum(diffs)
                
                success_f = f[improved_mask]
                success_cr = cr[improved_mask]
                
                # Weighted Lehmer Mean for F
                mean_f = np.sum(weights * (success_f ** 2)) / (np.sum(weights * success_f) + 1e-10)
                mem_f[k_mem] = mean_f
                
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(weights * success_cr)
                mem_cr[k_mem] = mean_cr
                
                k_mem = (k_mem + 1) % H
            
            # Update Population
            population[improved_mask] = trial[improved_mask]
            fitness[improved_mask] = new_fitness[improved_mask]

    return best_val
