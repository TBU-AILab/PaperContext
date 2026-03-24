#The following algorithm implements **L-SHADE (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution)** with a **Dynamic Time-Based Budget** and **Gaussian Local Search**.
#
#**Why this is an improvement over the previous best (SHADE):**
#1.  **Linear Population Reduction:** Unlike standard SHADE (which keeps population constant), this algorithm starts with a large population to explore the global landscape and linearly reduces it to a very small size. This creates "evolutionary pressure" that forces the algorithm to converge efficiently in the final stages, typically yielding higher precision.
#2.  **Dynamic Time-to-Eval Mapping:** Standard evolutionary algorithms assume a fixed number of function evaluations (MaxFES). Since this problem is constrained by `max_time`, this algorithm dynamically estimates the remaining number of possible evaluations based on the CPU speed of `func`. This ensures the L-SHADE reduction schedule perfectly fits the available time.
#3.  **Gaussian Local Search:** Differential Evolution is a global searcher and sometimes struggles to refine the last few decimal points. This implementation adds a lightweight Gaussian Local Search around the global best solution during the exploitation phase to fine-tune the minimum value.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Dynamic Time Budgeting and Local Search.
    
    Key Mechanics:
    1. Linear Population Size Reduction (LPSR) to transition from exploration to exploitation.
    2. Adaptive control parameters (F, CR) using historical success memory (SHADE).
    3. Dynamic estimation of remaining evaluations based on elapsed time.
    4. Restart mechanism with 'best-solution' preservation to escape local optima.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Reserve a small buffer (2%) to ensure we return cleanly before timeout
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    # --- Problem Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    best_val = float('inf')
    best_vec = None

    # --- Helper: Safe Evaluation ---
    # Wraps func to handle time checks and global best updates
    def evaluate(x):
        nonlocal best_val, best_vec
        # Bound constraints (clipping)
        x_clipped = np.clip(x, min_b, max_b)
        val = func(x_clipped)
        
        if val < best_val:
            best_val = val
            best_vec = x_clipped.copy()
        return val

    # --- 1. Speed Estimation (Calibration) ---
    # Run a few dummy evaluations to estimate 'seconds per evaluation'
    n_est = 3
    t0_cal = datetime.now()
    for _ in range(n_est):
        if datetime.now() >= end_time: return best_val
        evaluate(min_b + np.random.rand(dim) * diff_b)
    t1_cal = datetime.now()
    avg_eval_time = (t1_cal - t0_cal).total_seconds() / n_est
    # Add a safety margin to the estimate
    avg_eval_time = max(avg_eval_time, 1e-6)

    # --- L-SHADE Hyperparameters ---
    # Initial population size (N_init) - typically 18*D for L-SHADE
    # We cap it to prevent overhead in high dimensions/slow functions
    N_init = int(min(300, max(30, 18 * dim)))
    # Minimum population size (N_min)
    N_min = 4
    # History memory size
    H = 6 

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check if enough time remains for a meaningful restart
        # We need at least enough time for a few generations
        time_now = datetime.now()
        remaining_seconds = (end_time - time_now).total_seconds()
        
        if remaining_seconds < 0.2: 
            return best_val
        
        # Estimate total budget (MaxFES) for this specific run
        # If this is the first run, we use all time. If a restart, we use remaining time.
        # We allocate budget assuming we want to finish the schedule within remaining time.
        estimated_evals_left = int(remaining_seconds / avg_eval_time)
        
        # If budget is too small for full N_init, scale N_init down
        current_pop_size = N_init
        if estimated_evals_left < current_pop_size * 2:
            current_pop_size = max(N_min + 2, estimated_evals_left // 10)
        
        max_evals_for_run = estimated_evals_left
        evals_used = 0
        
        # --- Initialization ---
        population = min_b + np.random.rand(current_pop_size, dim) * diff_b
        # If we have a best solution from a previous run, inject it (Preserve Elite)
        if best_vec is not None:
            population[0] = best_vec.copy()
            
        fitness = np.full(current_pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(current_pop_size):
            if datetime.now() >= end_time: return best_val
            fitness[i] = evaluate(population[i])
            evals_used += 1

        # Memory for SHADE
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # --- Evolution Loop ---
        while True:
            # 1. Time & Budget Check
            if datetime.now() >= end_time: return best_val
            
            # Dynamic budget update: Recalculate max_evals based on actual speed
            # (Evaluation time might vary or initial estimate might be off)
            current_time = datetime.now()
            time_spent_run = (current_time - time_now).total_seconds()
            if time_spent_run > 0.05: # Only update if some time passed
                rate = evals_used / time_spent_run
                time_left = (end_time - current_time).total_seconds()
                max_evals_for_run = evals_used + int(rate * time_left)

            # 2. Linear Population Size Reduction (LPSR)
            # Formula: N_{g+1} = round( (N_min - N_init) * (FES / MAX_FES) + N_init )
            progress = min(1.0, evals_used / (max_evals_for_run + 1e-10))
            new_pop_size = int(round((N_min - N_init) * progress + N_init))
            new_pop_size = max(N_min, new_pop_size)
            
            if current_pop_size > new_pop_size:
                # Reduce population: remove worst individuals
                sorted_indices = np.argsort(fitness)
                # Keep top 'new_pop_size'
                keep_idx = sorted_indices[:new_pop_size]
                population = population[keep_idx]
                fitness = fitness[keep_idx]
                current_pop_size = new_pop_size
                
                # Resize archive to match current population size
                if len(archive) > current_pop_size:
                    random.shuffle(archive)
                    archive = archive[:current_pop_size]

            # 3. Convergence Check (Restart Trigger)
            # If population variance is zero or size is minimal and no progress
            if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness)) < 1e-9:
                break # Break inner loop to restart

            # 4. Generate Parameters (F, CR) using Memory
            # Randomly select memory index
            r_idx = np.random.randint(0, H, current_pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate F: Cauchy Distribution
            # cauchy = loc + scale * tan(pi * (rand - 0.5))
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(current_pop_size) - 0.5))
            # Fix F <= 0 (regenerate) and F > 1 (clip to 1)
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f = np.minimum(f, 1.0)
            
            # Generate CR: Normal Distribution
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)

            # 5. Mutation: current-to-pbest/1
            # Sort population to find p-best
            sorted_idx = np.argsort(fitness)
            
            # p varies for diversity, typically top 5% to 20%
            p = np.random.uniform(2.0/current_pop_size, 0.2)
            top_p_cnt = max(2, int(p * current_pop_size))
            top_p_indices = sorted_idx[:top_p_cnt]
            
            pbest_indices = np.random.choice(top_p_indices, current_pop_size)
            x_pbest = population[pbest_indices]
            
            # Select r1 (distinct from i)
            r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
            # Simple collision fix
            collision = (r1_indices == np.arange(current_pop_size))
            while np.any(collision):
                r1_indices[collision] = np.random.randint(0, current_pop_size, np.sum(collision))
                collision = (r1_indices == np.arange(current_pop_size))
            x_r1 = population[r1_indices]
            
            # Select r2 (distinct from i and r1) from Population U Archive
            if len(archive) > 0:
                archive_arr = np.array(archive)
                union_pop = np.vstack((population, archive_arr))
            else:
                union_pop = population
            
            union_size = len(union_pop)
            r2_indices = np.random.randint(0, union_size, current_pop_size)
            # Collision fix for r2
            collision = (r2_indices == np.arange(current_pop_size)) | (r2_indices == r1_indices)
            while np.any(collision):
                r2_indices[collision] = np.random.randint(0, union_size, np.sum(collision))
                collision = (r2_indices == np.arange(current_pop_size)) | (r2_indices == r1_indices)
            x_r2 = union_pop[r2_indices]
            
            # Calculate Mutation Vectors
            # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # 6. Crossover (Binomial)
            mask_rand = np.random.rand(current_pop_size, dim)
            cross_mask = mask_rand < cr[:, np.newaxis]
            # Ensure at least one dimension comes from mutant
            j_rand = np.random.randint(0, dim, current_pop_size)
            cross_mask[np.arange(current_pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 7. Selection
            new_fitness = np.empty(current_pop_size)
            success_mask = np.zeros(current_pop_size, dtype=bool)
            
            # We must iterate to evaluate because 'func' expects single vector
            # (or we could assume func handles batch, but prompt says np.array -> float)
            for i in range(current_pop_size):
                if datetime.now() >= end_time: return best_val
                
                val = evaluate(trial_pop[i])
                evals_used += 1
                new_fitness[i] = val
                
                if val < fitness[i]:
                    success_mask[i] = True
            
            # 8. Update Archive and Memory
            success_indices = np.where(success_mask)[0]
            if len(success_indices) > 0:
                # Add overwritten parents to archive
                for idx in success_indices:
                    archive.append(population[idx].copy())
                
                # Maintain archive size
                while len(archive) > current_pop_size:
                    archive.pop(random.randint(0, len(archive)-1))
                
                # Update Memory (Lehmer Mean)
                diff_fit = fitness[success_indices] - new_fitness[success_indices]
                total_diff = np.sum(diff_fit)
                if total_diff > 0:
                    weights = diff_fit / total_diff
                    
                    s_f = f[success_indices]
                    s_cr = cr[success_indices]
                    
                    mean_f = np.sum(weights * (s_f**2)) / (np.sum(weights * s_f) + 1e-9)
                    mean_cr = np.sum(weights * s_cr)
                    
                    mem_f[k_mem] = mean_f
                    mem_cr[k_mem] = mean_cr
                    k_mem = (k_mem + 1) % H
            
            # Update Population
            population[success_mask] = trial_pop[success_mask]
            fitness[success_mask] = new_fitness[success_mask]
            
            # 9. Gaussian Local Search (Refinement)
            # Occasionally (e.g., if we are in exploitation phase or improved global best)
            # try to refine the global best slightly
            if best_vec is not None and random.random() < 0.1:
                # Scale step size based on current population spread
                # As we converge, the spread decreases, so step size decreases automatically
                step_scale = np.mean(np.std(population, axis=0)) 
                if step_scale > 1e-12:
                    neighbor = best_vec + np.random.normal(0, step_scale, dim)
                    if datetime.now() >= end_time: return best_val
                    evaluate(neighbor)
                    evals_used += 1

    return best_val
