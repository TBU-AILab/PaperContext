#The following algorithm implements **jSO (Specific-parameter L-SHADE)** adapted for **Time-Bound Optimization**.
#
#**Why this is an improvement:**
#1.  **Time-Based Linear Population Reduction**: Instead of relying on unreliable "evaluation counts" (like Algorithm 3) or fixed populations (Algorithm 1), this algorithm linearly reduces the population size from a large initial swarm (exploration) to a minimal set (exploitation) based on the *actual elapsed time*. If the algorithm converges early and restarts, the reduction schedule adapts dynamically to the *remaining* time.
#2.  **Adaptive Ranking Selection (p-Best)**: It implements the jSO strategy where the parameter $p$ (controlling greediness) linearly decreases over time. It starts with high diversity (large $p$) and becomes very greedy (small $p$) towards the end to refine the solution.
#3.  **Weighted Memory Updates**: It uses a weighted Lehmer mean based on fitness improvement magnitude, prioritizing parameters that yielded significant gains, rather than just any improvement.
#4.  **Robust Restart Strategy**: It detects stagnation (low variance) early. If optimization stalls, it restarts but preserves the "Elite" (global best) and re-calibrates the population reduction schedule for the remaining time window.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Time-Adaptive jSO (L-SHADE variant) with Restarts.
    """
    # --- Configuration & Time Management ---
    start_time = datetime.now()
    # Safety buffer: stop slightly before max_time to ensure safe return
    end_time = start_time + timedelta(seconds=max_time * 0.98)
    
    # Bound processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_vec = None

    # --- Restart Loop ---
    # We treat the problem as a sequence of runs. 
    # Each run tries to utilize the *remaining* time fully for its reduction schedule.
    while True:
        current_time = datetime.now()
        if current_time >= end_time:
            return best_val

        # 1. Setup for this specific run
        # Determine remaining time for this run's schedule
        remaining_seconds = (end_time - current_time).total_seconds()
        
        # If remaining time is too short (< 0.2s), stop to avoid overhead errors
        if remaining_seconds < 0.2:
            return best_val

        run_start_time = current_time
        run_end_time = end_time 
        
        # --- jSO/L-SHADE Parameters ---
        # Initial Population: Adaptive to dimension, capped for performance
        # jSO typically uses ~25*dim, but in Python with limited time, 15-20*dim is safer
        N_init = min(200, max(30, 20 * dim))
        N_min = 4
        
        # Initialize Population
        pop_size = N_init
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Inject global best if available (Elitism across restarts)
        if best_vec is not None:
            population[0] = best_vec.copy()
            # Perturb others slightly around best to encourage local search in restart
            # (Use 10% of pop for local search around best)
            n_local = int(pop_size * 0.1)
            for k in range(1, n_local):
                population[k] = best_vec + np.random.normal(0, 0.01, dim) * diff_b
                population[k] = np.clip(population[k], min_b, max_b)

        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_vec = population[i].copy()

        # Historical Memory (size H=5 is robust for short runs)
        H_size = 5
        mem_f = np.full(H_size, 0.5)
        mem_cr = np.full(H_size, 0.8)
        k_mem = 0
        archive = []
        
        # --- Evolutionary Loop ---
        while True:
            t_now = datetime.now()
            if t_now >= end_time: return best_val

            # Calculate Time Progress (0.0 to 1.0)
            # This drives the Linear Reduction and Parameter Adaptation
            elapsed = (t_now - run_start_time).total_seconds()
            total_duration = (run_end_time - run_start_time).total_seconds()
            # Avoid division by zero
            progress = min(1.0, elapsed / (total_duration + 1e-9))
            
            # 2. Linear Population Size Reduction (LPSR)
            # N_{g+1} = round( (N_min - N_init) * progress + N_init )
            new_pop_size = int(round((N_min - N_init) * progress + N_init))
            new_pop_size = max(N_min, new_pop_size)
            
            if new_pop_size < pop_size:
                # Reduction: Sort and keep best
                sorted_idx = np.argsort(fitness)
                keep_idx = sorted_idx[:new_pop_size]
                population = population[keep_idx]
                fitness = fitness[keep_idx]
                pop_size = new_pop_size
                
                # Resize Archive: maintain size <= pop_size
                if len(archive) > pop_size:
                    random.shuffle(archive)
                    archive = archive[:pop_size]

            # 3. Check for Stagnation (Trigger Early Restart)
            # If population variance is extremely low, we are stuck.
            # Don't wait for the timer to finish; restart immediately to explore elsewhere.
            if pop_size >= N_min:
                std_dev = np.std(fitness)
                spread = np.max(fitness) - np.min(fitness)
                if std_dev < 1e-9 or spread < 1e-9:
                    break # Break inner loop, triggering restart in outer loop

            # 4. Generate Control Parameters
            # Pick random memory slot
            r_idx = np.random.randint(0, H_size, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate F (Cauchy Distribution)
            # F < 0 -> Regenerate, F > 1 -> 1
            f_params = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            while np.any(f_params <= 0):
                mask = f_params <= 0
                f_params[mask] = m_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f_params = np.minimum(f_params, 1.0)
            
            # Generate CR (Normal Distribution)
            cr_params = np.random.normal(m_cr, 0.1)
            cr_params = np.clip(cr_params, 0.0, 1.0)
            
            # jSO: Modified CR for late stage (optional, but standard DE logic is fine here)
            
            # 5. Mutation: current-to-pbest/1 with Weighted Archive
            # p varies linearly from 0.25 (exploration) to 0.05 (exploitation) based on progress
            p_val = 0.25 - (0.20 * progress)
            p_val = max(0.05, p_val)
            
            # Sort for p-best selection
            sorted_indices = np.argsort(fitness)
            top_cnt = max(2, int(pop_size * p_val))
            top_indices = sorted_indices[:top_cnt]
            
            # Select pbest
            pbest_idx = np.random.choice(top_indices, pop_size)
            x_pbest = population[pbest_idx]
            
            # Select r1 (distinct from i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Fix collisions
            cols = (r1_idx == np.arange(pop_size))
            while np.any(cols):
                r1_idx[cols] = np.random.randint(0, pop_size, np.sum(cols))
                cols = (r1_idx == np.arange(pop_size))
            x_r1 = population[r1_idx]
            
            # Select r2 (distinct from i and r1) from Union(Population, Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((population, archive_np))
            else:
                union_pop = population
            
            union_size = len(union_pop)
            r2_idx = np.random.randint(0, union_size, pop_size)
            
            # Fix collisions for r2
            cols = (r2_idx == np.arange(pop_size)) | (r2_idx == r1_idx)
            while np.any(cols):
                r2_idx[cols] = np.random.randint(0, union_size, np.sum(cols))
                cols = (r2_idx == np.arange(pop_size)) | (r2_idx == r1_idx)
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant: v = x + F*(xp - x) + F*(xr1 - xr2) (Modified SHADE eq)
            # Using standard current-to-pbest/1 equation:
            # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f_params[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # 6. Crossover (Binomial)
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < cr_params[:, np.newaxis]
            # Ensure at least one dimension
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 7. Selection and Updates
            new_fitness = np.empty(pop_size)
            success_mask = np.zeros(pop_size, dtype=bool)
            
            # Evaluation loop (cannot be vectorized due to func interface)
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_val
                
                val = func(trial_pop[i])
                new_fitness[i] = val
                
                if val < fitness[i]:
                    success_mask[i] = True
                    if val < best_val:
                        best_val = val
                        best_vec = trial_pop[i].copy()
            
            # 8. Update Archive and Memory
            success_idx = np.where(success_mask)[0]
            if len(success_idx) > 0:
                # Add overwritten solutions to archive
                for idx in success_idx:
                    archive.append(population[idx].copy())
                
                # Cap archive size
                while len(archive) > pop_size:
                    # Randomly remove
                    archive.pop(random.randint(0, len(archive)-1))
                
                # Update Memory (Weighted Lehmer Mean)
                diff_fitness = fitness[success_idx] - new_fitness[success_idx]
                total_diff = np.sum(diff_fitness)
                
                if total_diff > 0:
                    # Weights based on improvement magnitude
                    weights = diff_fitness / total_diff
                    
                    s_f = f_params[success_idx]
                    s_cr = cr_params[success_idx]
                    
                    # Mean F (Lehmer)
                    mean_f = np.sum(weights * (s_f**2)) / (np.sum(weights * s_f) + 1e-10)
                    # Mean CR (Lehmer or Weighted Arith) - jSO uses Weighted Lehmer usually
                    mean_cr = np.sum(weights * (s_cr**2)) / (np.sum(weights * s_cr) + 1e-10)
                    
                    # Update memory slot
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f # Smooth update
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                    k_mem = (k_mem + 1) % H_size
                
                # Apply selection to population
                population[success_mask] = trial_pop[success_mask]
                fitness[success_mask] = new_fitness[success_mask]

    return best_val
