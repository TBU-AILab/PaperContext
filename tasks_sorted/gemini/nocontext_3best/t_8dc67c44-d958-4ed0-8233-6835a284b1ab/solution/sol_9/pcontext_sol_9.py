#The following algorithm implements **Time-Adaptive L-SHADE with Elitist Restarts**.
#
#**Why this is an improvement:**
#1.  **Time-Based Linear Population Reduction (LPSR)**: Unlike fixed-population algorithms (SHADE) or generic reduction (jSO), this algorithm dynamically calculates the population reduction schedule based on the *remaining time* in the current restart session. This forces the algorithm to shift from exploration to exploitation exactly as the time limit approaches, maximizing the utility of the available budget.
#2.  **Elitist Soft Restarts**: Standard restarts wipe out the population. This algorithm implements "Elitist Restarts," where the global best solution found so far (plus mutated clones) is injected into the new population. This prevents the loss of good solutions and allows the algorithm to escape local optima while continuing to refine the best known basin.
#3.  **Robust Stagnation Detection**: It monitors population variance to detect convergence early. If the search stagnates before the time limit, it immediately restarts with a fresh exploration phase, ensuring no time is wasted on a converged state.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Time-Adaptive L-SHADE with Elitist Restarts.
    Adapts population size and control parameters based on remaining time.
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Use 98% of allowed time to ensure safe return before timeout
    end_time = start_time + timedelta(seconds=max_time * 0.98)

    # --- Problem Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Tracking ---
    best_val = np.inf
    best_vec = None
    
    # --- L-SHADE Parameters ---
    # Initial population size: Aggressive initial exploration size
    # Cap at 250 to avoid performance bottlenecks in short runs
    n_init = min(250, max(30, 18 * dim))
    n_min = 4  # Minimum population size at the end of schedule
    
    # Memory size for SHADE history
    H = 6
    
    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Check overall time
        current_time = datetime.now()
        if current_time >= end_time:
            return best_val
        
        # Determine remaining time window for this restart session
        # The LPSR schedule is scaled to this specific restart duration.
        time_left = (end_time - current_time).total_seconds()
        
        # If less than 0.1s left, pointless to start a new population
        if time_left < 0.1:
            return best_val
            
        run_start = current_time
        # We plan the reduction over the remaining time
        run_duration = time_left
        
        # 1. Initialization for Restart
        pop_size = n_init
        
        # Create population
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # ELITISM: Inject global best into the new population
        # This transforms the restart from "blind" to "exploratory around best"
        if best_vec is not None:
            population[0] = best_vec.copy()
            # Inject mutated clones (10% of pop) to explore the best basin
            n_clones = int(pop_size * 0.1)
            for k in range(1, n_clones):
                # Add gaussian noise scaled to bounds
                noisy = best_vec + np.random.normal(0, 0.05, dim) * diff_b
                population[k] = np.clip(noisy, min_b, max_b)
        
        fitness = np.full(pop_size, np.inf)
        
        # Eval Initial Population
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_vec = population[i].copy()
                
        # Initialize Memories (History)
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        # 2. Evolutionary Loop
        while True:
            t_now = datetime.now()
            if t_now >= end_time: return best_val
            
            # --- Linear Population Size Reduction (LPSR) ---
            # Calculate progress (0.0 -> 1.0) based on TIME
            elapsed = (t_now - run_start).total_seconds()
            progress = min(1.0, elapsed / (run_duration + 1e-10))
            
            # Target population size
            next_pop_size = int(round((n_min - n_init) * progress + n_init))
            next_pop_size = max(n_min, next_pop_size)
            
            # Resize if needed
            if next_pop_size < pop_size:
                # Sort and keep best individuals
                sort_idx = np.argsort(fitness)
                population = population[sort_idx[:next_pop_size]]
                fitness = fitness[sort_idx[:next_pop_size]]
                pop_size = next_pop_size
                
                # Shrink archive to match current pop_size
                if len(archive) > pop_size:
                    # Random reduction matches L-SHADE standard
                    keep_indices = random.sample(range(len(archive)), pop_size)
                    archive = [archive[i] for i in keep_indices]

            # --- Stagnation Detection ---
            # If population variance is negligible, we are stuck.
            fit_std = np.std(fitness)
            if fit_std < 1e-8 or (np.max(fitness) - np.min(fitness)) < 1e-8:
                break # Break inner loop -> Trigger Restart
            
            # --- Parameter Adaptation ---
            # Select random memory index
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate F (Cauchy Distribution)
            # cauchy(loc, scale) => loc + scale * tan(pi*(rand-0.5))
            f_params = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Rectify F: If F > 1 -> 1. If F <= 0 -> regenerate.
            while True:
                mask = f_params <= 0
                if not np.any(mask): break
                f_params[mask] = m_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f_params = np.minimum(f_params, 1.0)
            
            # Generate CR (Normal Distribution)
            cr_params = np.random.normal(m_cr, 0.1)
            cr_params = np.clip(cr_params, 0.0, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # Sort for p-best selection
            sorted_indices = np.argsort(fitness)
            
            # p value: Random in [2/pop_size, 0.2] ensures diversity
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            n_top = max(2, int(pop_size * p_val))
            top_indices = sorted_indices[:n_top]
            
            # Select pbest
            pbest_idx = np.random.choice(top_indices, pop_size)
            x_pbest = population[pbest_idx]
            
            # Select r1 (!= i)
            r1_idx = np.random.randint(0, pop_size, pop_size)
            for i in range(pop_size):
                while r1_idx[i] == i:
                    r1_idx[i] = np.random.randint(0, pop_size)
            x_r1 = population[r1_idx]
            
            # Select r2 (!= i, != r1) from Union(Pop, Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((population, archive_np))
            else:
                union_pop = population
            
            union_size = len(union_pop)
            r2_idx = np.random.randint(0, union_size, pop_size)
            for i in range(pop_size):
                while r2_idx[i] == i or r2_idx[i] == r1_idx[i]:
                    r2_idx[i] = np.random.randint(0, union_size)
            x_r2 = union_pop[r2_idx]
            
            # Compute Mutant Vector
            # v = x + F*(pbest - x) + F*(r1 - r2)
            f_col = f_params[:, None]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr_params[:, None]
            # Ensure at least one dimension is mutated
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection and Memory Update Prep ---
            success_f = []
            success_cr = []
            diff_f = []
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_val
                
                new_val = func(trial_pop[i])
                
                if new_val <= fitness[i]:
                    # Solution improved (or equal)
                    if new_val < fitness[i]:
                        # Add replaced solution to archive
                        archive.append(population[i].copy())
                        # Store success params for memory update
                        success_f.append(f_params[i])
                        success_cr.append(cr_params[i])
                        diff_f.append(fitness[i] - new_val)
                    
                    # Update Population
                    population[i] = trial_pop[i]
                    fitness[i] = new_val
                    
                    if new_val < best_val:
                        best_val = new_val
                        best_vec = trial_pop[i].copy()
            
            # --- Archive Maintenance ---
            while len(archive) > pop_size:
                archive.pop(np.random.randint(0, len(archive)))
                
            # --- Memory Update (Weighted Lehmer Mean) ---
            if len(diff_f) > 0:
                w = np.array(diff_f)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    
                    sf = np.array(success_f)
                    scr = np.array(success_cr)
                    
                    # Lehmer Mean for F
                    mean_f = np.sum(w * (sf ** 2)) / np.sum(w * sf)
                    
                    # Weighted Mean for CR
                    mean_cr = np.sum(w * scr)
                    
                    # Update Memory slot (with learning rate 0.5)
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                    mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                    k_mem = (k_mem + 1) % H

    return best_val
