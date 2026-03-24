#The following algorithm implements **L-SHADE with Soft Restarts (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution)**.
#
#**Reasoning for Improvement:**
#1.  **Best of Both Worlds**: It combines the state-of-the-art **L-SHADE** (which linearly reduces population size to focus computational resources on exploitation as time runs out) with an **Adaptive Restart** mechanism.
#2.  **Dynamic Time Management**: Standard L-SHADE assumes a fixed budget. This implementation calculates reduction schedules based on "Session Time." If the population converges early (stagnation), it triggers a restart, keeping the best solution found, and generates a *new* reduction schedule for the *remaining* time. This prevents the algorithm from idling with a small, stagnant population.
#3.  **Midpoint Bound Handling**: Instead of simple clipping (which piles points on the bounds), it uses midpoint correction `(parent + bound) / 2`. This preserves population diversity near the edges of the search space.
#4.  **SHADE Features**: Utilizes historical memory ($H=5$) to adapt mutation factor $F$ and crossover rate $CR$ distributions (Cauchy and Normal) based on successful improvements, utilizing weighted Lehmer means.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' using L-SHADE with Linear Population Size Reduction
    and Adaptive Restarts within a time budget.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Reserve a small buffer (2%) to ensure strictly valid return
    time_limit = timedelta(seconds=max_time * 0.98)
    
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    global_best_val = float('inf')
    
    # SHADE Parameters
    H = 5  # Memory size
    
    # --- Main Loop (Handles Restarts) ---
    while not check_time():
        
        # --- Session Initialization ---
        session_start = datetime.now()
        
        # Population Sizing for this session
        # L-SHADE standard: N_init = 18 * dim
        # Clamped to ensure speed on high dims and diversity on low dims
        N_init = int(18 * dim)
        N_init = max(30, min(300, N_init))
        N_min = 4
        
        curr_pop_size = N_init
        
        # Initialize Population
        population = min_b + np.random.rand(curr_pop_size, dim) * diff_b
        fitness = np.full(curr_pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(curr_pop_size):
            if check_time(): return global_best_val
            val = func(population[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # Sort population by fitness (required for L-SHADE reduction & p-best)
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Initialize SHADE Memory and Archive
        mem_M_CR = np.full(H, 0.5)
        mem_M_F = np.full(H, 0.5)
        k_mem = 0
        archive = [] # Stores replaced parent vectors
        
        # --- Evolutionary Session Loop ---
        while not check_time():
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate progress relative to the REMAINING global time.
            # We treat the current restart session as utilizing all remaining time.
            now = datetime.now()
            session_end_target = start_time + time_limit
            total_session_time = (session_end_target - session_start).total_seconds()
            elapsed_session_time = (now - session_start).total_seconds()
            
            # Avoid division by zero
            if total_session_time > 1e-4:
                progress = elapsed_session_time / total_session_time
            else:
                progress = 1.0
            
            # Calculate Target Population Size based on progress
            # Reduces from N_init to N_min
            target_size = int(round(N_init + (N_min - N_init) * progress))
            target_size = max(N_min, target_size)
            
            # Apply Reduction
            if curr_pop_size > target_size:
                n_remove = curr_pop_size - target_size
                # Since population is sorted at end of loop, worst are at the end
                population = population[:-n_remove]
                fitness = fitness[:-n_remove]
                curr_pop_size = target_size
                
                # Resize Archive: |A| <= |P|
                while len(archive) > curr_pop_size:
                    archive.pop(np.random.randint(0, len(archive)))

            # 2. Check for Convergence (Stagnation)
            # If population variance is negligible, restart to use remaining time elsewhere.
            if curr_pop_size >= N_min:
                fit_std = np.std(fitness)
                if fit_std < 1e-8:
                    break # Break inner loop -> triggers restart in outer loop

            # 3. Parameter Generation
            # Each individual picks a random memory slot
            r_idxs = np.random.randint(0, H, curr_pop_size)
            m_cr = mem_M_CR[r_idxs]
            m_f = mem_M_F[r_idxs]
            
            # Generate CR (Normal dist, close to memory value)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy dist, heavy tail for exploration)
            f = m_f + 0.1 * np.random.standard_cauchy(curr_pop_size)
            
            # Repair F
            f[f > 1] = 1.0 # Clip top
            # Resample if <= 0
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = m_f[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
                f[f > 1] = 1.0
            
            # 4. Mutation: current-to-pbest/1
            # p determines the greediness. Random in [2/N, 0.2] is robust.
            p_min = 2.0 / curr_pop_size
            p_val = np.random.uniform(p_min, 0.2)
            p_top = int(max(2, round(p_val * curr_pop_size)))
            
            # Select p-best (from top sorted individuals)
            pbest_idxs = np.random.randint(0, p_top, curr_pop_size)
            x_pbest = population[pbest_idxs]
            
            # Select r1 (!= i)
            r1_idxs = np.random.randint(0, curr_pop_size, curr_pop_size)
            # Handle collisions (r1 == i)
            col_mask = (r1_idxs == np.arange(curr_pop_size))
            r1_idxs[col_mask] = (r1_idxs[col_mask] + 1) % curr_pop_size
            x_r1 = population[r1_idxs]
            
            # Select r2 (!= r1, != i) from Population Union Archive
            if len(archive) > 0:
                arr_archive = np.array(archive)
                p_u_a = np.vstack((population, arr_archive))
            else:
                p_u_a = population
                
            r2_idxs = np.random.randint(0, len(p_u_a), curr_pop_size)
            # Skip strict r2 collision check for speed (standard practice in vector DE)
            x_r2 = p_u_a[r2_idxs]
            
            # Compute Mutant Vector V
            f_col = f[:, np.newaxis]
            v = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, curr_pop_size)
            mask = np.random.rand(curr_pop_size, dim) < cr[:, np.newaxis]
            mask[np.arange(curr_pop_size), j_rand] = True # Ensure 1 param changes
            
            u = np.where(mask, v, population)
            
            # 6. Bound Constraint Handling (Midpoint Correction)
            # If outside bounds, place halfway between parent and bound.
            # Better than clipping for convergence near edges.
            low_mask = u < min_b
            high_mask = u > max_b
            
            # Vectorized midpoint calculation
            # Note: broadcast (min_b + population) matches shape of u
            u[low_mask] = (min_b + population)[low_mask] / 2.0
            u[high_mask] = (max_b + population)[high_mask] / 2.0
            
            # 7. Selection and Memory Update
            new_fitness = np.empty(curr_pop_size)
            improved_indices = []
            diff_vals = []
            
            # Evaluate trial vectors
            for i in range(curr_pop_size):
                if check_time(): return global_best_val
                
                val = func(u[i])
                new_fitness[i] = val
                
                if val <= fitness[i]:
                    improved_indices.append(i)
                    diff_vals.append(fitness[i] - val)
                    
                    if val < global_best_val:
                        global_best_val = val
            
            # 8. Updates
            if improved_indices:
                improved_indices = np.array(improved_indices)
                diff_vals = np.array(diff_vals)
                
                # Add replaced parents to Archive
                for idx in improved_indices:
                    archive.append(population[idx].copy())
                
                # Resize Archive
                while len(archive) > curr_pop_size:
                    archive.pop(np.random.randint(0, len(archive)))
                
                # Update Population
                population[improved_indices] = u[improved_indices]
                fitness[improved_indices] = new_fitness[improved_indices]
                
                # Update SHADE Memory (Weighted Lehmer Mean)
                total_diff = np.sum(diff_vals)
                if total_diff > 0:
                    weights = diff_vals / total_diff
                    
                    # Update M_CR
                    s_cr = cr[improved_indices]
                    m_cr_new = np.sum(weights * s_cr)
                    
                    # Update M_F
                    s_f = f[improved_indices]
                    num_f = np.sum(weights * (s_f ** 2))
                    den_f = np.sum(weights * s_f)
                    m_f_new = num_f / den_f if den_f > 0 else 0.5
                    
                    mem_M_CR[k_mem] = m_cr_new
                    mem_M_F[k_mem] = np.clip(m_f_new, 0, 1)
                    k_mem = (k_mem + 1) % H
            
            # 9. Sort Population (Crucial step for LPSR and p-best logic next iter)
            sort_order = np.argsort(fitness)
            population = population[sort_order]
            fitness = fitness[sort_order]

    return global_best_val
