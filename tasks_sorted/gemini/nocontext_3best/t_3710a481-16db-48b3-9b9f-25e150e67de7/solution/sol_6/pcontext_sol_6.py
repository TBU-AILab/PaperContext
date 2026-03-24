#The following is an improved algorithm based on **L-SHADE (Linear Success-History based Adaptive Differential Evolution)** with a **Dynamic Restart Strategy**.
#
#### Improvements over Previous Best (Algorithm 1)
#1.  **Linear Population Size Reduction (LPSR)**: Instead of a fixed population size, this algorithm starts with a larger population to explore the search space and linearly reduces it over time. This mimics the strategy of top-performing algorithms in evolutionary computation competitions (like CEC 2014 winner), forcing exploitation as the time budget decreases.
#2.  **Epoch-Based Restart**: The standard L-SHADE assumes a fixed budget. Here, we implement a restart mechanism where the "budget" for the population reduction is dynamically recalculated based on the *remaining time* after a restart. If the population converges early, it restarts and squeezes the next population faster.
#3.  **Adaptive Archive**: The external archive size limits are dynamically adjusted to match the shrinking population, ensuring diversity maintenance scales correctly with the search phase.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Restart Strategy (Linear Population Size Reduction)
    """
    # Initialize main timer
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Initial population size: 18 * dim is a standard setting for L-SHADE
    # It provides high initial diversity.
    N_init = int(max(30, 18 * dim))
    # Minimum population size to reach by the end of the epoch
    N_min = 4
    
    # Archive size parameter (Archive capacity = pop_size * arc_rate)
    arc_rate = 1.0 
    
    # History memory size for adaptive parameters
    H = 6
    
    # Pre-process bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global Best tracking
    best_val = float('inf')
    best_vec = None 
    
    # --- Main Optimization Loop (Restarts allowed) ---
    while True:
        # Check if time is already up
        now = datetime.now()
        if now - start_time >= time_limit:
            return best_val
            
        # Calculate budget for this epoch (Restart)
        remaining_seconds = (time_limit - (now - start_time)).total_seconds()
        
        # If remaining time is too trivial to start a meaningful run, stop
        if remaining_seconds < 0.1:
            return best_val
            
        epoch_start = now
        epoch_duration = remaining_seconds
        
        # --- Initialization for this Epoch ---
        pop_size = N_init
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Injection of global best (Biasing restart to not lose progress)
        if best_vec is not None:
            pop[0] = best_vec.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize Adaptive Memory (M_F, M_CR)
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive_cap = int(pop_size * arc_rate)
        archive = np.zeros((archive_cap, dim))
        arc_count = 0
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
                
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_vec = pop[i].copy()
                
        # --- Generation Loop ---
        while True:
            t_curr = datetime.now()
            if t_curr - start_time >= time_limit:
                return best_val
            
            # 1. Linear Population Size Reduction (LPSR)
            # ------------------------------------------
            # Calculate progress within this specific restart epoch
            t_epoch = (t_curr - epoch_start).total_seconds()
            progress = t_epoch / epoch_duration
            
            # Calculate target population size
            if progress > 1.0:
                target_size = N_min
            else:
                target_size = int(round((N_min - N_init) * progress + N_init))
            
            # Ensure safety
            target_size = max(N_min, target_size)
            
            # Apply reduction if needed
            if target_size < pop_size:
                # Sort population by fitness (worst at the end)
                sorted_indices = np.argsort(fitness)
                
                # Keep top 'target_size'
                pop = pop[sorted_indices[:target_size]]
                fitness = fitness[sorted_indices[:target_size]]
                
                # Update current size
                pop_size = target_size
                
                # Resize Archive: remove elements if over capacity
                # We simply truncate to new capacity
                archive_cap = int(pop_size * arc_rate)
                if arc_count > archive_cap:
                    archive = archive[:archive_cap]
                    arc_count = archive_cap
            
            # 2. Convergence Check (Restart Trigger)
            # --------------------------------------
            # Trigger restart if population is minimal or fitness has converged
            fit_range = 0.0
            if not np.isinf(fitness).any():
                fit_range = np.max(fitness) - np.min(fitness)
                
            if pop_size <= N_min or fit_range < 1e-8:
                # Stop this epoch, triggering a restart
                break
                
            # 3. Parameter Generation (Vectorized SHADE)
            # ------------------------------------------
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate F (Cauchy distribution)
            F = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Fix F values: If F <= 0, regenerate. If F > 1, clip to 1.
            while True:
                mask_neg = F <= 0
                if not np.any(mask_neg):
                    break
                # Only regenerate negative values
                F[mask_neg] = m_f[mask_neg] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask_neg)) - 0.5))
            F = np.minimum(F, 1.0)
            
            # Generate CR (Normal distribution)
            CR = np.random.normal(m_cr, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
            
            # 4. Mutation (current-to-pbest/1)
            # --------------------------------
            # Find p-best (top 11%)
            sorted_idx = np.argsort(fitness)
            p_best_num = max(2, int(0.11 * pop_size))
            p_best_indices = sorted_idx[:p_best_num]
            
            pbest_choice = np.random.choice(p_best_indices, pop_size)
            x_pbest = pop[pbest_choice]
            
            # r1: random from pop, distinct from i
            r1_idx = np.random.randint(0, pop_size, pop_size)
            # Simple collision fix
            cols = (r1_idx == np.arange(pop_size))
            if np.any(cols):
                r1_idx[cols] = np.random.randint(0, pop_size, np.sum(cols))
            x_r1 = pop[r1_idx]
            
            # r2: random from (Pop U Archive), distinct from i and r1
            if arc_count > 0:
                pool = np.vstack((pop, archive[:arc_count]))
            else:
                pool = pop
            
            r2_idx = np.random.randint(0, pool.shape[0], pop_size)
            x_r2 = pool[r2_idx]
            
            # Differential Mutation Vector
            mutant = pop + F[:, None] * (x_pbest - pop) + F[:, None] * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            # -----------------------
            rand_cross = np.random.rand(pop_size, dim)
            mask_cross = rand_cross < CR[:, None]
            # Ensure at least one dimension is changed
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cross, mutant, pop)
            
            # 6. Bound Constraints
            trial = np.clip(trial, min_b, max_b)
            
            # 7. Selection & Adaptation
            # -------------------------
            success_f = []
            success_cr = []
            diff_fits = []
            
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    # Good trial (improvement or neutral)
                    imp = fitness[i] - f_trial
                    
                    # Add parent to Archive
                    if arc_count < archive.shape[0]:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Replace random archive member
                        rep_idx = np.random.randint(0, archive.shape[0])
                        archive[rep_idx] = pop[i].copy()
                        
                    new_pop[i] = trial[i]
                    new_fitness[i] = f_trial
                    
                    if imp > 0:
                        success_f.append(F[i])
                        success_cr.append(CR[i])
                        diff_fits.append(imp)
                        
                    if f_trial < best_val:
                        best_val = f_trial
                        best_vec = trial[i].copy()
                        
            pop = new_pop
            fitness = new_fitness
            
            # 8. Update Memory (History)
            # --------------------------
            if len(diff_fits) > 0:
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                df = np.array(diff_fits)
                
                # Calculate weights based on fitness improvement
                total_df = np.sum(df)
                if total_df == 0:
                    w = np.ones_like(df) / len(df)
                else:
                    w = df / total_df
                    
                # Weighted Lehmer Mean for F
                f_num = np.sum(w * (s_f**2))
                f_den = np.sum(w * s_f)
                if f_den == 0:
                    mean_f = 0.5
                else:
                    mean_f = f_num / f_den
                
                # Weighted Arithmetic Mean for CR
                mean_cr = np.sum(w * s_cr)
                
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
                k_mem = (k_mem + 1) % H
                
    return best_val
