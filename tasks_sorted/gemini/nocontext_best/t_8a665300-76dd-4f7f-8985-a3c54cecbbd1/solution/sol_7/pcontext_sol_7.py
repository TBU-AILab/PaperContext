#The following algorithm implements **L-SHADE-RSP (Linear Success-History Adaptive Differential Evolution with Restart Strategy and Polishing)**.
#
#**Key Improvements over the previous algorithm:**
#1.  **Linear Population Size Reduction (LPSR):** The population size dynamically shrinks from a large number (exploration) to a small number (exploitation) as time progresses. This drastically improves convergence speed towards the end.
#2.  **Success-History Memory (SHADE):** Instead of simple adaptive means, it uses a history memory to store successful $F$ and $CR$ settings, utilizing a weighted Lehmer mean based on fitness improvement magnitudes.
#3.  **Soft Restarts:** If the population converges (low variance) or stagnates before time runs out, the algorithm triggers a "Soft Restart." It keeps the best solution, scrambles the rest, and resets parameters to find better basins of attraction (crucial for multimodal functions).
#4.  **Coordinate Descent Polish:** In the final milliseconds, a lightweight coordinate descent refines the best solution to ensure high precision, avoiding the overhead of Nelder-Mead in high dimensions.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Implements L-SHADE with Linear Population Size Reduction, 
    Soft Restart mechanism, and Local Coordinate Descent Polishing.
    """
    
    # --- Configuration & Helper Functions ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Algorithm Hyperparameters
    r_arc = 2.0             # Archive size ratio
    p_best_rate = 0.11      # Top 11% for p-best selection
    memory_size = 6         # Size of parameter history memory
    
    # Initial Population settings
    # Start with a high density to cover space, end with minimal for fast convergence
    pop_size_init = int(round(18 * dim)) 
    pop_size_min = 4 
    
    # --- State Variables ---
    best_fitness = float('inf')
    best_solution = np.zeros(dim)
    
    # Time management helper
    def get_remaining_time():
        elapsed = datetime.now() - start_time
        return max_time - elapsed.total_seconds()
        
    def check_timeout(buffer_sec=0.0):
        return (datetime.now() - start_time) >= (time_limit - timedelta(seconds=buffer_sec))

    # --- Initialization Logic ---
    def initialize_population(count):
        pop = np.zeros((count, dim))
        for d in range(dim):
            pop[:, d] = np.random.uniform(min_b[d], max_b[d], count)
        return pop

    # --- Main Optimization Cycle (Restart Loop) ---
    # We loop allowing restarts if convergence happens too early
    while not check_timeout(buffer_sec=0.05 * max_time):
        
        # 1. Reset/Init State for this epoch
        pop_size = pop_size_init
        population = initialize_population(pop_size)
        
        # Evaluate initial population
        fitness_vals = np.array([func(ind) for ind in population])
        
        # Update Global Best
        min_idx = np.argmin(fitness_vals)
        if fitness_vals[min_idx] < best_fitness:
            best_fitness = fitness_vals[min_idx]
            best_solution = population[min_idx].copy()
            
        # L-SHADE Memory Initialization
        memory_sf = np.full(memory_size, 0.5)
        memory_scr = np.full(memory_size, 0.5)
        k_mem_idx = 0
        
        archive = []
        
        # Stagnation counter for restart
        stagnation_count = 0
        last_best_fit_in_epoch = best_fitness
        
        # --- Evolutionary Loop ---
        while True:
            # Check Global Time
            if check_timeout(buffer_sec=0.02 * max_time):
                break
                
            # Current progress (0.0 to 1.0)
            elapsed_sec = (datetime.now() - start_time).total_seconds()
            progress = min(1.0, elapsed_sec / max_time)
            
            # --- Linear Population Size Reduction (LPSR) ---
            # Calculate target size based on time progress
            plan_pop_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
            plan_pop_size = max(pop_size_min, plan_pop_size)
            
            if pop_size > plan_pop_size:
                # Reduce population: remove worst individuals
                # Sort by fitness
                sorted_indices = np.argsort(fitness_vals)
                population = population[sorted_indices]
                fitness_vals = fitness_vals[sorted_indices]
                
                # Trim
                new_pop_cnt = plan_pop_size
                population = population[:new_pop_cnt]
                fitness_vals = fitness_vals[:new_pop_cnt]
                pop_size = new_pop_cnt
                
                # Update Archive Size limit
                max_archive_size = int(round(pop_size * r_arc))
                if len(archive) > max_archive_size:
                    del archive[max_archive_size:]

            # --- Parameter Generation ---
            # Select random memory index for each individual
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_sf = memory_sf[r_idx]
            mu_scr = memory_scr[r_idx]
            
            # Generate CR (Normal dist)
            cr = np.random.normal(mu_scr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy dist)
            # Cauchy = tan(pi * (rand - 0.5))
            f = mu_sf + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f = np.where(f <= 0, 0.5, f) # If <= 0, regenerate (simplified to 0.5 here for speed)
            f = np.clip(f, 0, 1)
            
            # --- Mutation: current-to-pbest/1 ---
            # Sort population for p-best selection
            sorted_indices = np.argsort(fitness_vals)
            pop_sorted = population[sorted_indices]
            
            # Create Union of Pop and Archive for second difference vector
            if len(archive) > 0:
                archive_np = np.array(archive)
                pool = np.vstack((population, archive_np))
            else:
                pool = population
            
            new_pop = np.zeros_like(population)
            new_fit = np.zeros_like(fitness_vals)
            
            successful_f = []
            successful_cr = []
            improvement_diffs = []
            
            # Vectorized-like loop (still explicit for clarity/safety with func calls)
            for i in range(pop_size):
                if check_timeout(buffer_sec=0.0): break
                
                x_i = population[i]
                
                # Select p-best
                p_cnt = max(1, int(round(p_best_rate * pop_size)))
                p_idx = np.random.randint(0, p_cnt)
                x_pbest = pop_sorted[p_idx]
                
                # Select r1 (distinct from i)
                r1 = np.random.randint(0, pop_size)
                while r1 == i:
                    r1 = np.random.randint(0, pop_size)
                x_r1 = population[r1]
                
                # Select r2 (distinct from i and r1, from pool)
                r2 = np.random.randint(0, len(pool))
                pool_idx_i = i # Current index in pool (if pool==pop)
                # Logic assumes pool starts with pop. If r2 < pop_size, ensure != i and != r1
                while (r2 < pop_size and (r2 == i or r2 == r1)):
                    r2 = np.random.randint(0, len(pool))
                x_r2 = pool[r2]
                
                # Mutation Equation
                # v = x + F*(pbest - x) + F*(r1 - r2)
                mutant = x_i + f[i] * (x_pbest - x_i) + f[i] * (x_r1 - x_r2)
                
                # Binomial Crossover
                j_rand = np.random.randint(0, dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                trial = np.where(mask, mutant, x_i)
                
                # Bound Handling (Bounce-back)
                # If out of bounds, set value to midpoint between violated bound and old value
                lower_vio = trial < min_b
                upper_vio = trial > max_b
                trial[lower_vio] = (x_i[lower_vio] + min_b[lower_vio]) / 2.0
                trial[upper_vio] = (x_i[upper_vio] + max_b[upper_vio]) / 2.0
                
                # Selection
                f_trial = func(trial)
                
                if f_trial <= fitness_vals[i]:
                    new_pop[i] = trial
                    new_fit[i] = f_trial
                    
                    # Store success info
                    if f_trial < fitness_vals[i]:
                        successful_f.append(f[i])
                        successful_cr.append(cr[i])
                        improvement_diffs.append(fitness_vals[i] - f_trial)
                        
                        # Add replaced to archive
                        archive.append(x_i.copy())
                    
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial.copy()
                else:
                    new_pop[i] = x_i
                    new_fit[i] = fitness_vals[i]
            
            population = new_pop
            fitness_vals = new_fit
            
            # --- Archive Maintenance ---
            max_archive_size = int(round(pop_size * r_arc))
            if len(archive) > max_archive_size:
                # Randomly remove excess
                rem_cnt = len(archive) - max_archive_size
                for _ in range(rem_cnt):
                    archive.pop(np.random.randint(0, len(archive)))

            # --- Memory Update (Weighted Lehmer Mean) ---
            if len(successful_f) > 0:
                weights = np.array(improvement_diffs)
                total_w = np.sum(weights)
                if total_w > 0:
                    weights /= total_w
                    
                    # Mean SCR
                    m_scr = np.sum(weights * np.array(successful_cr))
                    
                    # Mean SF (Lehmer)
                    sf_arr = np.array(successful_f)
                    m_sf = np.sum(weights * (sf_arr ** 2)) / (np.sum(weights * sf_arr) + 1e-10)
                    
                    memory_sf[k_mem_idx] = m_sf
                    memory_scr[k_mem_idx] = m_scr
                    
                    k_mem_idx = (k_mem_idx + 1) % memory_size

            # --- Restart Detection ---
            # 1. Check if population converged (variance is tiny)
            # 2. Check if stagnant for too long
            fit_std = np.std(fitness_vals)
            current_best = np.min(fitness_vals)
            
            if current_best < last_best_fit_in_epoch:
                last_best_fit_in_epoch = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # Triggers: 
            # A. Extreme convergence (std dev near zero)
            # B. Stagnation for significant generations (relative to remaining time)
            # C. Only restart if we have enough time left to make it worth it (>15%)
            if get_remaining_time() > 0.15 * max_time:
                # Tolerance for convergence
                if fit_std < 1e-8 or stagnation_count > 60:
                    # Break inner loop to trigger restart in outer loop
                    break

    # --- Final Polish: Coordinate Descent ---
    # Use whatever remaining time to refine the single best solution
    # along the axes. This is cheaper than Nelder-Mead for high dim.
    
    current_x = best_solution.copy()
    current_f = best_fitness
    
    # Start step size roughly based on domain size
    step_size = np.mean(diff_b) * 0.01 
    min_step = 1e-8
    
    while get_remaining_time() > 0.05 and step_size > min_step:
        improved = False
        for i in range(dim):
            if get_remaining_time() <= 0: break
            
            # Try positive direction
            orig_val = current_x[i]
            current_x[i] = np.clip(orig_val + step_size, min_b[i], max_b[i])
            f_new = func(current_x)
            
            if f_new < current_f:
                current_f = f_new
                best_solution[i] = current_x[i] # Update specific dim
                best_fitness = f_new
                improved = True
                continue
            
            # Try negative direction
            current_x[i] = np.clip(orig_val - step_size, min_b[i], max_b[i])
            f_new = func(current_x)
            
            if f_new < current_f:
                current_f = f_new
                best_solution[i] = current_x[i]
                best_fitness = f_new
                improved = True
            else:
                # Revert
                current_x[i] = orig_val
                
        if not improved:
            step_size *= 0.5
        
    return best_fitness
