#Here is a fully operational, self-contained implementation of **L-SHADE with Coordinate Descent (Hybrid Memetic Algorithm)**.
#
#### Improvements over the previous version:
#1.  **L-SHADE Architecture**: Instead of JADE's single moving average for parameters, this uses a **History Memory** ($H$) of successful parameters ($F$ and $CR$). This allows the algorithm to recall multiple successful search strategies simultaneously (e.g., some agents exploring with high $F$, others exploiting with low $F$).
#2.  **Linear Population Size Reduction (LPSR)**: The population size starts large to maximize exploration and linearly decreases over time. This shifts computational resources from exploration (early stage) to exploitation (late stage) significantly improving convergence speed.
#3.  **Hybrid Local Search (Coordinate Descent)**: Evolutionary algorithms are great at finding the "basin" of the global minimum but slow to refine the final decimals. This implementation injects a greedy **Coordinate Descent** step whenever a new global best is found, rapidly polishing the solution.
#4.  **Weighted Lehmer Mean**: Uses a more statistically robust method for updating parameter memories based on fitness improvement magnitude.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE (Linear Success-History Adaptive Differential Evolution) 
    with Coordinate Descent Local Search.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    def get_progress():
        """Returns normalized time progress (0.0 to 1.0)"""
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        return min(1.0, elapsed / max_time)

    def is_time_up():
        return datetime.now() >= end_time

    # --- Configuration ---
    # L-SHADE Population Sizing
    # Start with a larger population for exploration, reduce to min_pop
    r_N_init = 18 * dim 
    r_N_min = 4
    pop_size = r_N_init
    
    # Memory for Adaptive Parameters (History size H=5)
    mem_size = 5
    memory_sf = np.full(mem_size, 0.5) # Memory for Scaling Factor F
    memory_scr = np.full(mem_size, 0.5) # Memory for Crossover Rate CR
    mem_k = 0 # cyclic index for memory update

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Latin Hypercube Sampling for initial population
    population = np.empty((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(0, 1, pop_size + 1)
        offsets = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(offsets)
        population[:, d] = min_b[d] + offsets * diff_b[d]

    fitness = np.full(pop_size, float('inf'))
    
    # Best Solution Tracking
    best_val = float('inf')
    best_vec = np.zeros(dim)

    # Initial Evaluation
    for i in range(pop_size):
        if is_time_up(): return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()

    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]

    # --- Local Search Helper (Coordinate Descent) ---
    def local_search(current_best_vec, current_best_val, step_size_ratio=0.01):
        """
        Greedy local search around the best solution. 
        Only runs for a limited budget per invocation.
        """
        vec = current_best_vec.copy()
        val = current_best_val
        improved = False
        
        # Determine step size based on bounds
        step = diff_b * step_size_ratio
        
        # Random permutation of dimensions to search
        dims_to_search = np.random.permutation(dim)
        
        for d in dims_to_search:
            if is_time_up(): break
            
            # Try moving positive direction
            vec[d] += step[d]
            # Clip
            if vec[d] > max_b[d]: vec[d] = max_b[d]
            
            new_val = func(vec)
            if new_val < val:
                val = new_val
                current_best_vec[d] = vec[d]
                improved = True
                continue # Greedy: Found improvement, move to next dim
            
            # Revert and try negative direction
            vec[d] -= 2 * step[d] # Move to original - step
            # Clip
            if vec[d] < min_b[d]: vec[d] = min_b[d]
            
            new_val = func(vec)
            if new_val < val:
                val = new_val
                current_best_vec[d] = vec[d]
                improved = True
            else:
                # Revert change
                vec[d] += step[d]
                
        return vec, val, improved

    # --- Main Loop ---
    while not is_time_up():
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate allowed population size based on time progress
        progress = get_progress()
        new_pop_size = int(round((r_N_min - r_N_init) * progress + r_N_init))
        
        if new_pop_size < pop_size:
            # Shrink population (remove worst solutions, array is already sorted)
            pop_size = new_pop_size
            population = population[:pop_size]
            fitness = fitness[:pop_size]
        
        # 2. Parameter Generation
        # Select random memory index for each individual
        r_idx = np.random.randint(0, mem_size, pop_size)
        m_cr = memory_scr[r_idx]
        m_f = memory_sf[r_idx]
        
        # Generate CR: Normal(mean=M_cr, std=0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F: Cauchy(loc=M_f, scale=0.1)
        # If F > 1 clamp to 1, if F <= 0 regenerate
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints vectorized
        f[f > 1] = 1.0
        # Retry logic for F <= 0 (simplified: clamp to small value)
        f[f <= 0] = 0.5 

        # 3. Mutation: current-to-pbest/1
        # p_best rate also varies slightly to encourage exploitation late game
        p_rate = max(2.0/pop_size, 0.2 * (1 - progress))
        p_num = max(1, int(round(pop_size * p_rate)))
        
        # Choose pbest from top p% individuals
        pbest_indices = np.random.randint(0, p_num, pop_size)
        x_pbest = population[pbest_indices]
        
        # Choose r1, r2 distinct
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Current sorted population allows index 0 to be best.
        # Vectorized mutation: V = Xi + F*(Xpbest - Xi) + F*(Xr1 - Xr2)
        F_col = f[:, np.newaxis]
        mutants = population + F_col * (x_pbest - population) + F_col * (population[r1] - population[r2])
        
        # 4. Crossover: Binomial
        j_rand = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
        mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutants, population)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 5. Selection
        fitness_trial = np.zeros(pop_size)
        
        # To store successful parameters
        good_cr = []
        good_f = []
        diff_fitness = []
        
        for i in range(pop_size):
            if is_time_up(): return best_val
            
            f_trial_val = func(trial_pop[i])
            fitness_trial[i] = f_trial_val
            
            fit_diff = fitness[i] - f_trial_val
            
            # Selection (Greedy)
            if f_trial_val <= fitness[i]:
                # Update individual
                population[i] = trial_pop[i]
                fitness[i] = f_trial_val
                
                # Check Global Best
                if f_trial_val < best_val:
                    best_val = f_trial_val
                    best_vec = trial_pop[i].copy()
                    
                    # TRIGGER LOCAL SEARCH
                    # If we found a new basin, refine it immediately
                    ls_vec, ls_val, improved = local_search(best_vec, best_val, step_size_ratio=0.001)
                    if improved:
                        best_val = ls_val
                        best_vec = ls_vec
                        population[i] = ls_vec # Push refined back to pop
                        fitness[i] = ls_val

                # Store success info for adaptation
                if fit_diff > 0:
                    good_cr.append(cr[i])
                    good_f.append(f[i])
                    diff_fitness.append(fit_diff)
        
        # 6. Adaptive Parameter Update (Memory)
        num_success = len(good_f)
        if num_success > 0:
            good_f = np.array(good_f)
            good_cr = np.array(good_cr)
            diff_fitness = np.array(diff_fitness)
            
            # Weighted Lehmer Mean for F
            # Mean = sum(w * f^2) / sum(w * f)
            total_diff = np.sum(diff_fitness)
            if total_diff == 0: weights = np.ones(num_success) / num_success
            else: weights = diff_fitness / total_diff
            
            mean_scr = np.sum(weights * good_cr)
            
            denom_f = np.sum(weights * good_f)
            if denom_f == 0: mean_sf = 0.5
            else: mean_sf = np.sum(weights * (good_f ** 2)) / denom_f
            
            # Update Memory (at index k)
            memory_sf[mem_k] = mean_sf
            memory_scr[mem_k] = mean_scr
            
            mem_k = (mem_k + 1) % mem_size
            
            # Sort population for next iteration (required for pbest logic)
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Sync best_vec to ensure consistency
            if fitness[0] < best_val:
                best_val = fitness[0]
                best_vec = population[0].copy()

    return best_val
