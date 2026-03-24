#The following Python code implements a **Success-History based Adaptive Differential Evolution (SHADE)** algorithm.
#
#### Improvements Explained:
#1.  **Adaptive Parameters (SHADE)**: Instead of fixed crossover (`CR`) and mutation (`F`) factors, this algorithm maintains a **memory** of successful parameters. It generates `F` and `CR` for each individual based on historical success, allowing it to adapt to the specific landscape of the function (e.g., separable vs. non-separable, unimodal vs. multimodal).
#2.  **Current-to-pBest Strategy**: It uses the `current-to-pbest/1/bin` mutation strategy. This balances exploration and exploitation better than `current-to-best` by guiding solutions towards a random selection of the top `p%` individuals, reducing the risk of premature convergence to local optima.
#3.  **Weighted Lehmer Mean**: The memory update uses a weighted mean based on fitness improvement magnitudes. Parameters that result in larger fitness gains have a stronger influence on future generations.
#4.  **Stagnation Detection**: Alongside standard deviation convergence, it tracks stagnation (no improvement in best fitness) to trigger restarts sooner, maximizing the search coverage within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Success-History based Adaptive Differential Evolution (SHADE)
    with Restart mechanism.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    # Use a small buffer to ensure safe return
    time_limit = timedelta(seconds=max_time - 0.05)

    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Pre-computation ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')

    # --- SHADE Parameters ---
    H = 5  # Memory size (History length)
    
    # --- Main Restart Loop ---
    while True:
        if check_time():
            return global_best_val

        # 1. Initialization per Restart
        # Population size: adaptive based on dimension, slightly larger for SHADE
        pop_size = int(np.clip(15 * dim, 30, 150))
        
        # Initialize Population (Uniform)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time():
                return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val

        # Initialize Memory
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5)
        mem_k = 0  # Memory index pointer

        # Stagnation tracking
        stag_count = 0
        last_best = np.min(fitness)

        # 2. Evolutionary Loop
        while True:
            if check_time():
                return global_best_val

            # --- Parameter Generation ---
            # Select random memory slot for each individual
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]

            # Generate CR: Normal(m_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)

            # Generate F: Cauchy(m_f, 0.1)
            # F needs to be > 0. If > 1, clamp to 1.
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Retry if F <= 0
            bad_f = f <= 0
            while np.any(bad_f):
                count = np.sum(bad_f)
                f[bad_f] = m_f[bad_f] + 0.1 * np.tan(np.pi * (np.random.rand(count) - 0.5))
                bad_f = f <= 0
            f = np.minimum(f, 1.0)

            # --- Mutation: current-to-pbest/1 ---
            # Sort population by fitness
            sorted_idx = np.argsort(fitness)
            
            # p-best selection: random top p% (between 2/N and 0.2)
            p_curr = np.random.uniform(2/pop_size, 0.2)
            top_cut = int(max(2, pop_size * p_curr))
            top_p_indices = sorted_idx[:top_cut]
            
            # Select pbest for each target
            pbest_indices = np.random.choice(top_p_indices, pop_size)
            x_pbest = pop[pbest_indices]
            
            # Select r1, r2 (random, distinct from target ideally, but collision negligible)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            # Compute Mutant Vector V
            mutant = pop + f[:, None] * (x_pbest - pop) + f[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover: Binomial ---
            mask = np.random.rand(pop_size, dim) < cr[:, None]
            # Enforce taking at least one dimension from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask, mutant, pop)

            # --- Selection & Memory Update Helpers ---
            succ_mask = np.zeros(pop_size, dtype=bool)
            diff_fitness = np.zeros(pop_size)
            
            for i in range(pop_size):
                if check_time():
                    return global_best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial < fitness[i]:
                    succ_mask[i] = True
                    diff_fitness[i] = fitness[i] - f_trial
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
            
            # --- Update History Memory ---
            if np.any(succ_mask):
                good_f = f[succ_mask]
                good_cr = cr[succ_mask]
                delta_f = diff_fitness[succ_mask]
                
                # Weighted Lehmer Mean
                total_delta = np.sum(delta_f)
                if total_delta > 0:
                    weights = delta_f / total_delta
                    
                    # Mean WA (Weighted Arithmetic) for CR
                    mean_cr = np.sum(weights * good_cr)
                    
                    # Mean WL (Weighted Lehmer) for F
                    denom = np.sum(weights * good_f)
                    if denom > 0:
                        mean_f = np.sum(weights * (good_f ** 2)) / denom
                    else:
                        mean_f = 0.5
                    
                    # Update Memory
                    mem_f[mem_k] = np.clip(mean_f, 0, 1)
                    mem_cr[mem_k] = np.clip(mean_cr, 0, 1)
                    mem_k = (mem_k + 1) % H

            # --- Convergence & Stagnation Check ---
            current_best_val = np.min(fitness)
            pop_std = np.std(fitness)
            
            # 1. Population converged
            if pop_std < 1e-8:
                break
            
            # 2. Stagnation (Best value not improving)
            if (last_best - current_best_val) < 1e-8:
                stag_count += 1
            else:
                stag_count = 0
                last_best = current_best_val
            
            # If stagnant for too many generations (relative to population size), restart
            if stag_count > 40:
                break

    return global_best_val
