#The proposed algorithm is a **Memetic Algorithm** that hybridizes **L-SHADE** (Linear Population Size Reduction Success-History Adaptive Differential Evolution) with **MTS-LS1** (Multiple Trajectory Search - Local Search 1). 
#
#### Key Improvements:
#1.  **Time-Adaptive Population Reduction**: Utilizes the `max_time` constraint to linearly reduce the population size from a large value (exploration) to a small value (exploitation), ensuring the computational budget is used effectively.
#2.  **Stagnation-Triggered Local Search (MTS-LS1)**: Integrates a coordinate-descent-based local search (MTS-LS1) that is triggered when the global search (DE) stagnates. This allows for rapid refinement of the best solution in steep valleys where standard mutation might fail.
#3.  **Dynamic Search Range**: The local search maintains a dynamic search range (`sr`) for each dimension, halving it when no improvement is found, allowing it to "zoom in" on the optimum.
#4.  **Soft Restarts with Elitism**: If the population converges or the search ranges become too small, the algorithm restarts with a new random population but injects the global best solution to preserve progress.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Memetic L-SHADE with MTS-LS1 Local Search and Soft Restarts.
    
    Combines the global search power of L-SHADE with the fine-tuning capability
    of MTS-LS1 (Coordinate Descent).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper Functions ---
    def check_limit():
        return datetime.now() - start_time >= time_limit
    
    def get_progress():
        """Returns time progress ratio [0.0, 1.0]"""
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)
    
    # --- Problem Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Parameters ---
    # L-SHADE Configuration
    initial_pop_size = int(np.clip(18 * dim, 30, 200))
    min_pop_size = 4
    arc_rate = 2.6
    memory_size = 6
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Main Loop (Restarts) ---
    while not check_limit():
        
        # 1. Initialize Population
        pop_size = initial_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Soft Restart: Inject best found solution to preserve progress
        start_eval_idx = 0
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_val
            start_eval_idx = 1
        
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_limit(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
        
        # Initialize SHADE Memory
        M_CR = np.full(memory_size, 0.5)
        M_F = np.full(memory_size, 0.5)
        k_mem = 0
        archive = []
        
        # Initialize Local Search (MTS-LS1) parameters
        # Search range for each dimension
        sr = diff_b * 0.4
        stag_count = 0
        
        # 2. Evolutionary Cycle
        while not check_limit():
            
            # --- Linear Population Size Reduction ---
            # Calculate target size based on time progress
            progress = get_progress()
            target_size = int(round(initial_pop_size - (initial_pop_size - min_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Remove worst individuals
                sorted_indices = np.argsort(fitness)
                keep_indices = sorted_indices[:target_size]
                
                pop = pop[keep_indices]
                fitness = fitness[keep_indices]
                pop_size = target_size
                
                # Resize Archive
                curr_arc_cap = int(pop_size * arc_rate)
                if len(archive) > curr_arc_cap:
                    del_count = len(archive) - curr_arc_cap
                    # Remove random elements
                    for _ in range(del_count):
                        archive.pop(np.random.randint(0, len(archive)))
            
            # --- Parameter Generation (SHADE) ---
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            # Handle F <= 0 (regenerate)
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = mu_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f = np.clip(f, 0, 1)
            
            # --- Mutation: current-to-pbest/1 ---
            sorted_indices = np.argsort(fitness)
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            num_pbest = int(max(2, pop_size * p_val))
            pbest_indices = sorted_indices[:num_pbest]
            
            # Select pbest, r1, r2
            idx_pbest = np.random.choice(pbest_indices, pop_size)
            x_pbest = pop[idx_pbest]
            
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[idx_r1]
            
            # r2 from Union(Population, Archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            idx_r2 = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[idx_r2]
            
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            rand_vals = np.random.rand(pop_size, dim)
            j_rand = np.random.randint(0, dim, pop_size)
            mask = rand_vals < cr[:, None]
            mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            succ_f = []
            succ_cr = []
            diff_fitness = []
            any_improvement = False
            
            for i in range(pop_size):
                if check_limit(): return best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        archive.append(pop[i].copy())
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        diff_fitness.append(fitness[i] - f_trial)
                        any_improvement = True
                    
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = pop[i].copy()
                        stag_count = 0 # Reset stagnation
            
            # Resize Archive if needed
            curr_arc_cap = int(pop_size * arc_rate)
            while len(archive) > curr_arc_cap:
                archive.pop(np.random.randint(0, len(archive)))
            
            # Update SHADE Memory
            if len(succ_f) > 0:
                w = np.array(diff_fitness)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    mean_f = np.sum(w * (np.array(succ_f)**2)) / np.sum(w * np.array(succ_f))
                    mean_cr = np.sum(w * np.array(succ_cr))
                    
                    M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                    M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
                    k_mem = (k_mem + 1) % memory_size
            
            if not any_improvement:
                stag_count += 1
            
            # --- Local Search (MTS-LS1) ---
            # Trigger: Stagnation (Global search stuck)
            # We apply coordinate descent on the best solution
            if stag_count > 5:
                # Randomize dimension order
                dims_check = np.random.permutation(dim)
                # Cap dimensions to check to ensure speed in high-dim
                if dim > 50:
                    dims_check = dims_check[:50]
                
                ls_improved = False
                # Work on a copy of best_sol
                curr_best_x = best_sol.copy()
                curr_best_f = best_val
                
                for d in dims_check:
                    if check_limit(): return best_val
                    
                    original_val = curr_best_x[d]
                    
                    # MTS-LS1: Try moving x[d] - sr[d]
                    curr_best_x[d] = np.clip(original_val - sr[d], min_b[d], max_b[d])
                    val = func(curr_best_x)
                    
                    if val < curr_best_f:
                        curr_best_f = val
                        best_val = val
                        best_sol = curr_best_x.copy()
                        ls_improved = True
                    else:
                        # MTS-LS1: Try moving x[d] + 0.5 * sr[d]
                        curr_best_x[d] = np.clip(original_val + 0.5 * sr[d], min_b[d], max_b[d])
                        val = func(curr_best_x)
                        
                        if val < curr_best_f:
                            curr_best_f = val
                            best_val = val
                            best_sol = curr_best_x.copy()
                            ls_improved = True
                        else:
                            # Both failed: Restore x and reduce sr[d]
                            curr_best_x[d] = original_val
                            sr[d] *= 0.5
                
                if ls_improved:
                    stag_count = 0
                    # Sync best solution back into population
                    best_idx = np.argmin(fitness)
                    pop[best_idx] = best_sol.copy()
                    fitness[best_idx] = best_val
            
            # --- Restart Conditions ---
            # 1. Population variance is extremely low (Convergence)
            # 2. Search range (sr) is exhausted (Local Search Convergence)
            # 3. Population size is minimal and still stagnating
            
            std_fit = np.std(fitness)
            if std_fit < 1e-9 or np.all(sr < 1e-12) or (pop_size <= min_pop_size and stag_count > 10):
                break
                
    return best_val
