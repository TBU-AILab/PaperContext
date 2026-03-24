#4. output value is: 28.5218732918451
#
# 4. algorithm code is:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    SHADE (Success-History based Adaptive Differential Evolution) with Restarts.
    
    This algorithm improves upon standard DE and L-SHADE by:
    1. Using the SHADE strategy (current-to-pbest/1) which balances greediness and diversity better than rand/1 or best/1.
    2. Employing Success-History Adaptation for F and CR parameters, eliminating manual tuning.
    3. Implementing a Restart Mechanism: detecting stagnation (low population variance or no improvement) 
       and restarting the search while preserving the elite solution. This prevents getting stuck in local optima.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Fixed moderate size to ensure high generation count within limited time.
    # We clip between 30 and 80. Small enough for speed, large enough for diversity.
    pop_size = int(np.clip(10 * dim, 30, 80))
    
    # SHADE Memory size
    memory_size = 5
    
    # ---------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Helper for time checking
    def check_limit():
        return datetime.now() - start_time >= time_limit

    # Global Best tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Main Optimization Loop (Restarts) ---
    while not check_limit():
        
        # 1. Initialization for new Epoch
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Inject global best into the new population to preserve elite (Soft Restart)
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol
            fitness[0] = best_val
            start_idx = 1
            
        # Evaluate initial population
        for i in range(start_idx, pop_size):
            if check_limit(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
        
        # Memory Initialization (Historical Success)
        m_cr = np.full(memory_size, 0.5)
        m_f = np.full(memory_size, 0.5)
        mem_k = 0
        
        # Archive (Stores inferior solutions replaced by better ones to maintain diversity)
        archive = []
        archive_capacity = pop_size
        
        # Stagnation detection counters
        gens_no_improv = 0
        
        # --- Epoch Loop ---
        while not check_limit():
            # 2. Parameter Adaptation
            # Select random memory index for each individual
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Retry if F <= 0 (Vectorized)
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = mu_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f = np.clip(f, 0, 1) # Cap F at 1.0
            
            # 3. Mutation: current-to-pbest/1
            # Sort population to find p-best
            sorted_indices = np.argsort(fitness)
            
            # Randomize p in [0.05, 0.2] (Standard practice for robustness in SHADE variants)
            p = np.random.uniform(0.05, 0.2)
            top_p_count = max(2, int(pop_size * p))
            top_p_indices = sorted_indices[:top_p_count]
            
            # Select x_pbest randomly from top p%
            x_pbest = pop[np.random.choice(top_p_indices, pop_size)]
            
            # Select x_r1 from population (random)
            r1 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1]
            
            # Select x_r2 from Union(Population, Archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            
            r2 = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[r2]
            
            # Compute Mutant Vector: v = x + F*(pbest - x) + F*(r1 - r2)
            # F needs to be column vector for broadcasting
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, None]
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 5. Boundary Handling (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # 6. Selection & Memory Update
            success_f = []
            success_cr = []
            improv_diff = []
            
            for i in range(pop_size):
                if check_limit(): return best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    # If strictly better, store success info and archive parent
                    if f_trial < fitness[i]:
                        archive.append(pop[i].copy())
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        improv_diff.append(fitness[i] - f_trial)
                    
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = pop[i].copy()
            
            # Resize Archive if too big
            while len(archive) > archive_capacity:
                archive.pop(np.random.randint(0, len(archive)))
            
            # Update Memory (Weighted Lehmer Mean)
            if len(success_f) > 0:
                w = np.array(improv_diff)
                total_w = np.sum(w)
                if total_w > 0:
                    w = w / total_w
                    f_arr = np.array(success_f)
                    cr_arr = np.array(success_cr)
                    
                    mean_f = np.sum(w * (f_arr**2)) / np.sum(w * f_arr)
                    mean_cr = np.sum(w * cr_arr)
                    
                    m_f[mem_k] = mean_f
                    m_cr[mem_k] = mean_cr
                    mem_k = (mem_k + 1) % memory_size
                
                gens_no_improv = 0
            else:
                gens_no_improv += 1
            
            # 7. Restart Check
            # Restart if population variance is low (convergence) or no improvement for too long
            # This allows the algorithm to re-explore from a fresh state (except elite)
            if np.std(fitness) < 1e-6 or gens_no_improv > 35:
                break 
                
    return best_val
