#The following algorithm is an implementation of **Time-Adaptive L-SHADE** (Linear Population Size Reduction Success-History Adaptive Differential Evolution).
#
#### Key Improvements:
#1.  **Time-Adaptive Population Reduction**: Standard L-SHADE reduces population size based on the number of function evaluations (NFE). Since we are constrained by time, not NFE, this algorithm dynamically adjusts the population size based on the ratio of elapsed time to `max_time`. This ensures the algorithm transitions from exploration (large population) to exploitation (small population) exactly as the deadline approaches.
#2.  **Dynamic Restart Mechanism**: If the population converges (variance drops below a threshold) or the population size hits the minimum limit while time remains, the algorithm triggers a "Soft Restart." It resets the population size but injects the global best solution to preserve progress.
#3.  **Robust Parameter Adaptation**: Uses the Success-History Adaptation for control parameters $F$ (Scaling Factor) and $CR$ (Crossover Rate), utilizing a weighted Lehmer mean to favor parameters that produced high-fitness improvements.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Time-Adaptive L-SHADE with Soft Restarts.
    
    This algorithm adapts the highly effective L-SHADE strategy to a time-based 
    budget. It linearly reduces population size as time progresses to shift 
    focus from exploration to exploitation, and restarts if convergence occurs 
    early.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper Functions ---
    def get_time_elapsed():
        return (datetime.now() - start_time).total_seconds()

    def check_limit():
        return datetime.now() - start_time >= time_limit

    # --- Problem Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Parameters ---
    # Initial population size: High enough for exploration, capped for speed
    # Typical L-SHADE uses 18*dim, we cap at 200 to ensure speed in high dims
    initial_pop_size = int(np.clip(18 * dim, 30, 200)) 
    min_pop_size = 4
    
    # External Archive size factor (Archive size = arc_rate * pop_size)
    arc_rate = 2.0  
    
    # Memory size for SHADE
    memory_size = 5
    
    # --- Global Best Tracking ---
    best_val = float('inf')
    best_sol = None
    
    # --- Outer Loop (Restarts) ---
    # We treat the available time as a resource. If we converge early, we restart
    # but strictly manage the remaining time for the next 'epoch'.
    
    while not check_limit():
        
        # 1. Initialize Population
        pop_size = initial_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Soft Restart: Inject best solution found so far
        start_idx = 0
        if best_sol is not None:
            pop[0] = best_sol.copy()
            fitness[0] = best_val
            start_idx = 1
        
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if check_limit(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
                
        # Initialize Memory
        m_cr = np.full(memory_size, 0.5)
        m_f = np.full(memory_size, 0.5)
        k_mem = 0
        
        # Initialize Archive
        archive = []
        
        # Track start of this epoch for local linear reduction
        epoch_start_time = get_time_elapsed()
        
        # --- Evolutionary Cycle ---
        while not check_limit():
            
            # 2. Time-Based Population Reduction (L-SHADE Strategy)
            # Calculate progress ratio (0.0 to 1.0) relative to total max_time
            # Note: We use total max_time to guide the reduction globally.
            # If we restarted, we are effectively refining the search with higher pressure.
            curr_time = get_time_elapsed()
            total_time_seconds = max_time
            
            # Progress ratio
            progress = curr_time / total_time_seconds
            
            # Target population size based on linear reduction
            # size = N_init - (N_init - N_min) * progress
            target_size = int(round(initial_pop_size - (initial_pop_size - min_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            # If current pop > target, remove worst individuals
            if pop_size > target_size:
                # Sort by fitness (descending is worst for minimization, so we remove end)
                # Actually argsort gives indices of smallest to largest.
                # Worst are at the end.
                sorted_indices = np.argsort(fitness)
                
                # Keep top 'target_size'
                keep_indices = sorted_indices[:target_size]
                
                pop = pop[keep_indices]
                fitness = fitness[keep_indices]
                pop_size = target_size
                
                # Adjust archive size relative to new pop_size
                curr_arc_cap = int(pop_size * arc_rate)
                if len(archive) > curr_arc_cap:
                    # Randomly remove excess
                    del_count = len(archive) - curr_arc_cap
                    # Create mask or just pop random elements
                    for _ in range(del_count):
                        archive.pop(np.random.randint(0, len(archive)))

            # 3. Parameter Adaptation
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_cr = m_cr[r_idx]
            mu_f = m_f[r_idx]
            
            # Generate CR (Normal Distribution, clipped)
            # If mu_cr = -1 (terminal), cr = 0. 
            # In standard SHADE, close to 1 is usually preferred later on, 
            # but we use standard normal generation.
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)
            
            # Generate F (Cauchy Distribution)
            # F = mu_f + 0.1 * tan(pi * (rand - 0.5))
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Check F constraints
            # If F > 1, clip to 1. If F <= 0, regenerate.
            while np.any(f <= 0):
                mask = f <= 0
                f[mask] = mu_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f = np.clip(f, 0, 1)
            
            # 4. Mutation (current-to-pbest/1)
            # p decreases linearly from 0.2 to 0.05 approx (jSO strategy) or fixed (SHADE)
            # We use a fixed range [2/N, 0.2]
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            p_val = np.clip(p_val, 2.0/pop_size, 0.2)
            
            # Sort for pbest selection
            sorted_indices = np.argsort(fitness)
            num_pbest = max(2, int(pop_size * p_val))
            pbest_indices = sorted_indices[:num_pbest]
            
            # Vectors needed
            idx_pbest = np.random.choice(pbest_indices, pop_size)
            x_pbest = pop[idx_pbest]
            
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            # Ensure r1 != i (omitted for speed as probability is low with reasonable N)
            x_r1 = pop[idx_r1]
            
            # Select r2 from Union(Pop, Archive)
            if len(archive) > 0:
                # Convert archive to array only once per gen to save time
                arc_np = np.array(archive)
                union_pop = np.vstack((pop, arc_np))
            else:
                union_pop = pop
                
            idx_r2 = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[idx_r2]
            
            # Mutation Equation: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover (Binomial)
            j_rand = np.random.randint(0, dim, pop_size)
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < cr[:, None]
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 6. Boundary Handling (Clipping)
            # Clipping is robust. Some papers suggest reflection, 
            # but clipping works better with current-to-pbest.
            trial = np.clip(trial, min_b, max_b)
            
            # 7. Selection & Memory Update
            success_f = []
            success_cr = []
            diff_fitness = []
            
            # Evaluate offspring
            for i in range(pop_size):
                if check_limit(): return best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    # Solution improved or equal
                    if f_trial < fitness[i]:
                        # Strict improvement details
                        archive.append(pop[i].copy())
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        diff_fitness.append(fitness[i] - f_trial)
                    
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = pop[i].copy()
                        
            # Resize Archive if it grew too big during loop
            curr_arc_cap = int(pop_size * arc_rate)
            while len(archive) > curr_arc_cap:
                archive.pop(np.random.randint(0, len(archive)))
                
            # Update SHADE Memory (Weighted Lehmer Mean)
            if len(success_f) > 0:
                w = np.array(diff_fitness)
                total_w = np.sum(w)
                # Avoid div by zero
                if total_w > 0:
                    w = w / total_w
                    
                    f_succ = np.array(success_f)
                    cr_succ = np.array(success_cr)
                    
                    # Lehmer Mean for F
                    mean_f = np.sum(w * (f_succ ** 2)) / np.sum(w * f_succ)
                    
                    # Arithmetic Mean for CR (weighted)
                    mean_cr = np.sum(w * cr_succ)
                    
                    # Update memory at pointer
                    m_f[k_mem] = 0.5 * m_f[k_mem] + 0.5 * mean_f # Smoothing (optional but stable)
                    m_cr[k_mem] = 0.5 * m_cr[k_mem] + 0.5 * mean_cr
                    
                    k_mem = (k_mem + 1) % memory_size
            
            # 8. Check for Convergence / Restart Conditions
            # If population variance is extremely low, we are stuck.
            # Or if population size has reached minimum and we are just idling.
            
            # Calculate std dev of fitness
            fit_std = np.std(fitness)
            
            # Conditions to break inner loop and trigger restart:
            # A. Variance is negligible (converged)
            # B. Population is minimal (finished L-SHADE cycle) AND variance is low-ish
            if fit_std < 1e-9:
                break
            
            # If we are at min population size, the linear reduction cycle is done.
            # If we still have time, we should restart to explore elsewhere 
            # or refine with new parameters.
            if pop_size <= min_pop_size:
                break
                
    return best_val
