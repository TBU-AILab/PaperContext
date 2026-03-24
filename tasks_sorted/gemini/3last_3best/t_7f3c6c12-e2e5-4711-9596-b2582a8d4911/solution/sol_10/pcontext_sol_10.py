#The following Python algorithm implements **SHADE-TimeAdaptive**, a highly optimized version of the SHADE algorithm designed specifically for time-constrained black-box optimization.
#
#### Key Improvements
#1.  **Time-Adaptive Exploration/Exploitation**: The algorithm dynamically adjusts the "p-best" selection parameter ($p$) based on the elapsed time. It starts with high exploration ($p=0.2$) and linearly decays to high exploitation ($p=0.05$) as the deadline approaches, ensuring the search converges efficiently within the time limit.
#2.  **Robust Soft Restart**: A restart mechanism is triggered by either population convergence (low standard deviation) or stagnation (no improvement in best fitness for 60 generations). The restart preserves the elite solution (Global Best) to maintain progress while randomizing the rest of the population and resetting SHADE's historical memory to explore new basins of attraction unbiasedly.
#3.  **Memory-Efficient Archive Handling**: The mutation strategy utilizes an external archive of inferior solutions to maintain diversity. The implementation optimizes the selection process (avoiding costly memory concatenations like `np.vstack`) to maximize the number of evaluations per second.
#4.  **Batch Time Checking**: To minimize system overhead, the strict time check is performed once per batch during initialization and once per generation during the loop, rather than for every individual evaluation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using SHADE-TimeAdaptive: A Self-Adaptive Differential Evolution 
    with Time-Based p-best Decay and Stagnation Restarts.
    """
    # --- Time Management ---
    start_time = datetime.now()
    # Reserve a small buffer (0.05s) to guarantee return before timeout
    deadline = start_time + timedelta(seconds=max_time - 0.05)
    
    # --- Configuration ---
    # Population Size: Adapted to dimension (18*D), clamped [30, 150]
    # Small enough for fast generations, large enough for reliable convergence
    pop_size = int(np.clip(18 * dim, 30, 150))
    
    # Archive Size: Stores historical bad solutions to maintain diversity
    archive_size = int(2.0 * pop_size)
    
    # SHADE Memory Parameters (History size H=5)
    H = 5
    mem_cr = np.full(H, 0.5) # Crossover Rate Memory
    mem_f = np.full(H, 0.5)  # Scaling Factor Memory
    k_mem = 0                # Memory Pointer
    
    # Pre-process Bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    best_fit = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    # Check time in batches to reduce overhead
    for i in range(pop_size):
        if (i % 10 == 0) and (datetime.now() >= deadline):
            return best_fit if best_sol is not None else float('inf')
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_sol = pop[i].copy()
            
    # Restart & Stagnation Counters
    no_improv_count = 0
    last_best_fit = best_fit
    
    # --- Main Optimization Loop ---
    while datetime.now() < deadline:
        
        # 1. Restart Mechanism
        # Triggered if population loses diversity OR fitness stagnates
        std_fit = np.std(fitness)
        
        if best_fit < last_best_fit:
            no_improv_count = 0
            last_best_fit = best_fit
        else:
            no_improv_count += 1
            
        if std_fit < 1e-9 or no_improv_count > 60:
            # --- Perform Soft Restart ---
            # Preserve the Elite (Global Best)
            elite_sol = best_sol.copy()
            elite_fit = best_fit
            
            # Re-initialize the rest of the population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = elite_sol # Inject elite
            
            fitness[:] = float('inf')
            fitness[0] = elite_fit
            
            # Reset SHADE Memory & Archive (Start fresh learning)
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            k_mem = 0
            n_archive = 0
            no_improv_count = 0
            
            # Evaluate new population (skip elite at index 0)
            for i in range(1, pop_size):
                if (i % 10 == 0) and (datetime.now() >= deadline): return best_fit
                val = func(pop[i])
                fitness[i] = val
                if val < best_fit:
                    best_fit = val
                    best_sol = pop[i].copy()
            continue
        
        # 2. Time-Adaptive Strategy
        # Decay p-best rate from 0.2 (exploration) to 0.05 (exploitation)
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = min(1.0, elapsed / max_time)
        p_best_rate = 0.2 - 0.15 * progress 
        
        # 3. Parameter Generation (Vectorized)
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_idx]
        m_f = mem_f[r_idx]
        
        # CR ~ Normal(m_cr, 0.1), clipped [0, 1]
        cr = np.clip(np.random.normal(m_cr, 0.1), 0, 1)
        
        # F ~ Cauchy(m_f, 0.1)
        f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        # Constrain F: >1 clamps to 1, <=0 clamps to 0.1 (ensure mutation)
        f = np.where(f > 1.0, 1.0, f)
        f = np.where(f <= 0.0, 0.1, f)
        
        # 4. Mutation: current-to-pbest/1
        # Sort population to identify p-best
        sorted_idx = np.argsort(fitness)
        num_pbest = max(2, int(pop_size * p_best_rate))
        
        # Select random p-best for each individual
        pbest_indices = sorted_idx[np.random.randint(0, num_pbest, pop_size)]
        x_pbest = pop[pbest_indices]
        
        # Select r1 (random distinct from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        collision = (r1_indices == np.arange(pop_size))
        r1_indices[collision] = (r1_indices[collision] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (random from Union of Pop and Archive)
        # Optimized to avoid creating large copies (vstack)
        r2_indices = np.random.randint(0, pop_size + n_archive, pop_size)
        x_r2 = np.zeros((pop_size, dim))
        
        mask_pop = r2_indices < pop_size
        x_r2[mask_pop] = pop[r2_indices[mask_pop]]
        
        if n_archive > 0:
            mask_arc = ~mask_pop
            arc_indices = r2_indices[mask_arc] - pop_size
            x_r2[mask_arc] = archive[arc_indices]
        else:
            # Fallback if archive empty (map out-of-bounds indices back to pop)
            x_r2[~mask_pop] = pop[r2_indices[~mask_pop] % pop_size]

        # Calculate Mutant Vector
        f_vec = f[:, np.newaxis]
        mutant = pop + f_vec * (x_pbest - pop) + f_vec * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask = rand_vals < cr[:, np.newaxis]
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, pop)
        
        # 6. Bound Constraints (Clipping)
        # Robust method for general black-box functions
        trial = np.clip(trial, min_b, max_b)
        
        # 7. Selection & Memory Update
        success_f = []
        success_cr = []
        diffs = []
        
        # Time check before evaluation loop
        if datetime.now() >= deadline: return best_fit
        
        for i in range(pop_size):
            f_trial = func(trial[i])
            
            # Greedy Selection
            if f_trial < fitness[i]:
                diff = fitness[i] - f_trial
                
                # Update Archive with replaced parent
                if n_archive < archive_size:
                    archive[n_archive] = pop[i].copy()
                    n_archive += 1
                else:
                    rep_idx = np.random.randint(0, archive_size)
                    archive[rep_idx] = pop[i].copy()
                
                # Update Population
                fitness[i] = f_trial
                pop[i] = trial[i]
                
                # Store success data
                success_f.append(f[i])
                success_cr.append(cr[i])
                diffs.append(diff)
                
                # Update Global Best
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_sol = trial[i].copy()
                    
        # 8. Adaptive Memory Update
        if len(success_f) > 0:
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            s_diff = np.array(diffs)
            
            # Weights based on fitness improvement magnitude
            weights = s_diff / np.sum(s_diff)
            
            # Weighted Mean for CR
            mean_cr = np.sum(weights * s_cr)
            mean_cr = np.clip(mean_cr, 0, 1)
            
            # Weighted Lehmer Mean for F
            num = np.sum(weights * (s_f ** 2))
            den = np.sum(weights * s_f)
            mean_f = num / den if den > 0 else 0.5
            mean_f = np.clip(mean_f, 0, 1)
            
            # Update Memory
            mem_cr[k_mem] = mean_cr
            mem_f[k_mem] = mean_f
            k_mem = (k_mem + 1) % H
            
    return best_fit
