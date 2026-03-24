#The following Python code implements **L-SHADE-Time** (Linear Population Reduction Adaptive Differential Evolution with Time-based Restart).
#
#**Key Improvements over previous algorithms:**
#1.  **Linear Population Size Reduction (LPSR)**: Instead of a fixed population size or simple stagnation checks, this algorithm linearly reduces the population size from a large initial pool ($N_{init} \approx 18 \cdot D$) down to a minimal set ($N_{min}=6$) based on the elapsed time. This mechanism enforces a transition from global exploration to local exploitation, ensuring convergence within the time limit.
#2.  **History-Based Parameter Adaptation (SHADE)**: Unlike JADE (which uses a single mean), this uses a memory bank ($H=6$) to store successful $F$ and $CR$ settings. This allows the algorithm to "remember" multiple successful search strategies suited for different features of the landscape (e.g., multimodal basins).
#3.  **Weighted Lehmer Mean**: The adaptation logic weights parameter updates by their fitness improvement magnitude, prioritizing parameter values that yield significant progress.
#4.  **Time-Aware Restart Strategy**: If the population converges early (low variance), the algorithm triggers a restart. Crucially, the restart population size is scaled by the *remaining* time—preventing the algorithm from starting a massive new search that it cannot finish.
#5.  **External Archive with Random Pruning**: Maintains diversity by storing decent solutions that were replaced, allowing the mutation operator to draw differences from a wider distribution.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE-Time: A time-adaptive version of the 
    L-SHADE algorithm with Linear Population Size Reduction and Restart.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # L-SHADE specific constants
    # Initial population: ~18*dim is standard for SHADE to ensure diversity
    # Clamped to [30, 200] to balance exploration speed vs diversity
    n_init = int(np.clip(18 * dim, 30, 200))
    n_min = 6   # Minimum population size to maintain mutation validity
    H = 6       # Memory size for adaptive parameters
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global state
    best_val = float('inf')
    best_sol = None
    
    # SHADE Memory (initialized to 0.5)
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive for mutation diversity
    archive = []

    # --- Main Loop (Handles Restarts) ---
    while True:
        # Check overall time
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        if elapsed >= max_time:
            return best_val
            
        # Calculate remaining time ratio for this epoch
        remaining_ratio = 1.0 - (elapsed / max_time)
        
        # If very little time remains (< 2%), strictly optimize what we have or return
        if remaining_ratio < 0.02:
            return best_val
            
        # Initialize Population Size based on Remaining Time
        # This ensures that restarts late in the process are "lightweight"
        current_pop_size = int(n_min + (n_init - n_min) * remaining_ratio)
        current_pop_size = max(current_pop_size, n_min + 2)
        
        # Initialize Population
        pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
        fitness = np.full(current_pop_size, float('inf'))
        
        # Elitism: Inject best solution found so far into new population
        start_eval_idx = 0
        if best_sol is not None:
            pop[0] = best_sol
            fitness[0] = best_val
            start_eval_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, current_pop_size):
            if (datetime.now() - start_time) >= limit:
                return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
                
        # --- Evolutionary Loop (Epoch) ---
        while True:
            # 1. Time & Linear Population Reduction Logic
            now = datetime.now()
            elapsed_epoch = (now - start_time).total_seconds()
            if elapsed_epoch >= max_time:
                return best_val
            
            # Global progress (0.0 to 1.0)
            progress = elapsed_epoch / max_time
            
            # Calculate target population size linearly decreasing with time
            n_target = int(n_min + (n_init - n_min) * (1.0 - progress))
            n_target = max(n_min, n_target)
            
            # Reduce population if needed
            if current_pop_size > n_target:
                n_remove = current_pop_size - n_target
                # Sort indices by fitness (worst at the end)
                sorted_indices = np.argsort(fitness)
                # Keep the top n_target
                keep = sorted_indices[:n_target]
                
                pop = pop[keep]
                fitness = fitness[keep]
                current_pop_size = n_target
                
                # Prune archive to match population size (L-SHADE rule)
                if len(archive) > current_pop_size:
                    # Randomly reduce archive
                    import random
                    random.shuffle(archive)
                    archive = archive[:current_pop_size]

            # 2. Parameter Generation (SHADE)
            r_idx = np.random.randint(0, H, current_pop_size)
            mu_cr = mem_cr[r_idx]
            mu_f = mem_f[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            
            # F ~ Cauchy(mu_f, 0.1)
            # Generated via tan: loc + scale * tan(pi * (rand - 0.5))
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(current_pop_size) - 0.5))
            
            # Handle F constraints
            # If F <= 0, regenerate until positive
            neg_mask = f <= 0
            while np.any(neg_mask):
                f[neg_mask] = mu_f[neg_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(neg_mask)) - 0.5))
                neg_mask = f <= 0
            # If F > 1, clamp to 1
            f[f > 1.0] = 1.0
            
            # 3. Mutation: current-to-pbest/1
            # Sort population to find p-best
            sorted_indices = np.argsort(fitness)
            # p-best ratio usually 0.11 or similar
            num_pbest = max(2, int(0.11 * current_pop_size))
            pbest_indices = sorted_indices[:num_pbest]
            
            # Select pbest for each individual
            r_pbest = np.random.choice(pbest_indices, current_pop_size)
            x_pbest = pop[r_pbest]
            
            # Select r1 (distinct from i)
            idxs = np.arange(current_pop_size)
            r1_raw = np.random.randint(0, current_pop_size - 1, current_pop_size)
            # Map raw index [0, N-2] to [0, N-1] skipping self
            r1 = np.where(r1_raw >= idxs, r1_raw + 1, r1_raw)
            x_r1 = pop[r1]
            
            # Select r2 (from Population U Archive)
            if len(archive) > 0:
                arr_archive = np.array(archive)
                union_pop = np.vstack((pop, arr_archive))
            else:
                union_pop = pop
            
            # Pick r2 random from union
            r2 = np.random.randint(0, len(union_pop), current_pop_size)
            x_r2 = union_pop[r2]
            
            # Compute Mutant: v = x + F*(pbest - x) + F*(r1 - r2)
            f_col = f[:, np.newaxis]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # 4. Crossover (Binomial)
            # Ensure at least one parameter comes from mutant
            j_rand = np.random.randint(0, dim, current_pop_size)
            rand_vals = np.random.rand(current_pop_size, dim)
            cross_mask = rand_vals < cr[:, np.newaxis]
            cross_mask[np.arange(current_pop_size), j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop)
            
            # 5. Bound Handling (Reflection)
            # Lower bounds
            mask_l = trial < min_b
            if np.any(mask_l):
                # Reflect: 2*min - x
                trial[mask_l] = 2 * min_b[np.where(mask_l)[1]] - trial[mask_l]
                trial = np.maximum(trial, min_b) # Safety clamp
            
            # Upper bounds
            mask_u = trial > max_b
            if np.any(mask_u):
                # Reflect: 2*max - x
                trial[mask_u] = 2 * max_b[np.where(mask_u)[1]] - trial[mask_u]
                trial = np.minimum(trial, max_b) # Safety clamp

            # 6. Evaluation & Selection
            succ_f = []
            succ_cr = []
            succ_diff = []
            
            for i in range(current_pop_size):
                if (datetime.now() - start_time) >= limit:
                    return best_val
                    
                val_trial = func(trial[i])
                
                # Greedy selection
                if val_trial <= fitness[i]:
                    # Update Archive with old solution
                    if val_trial < fitness[i]:
                        archive.append(pop[i].copy())
                        
                        # Record success for SHADE adaptation
                        succ_f.append(f[i])
                        succ_cr.append(cr[i])
                        succ_diff.append(fitness[i] - val_trial)
                    
                    # Replace
                    pop[i] = trial[i]
                    fitness[i] = val_trial
                    
                    if val_trial < best_val:
                        best_val = val_trial
                        best_sol = trial[i].copy()
                        
            # 7. Update SHADE Memory
            if len(succ_diff) > 0:
                s_f = np.array(succ_f)
                s_cr = np.array(succ_cr)
                s_diff = np.array(succ_diff)
                
                # Weights based on improvement magnitude
                total_diff = np.sum(s_diff)
                weights = s_diff / total_diff
                
                # Weighted Mean for CR
                mean_cr = np.sum(weights * s_cr)
                mem_cr[k_mem] = 0.5 * mem_cr[k_mem] + 0.5 * mean_cr
                
                # Weighted Lehmer Mean for F
                # sum(w * f^2) / sum(w * f)
                mean_f_num = np.sum(weights * (s_f**2))
                mean_f_den = np.sum(weights * s_f)
                if mean_f_den > 0:
                    mean_f = mean_f_num / mean_f_den
                    mem_f[k_mem] = 0.5 * mem_f[k_mem] + 0.5 * mean_f
                
                k_mem = (k_mem + 1) % H
                
            # 8. Convergence Check
            # If standard deviation of fitness is negligible, restart
            if np.std(fitness) < 1e-8:
                break # Break inner loop, trigger restart with remaining time

    return best_val
