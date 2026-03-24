#To improve upon the best result (9.10), I propose implementing the **L-SHADE (Linear Population Size Reduction Success-History based Adaptive Differential Evolution)** algorithm, but with a critical **Restart Mechanism** and **Midpoint Bound Handling**.
#
#**Why this combination?**
#1.  **L-SHADE Backbone:** L-SHADE is a state-of-the-art evolutionary algorithm that linearly reduces population size to shift from exploration to exploitation. This is generally superior to static parameter handling.
#2.  **Midpoint Bound Handling:** The previous best result used clipping. For many problems, optima lie near boundaries or require bouncing off "walls" to be found. The midpoint strategy (`(current + bound) / 2`) preserves direction better than simple clipping.
#3.  **Stagnation Restart:** Standard L-SHADE can occasionally converge prematurely to a local optimum (like 9.10). By monitoring population diversity (variance), we can detect stagnation, keep the best solution, and re-scatter the rest of the population to find better basins of attraction within the remaining time.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    L-SHADE with Restart Mechanism and Midpoint Bound Handling.
    Optimizes function 'func' within 'max_time'.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper Functions ---
    def get_honored_bounds(trial, old, low, high):
        """
        Midpoint target bound handling. 
        If a value is out of bounds, set it to the average of the old value and the bound.
        This helps avoiding sticking to the edges (clipping) while keeping valid search direction.
        """
        # Lower bound
        mask_low = trial < low
        if np.any(mask_low):
            trial[mask_low] = (old[mask_low] + low[mask_low]) / 2.0
            
        # Upper bound
        mask_high = trial > high
        if np.any(mask_high):
            trial[mask_high] = (old[mask_high] + high[mask_high]) / 2.0
            
        return trial

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Population Size Parameters (L-SHADE)
    # Start with a sufficiently large population for exploration
    initial_pop_size = int(round(18 * dim))
    initial_pop_size = np.clip(initial_pop_size, 30, 200) # Safety clamps
    min_pop_size = 4
    
    pop_size = initial_pop_size
    
    # Memory Parameters
    H = 5
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0
    
    # Archive
    archive = []
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * (max_b - min_b)
    fitness = np.full(pop_size, float('inf'))
    
    best_idx = -1
    best_fit = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_fit
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_idx = i
            
    if best_idx == -1: return float('inf')

    # --- Main Loop ---
    while True:
        elapsed = datetime.now() - start_time
        if elapsed >= time_limit:
            break
            
        progress = elapsed.total_seconds() / max_time
        
        # 1. Stagnation Check & Restart
        # If diversity is lost early, restart population around the best
        # Only do this if we haven't reduced the population too much yet (early/mid game)
        if pop_size > 10 and progress < 0.8:
            fit_std = np.std(fitness)
            if fit_std < 1e-9: # Converged
                # Keep best, scramble others
                for i in range(pop_size):
                    if i == best_idx: continue
                    pop[i] = min_b + np.random.rand(dim) * (max_b - min_b)
                    # We defer evaluation to the main loop to keep logic simple
                    fitness[i] = float('inf') # Mark for re-eval
                
                # Reset Archive and Memory to adapt to new landscape
                archive = []
                mem_cr.fill(0.5)
                mem_f.fill(0.5)
        
        # 2. Linear Population Size Reduction (LPSR)
        # Calculate target size based on time progress
        plan_pop_size = int(round(((min_pop_size - initial_pop_size) * progress) + initial_pop_size))
        plan_pop_size = max(min_pop_size, plan_pop_size)
        
        if pop_size > plan_pop_size:
            # Sort by fitness and reduce
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices[:plan_pop_size]]
            fitness = fitness[sort_indices[:plan_pop_size]]
            pop_size = plan_pop_size
            
            # Recalculate best index after sort
            best_idx = np.argmin(fitness)
            best_fit = fitness[best_idx]
            
            # Reduce archive size to match current pop capacity (roughly)
            target_arc = int(pop_size * 2.0)
            if len(archive) > target_arc:
                del_count = len(archive) - target_arc
                # Randomly delete
                for _ in range(del_count):
                    archive.pop(np.random.randint(0, len(archive)))

        # 3. Parameter Generation
        # Vectorized generation for efficiency
        r_indices = np.random.randint(0, H, pop_size)
        m_cr = mem_cr[r_indices]
        m_f = mem_f[r_indices]
        
        # CR: Normal(m_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # F: Cauchy(m_f, 0.1), clipped [0, 1]
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Repair F
        # If F > 1 -> 1. If F <= 0 -> Regenerate
        while True:
            mask_neg = f <= 0
            if not np.any(mask_neg): break
            f[mask_neg] = m_f[mask_neg] + 0.1 * np.random.standard_cauchy(np.sum(mask_neg))
        f = np.clip(f, 0, 1)
        
        # 4. Mutation: current-to-pbest/1
        # Dynamic p: linearly decreases from 0.2 to 0.05
        # This encourages exploration early and exploitation late
        p_val = 0.2 - (0.15 * progress)
        p_val = max(2.0/pop_size, p_val)
        
        # Identify p-best for each individual
        sorted_indices = np.argsort(fitness)
        pbest_indices = []
        top_count = int(max(2, round(p_val * pop_size)))
        
        # Vectorized p-best selection not trivial, doing loop for indices
        for i in range(pop_size):
            p_ind = np.random.randint(0, top_count)
            pbest_indices.append(sorted_indices[p_ind])
        pbest_indices = np.array(pbest_indices)
        x_pbest = pop[pbest_indices]
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        mask_same = (r1_indices == np.arange(pop_size))
        r1_indices[mask_same] = (r1_indices[mask_same] + 1) % pop_size
        x_r1 = pop[r1_indices]
        
        # Select r2 (distinct from i, r1) from Union(Pop, Archive)
        if len(archive) > 0:
            arr_arc = np.array(archive)
            union_pop = np.vstack((pop, arr_arc))
        else:
            union_pop = pop
            
        r2_indices = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Calculate mutant vectors
        f_col = f.reshape(-1, 1)
        mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 5. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross = np.random.rand(pop_size, dim) < cr.reshape(-1, 1)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, pop)
        
        # 6. Bound Handling (Midpoint)
        for i in range(pop_size):
            trial[i] = get_honored_bounds(trial[i], pop[i], min_b, max_b)
            
        # 7. Evaluation & Selection
        success_cr = []
        success_f = []
        diff_fitness = []
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fit
            
            # If restart happened, we might have inf fitness in pop, just evaluate trial
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Improvement
                if f_trial < fitness[i]:
                    archive.append(pop[i].copy())
                    success_cr.append(cr[i])
                    success_f.append(f[i])
                    # If old fitness was inf (restart), diff is just 0 or handled gracefully
                    diff = fitness[i] - f_trial
                    if not np.isinf(diff):
                        diff_fitness.append(diff)
                    else:
                        diff_fitness.append(1.0) # Arbitrary positive weight
                
                new_pop[i] = trial[i]
                new_fitness[i] = f_trial
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_idx = i
                    
        pop = new_pop
        fitness = new_fitness
        
        # Maintain archive size
        limit_arc = int(pop_size * 2.0)
        while len(archive) > limit_arc:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 8. Memory Update (Weighted Lehmer Mean)
        if len(diff_fitness) > 0:
            diff_np = np.array(diff_fitness)
            scr_np = np.array(success_cr)
            sf_np = np.array(success_f)
            
            total_diff = np.sum(diff_np)
            if total_diff > 0:
                weights = diff_np / total_diff
                
                # Mean CR
                mean_cr = np.sum(weights * scr_np)
                
                # Mean F (Lehmer)
                num = np.sum(weights * (sf_np**2))
                den = np.sum(weights * sf_np)
                mean_f = num / den if den > 0 else 0.5
                
                mem_cr[k_mem] = mean_cr
                mem_f[k_mem] = np.clip(mean_f, 0, 1)
                
                k_mem = (k_mem + 1) % H

    return best_fit
