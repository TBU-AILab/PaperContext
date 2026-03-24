#The following algorithm implements a simplified **L-SHADE (Linear Success-History Adaptive Differential Evolution)** with an external archive and a terminal **Gaussian Polishing** phase.
#
#### Rationale for Improvement
#The previous best result (35.85) was achieved by introducing adaptivity (jDE) and local polishing. To further minimize the output:
#1.  **History-Based Adaptation (L-SHADE)**: Instead of the random-walk parameter adaptation of jDE, this algorithm uses a memory ($H$) to store successful $F$ and $CR$ values, guiding the generation of new parameters based on what worked in the past.
#2.  **Linear Population Size Reduction (LPSR)**: The population size starts large to maximize exploration and linearly decreases over the `max_time` duration. This forces the algorithm to shift from exploration to fine-grained exploitation automatically.
#3.  **External Archive**: An archive stores good solutions replaced during selection. This preserves diversity and mitigates premature convergence by allowing the mutation strategy to pull difference vectors from a wider pool ($P \cup A$).
#4.  **Gaussian Polishing**: As in the previous version, if the population converges or time is nearly up, the algorithm switches to a focused hill-climbing strategy on the global best to extract maximum precision.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Success-History Adaptive DE)
    with Linear Population Size Reduction, External Archive, and Gaussian Polishing.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population Sizing
    # Start with a larger population for exploration, reduce linearly to min_pop_size.
    # Cap initial size to ensure performance within time limits.
    init_pop_size = int(max(30, min(18 * dim, 200))) 
    min_pop_size = 4
    
    # Archive parameters
    archive_factor = 2.6
    
    # Adaptation Memory
    mem_size = 5
    M_CR = np.full(mem_size, 0.5)
    M_F = np.full(mem_size, 0.5)
    k_mem = 0
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop_size = init_pop_size
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.zeros(pop_size)
    
    best_fit = float('inf')
    best_pos = None
    
    # Evaluate initial population
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_fit if best_pos is not None else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fit:
            best_fit = val
            best_pos = population[i].copy()
            
    # Sort population by fitness (required for rank-based mutation)
    sort_idx = np.argsort(fitness)
    population = population[sort_idx]
    fitness = fitness[sort_idx]
    
    # Initialize Archive (pre-allocated numpy array for speed)
    max_arc_size = int(init_pop_size * archive_factor)
    archive = np.zeros((max_arc_size, dim))
    arc_cnt = 0
    
    # State flags
    polishing_done = False
    
    # --- Main Optimization Loop ---
    while True:
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()
        
        # Check Global Time Limit
        if elapsed >= max_time:
            return best_fit
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate progress ratio (0.0 to 1.0)
        progress = elapsed / max_time
        
        # Calculate target population size based on remaining time
        target_size = int(round(min_pop_size + (init_pop_size - min_pop_size) * (1.0 - progress)))
        target_size = max(min_pop_size, target_size)
        
        # Shrink Population if needed
        if target_size < pop_size:
            # Keep best 'target_size' individuals (population is already sorted)
            population = population[:target_size]
            fitness = fitness[:target_size]
            pop_size = target_size
            
            # Shrink Archive capacity proportionally
            current_arc_cap = int(pop_size * archive_factor)
            if arc_cnt > current_arc_cap:
                # Truncate archive (effectively removing random old members as archive is unsorted)
                arc_cnt = current_arc_cap
        
        # 2. Convergence Check & Polishing Trigger
        # If population collapsed (low variance) OR we are near the end (95% time),
        # switch to single-point local search to refine the best solution.
        if not polishing_done:
            is_converged = (np.std(fitness) < 1e-8)
            near_end = (progress > 0.95)
            
            if is_converged or near_end:
                rem_time = max_time - elapsed
                if rem_time > 0.05: # Only if meaningful time remains
                    # Gaussian Walk / Hill Climber
                    p_curr = best_pos.copy()
                    p_fit = best_fit
                    sigma = np.max(diff_b) * 0.01 # Initial step size
                    
                    p_start = datetime.now()
                    while (datetime.now() - p_start).total_seconds() < rem_time:
                        if (datetime.now() - start_time).total_seconds() >= max_time:
                            return best_fit
                        
                        # Sample candidate
                        cand = p_curr + np.random.normal(0, 1, dim) * sigma
                        cand = np.clip(cand, min_b, max_b)
                        val = func(cand)
                        
                        if val < p_fit:
                            p_fit = val
                            p_curr = cand
                            sigma *= 1.1 # Increase step size on success
                            
                            if val < best_fit:
                                best_fit = val
                                best_pos = cand
                        else:
                            sigma *= 0.5 # Decrease step size on failure
                        
                        # Stop if step becomes negligible
                        if sigma < 1e-15:
                            break
                            
                polishing_done = True
                if near_end:
                    return best_fit

        # 3. Adaptive Parameter Generation
        # Pick random memory slot for each individual
        r_idx = np.random.randint(0, mem_size, pop_size)
        m_cr = M_CR[r_idx]
        m_f = M_F[r_idx]
        
        # Generate CR: Normal(M_CR, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F: Cauchy(M_F, 0.1)
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Handle F constraints (F > 0)
        # Resample if F <= 0
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad):
                break
            count = np.sum(mask_bad)
            f[mask_bad] = m_f[mask_bad] + 0.1 * np.random.standard_cauchy(count)
            
        f = np.clip(f, 0, 1) # Cap at 1.0
        
        # 4. Mutation Strategy: current-to-pbest/1
        # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        
        # Select 'p' for p-best (random between 2/pop and 20%)
        p_min = 2.0 / pop_size
        if p_min < 0.2:
            p_val = np.random.uniform(p_min, 0.2)
        else:
            p_val = p_min
            
        top_p_cnt = int(max(1, pop_size * p_val))
        
        # Indices for p-best
        pbest_indices = np.random.randint(0, top_p_cnt, pop_size)
        x_pbest = population[pbest_indices]
        
        # Indices for r1 (random from population)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        # Avoid self-selection r1 != i (approximate fix via rotation)
        same_mask = (r1_indices == np.arange(pop_size))
        r1_indices[same_mask] = (r1_indices[same_mask] + 1) % pop_size
        x_r1 = population[r1_indices]
        
        # Indices for r2 (random from Population U Archive)
        # Construct Union Logic
        union_size = pop_size + arc_cnt
        r2_indices = np.random.randint(0, union_size, pop_size)
        
        # Build x_r2 array
        x_r2 = np.zeros((pop_size, dim))
        
        # Mask for those picking from population
        in_pop_mask = r2_indices < pop_size
        x_r2[in_pop_mask] = population[r2_indices[in_pop_mask]]
        
        # Mask for those picking from archive
        in_arc_mask = ~in_pop_mask
        if np.any(in_arc_mask):
            arc_indices = r2_indices[in_arc_mask] - pop_size
            x_r2[in_arc_mask] = archive[arc_indices]
            
        # Calculate Mutant Vector
        f_v = f[:, np.newaxis]
        mutant = population + f_v * (x_pbest - population) + f_v * (x_r1 - x_r2)
        mutant = np.clip(mutant, min_b, max_b)
        
        # 5. Crossover (Binomial)
        rand_cr = np.random.rand(pop_size, dim)
        cross_mask = rand_cr < cr[:, np.newaxis]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)
        
        # 6. Selection & Memory Update
        succ_scr = [] # Successful CR
        succ_sf = []  # Successful F
        diff_vals = [] # Fitness improvement
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fit
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Solution improved
                imp = fitness[i] - f_trial
                
                # Add parent to archive before replacing
                if arc_cnt < max_arc_size:
                    archive[arc_cnt] = population[i].copy()
                    arc_cnt += 1
                else:
                    # Archive full: replace random member
                    rep_idx = np.random.randint(0, max_arc_size)
                    archive[rep_idx] = population[i].copy()
                
                # Store success data
                succ_scr.append(cr[i])
                succ_sf.append(f[i])
                diff_vals.append(imp)
                
                # Update Population
                population[i] = trial_pop[i]
                fitness[i] = f_trial
                
                if f_trial < best_fit:
                    best_fit = f_trial
                    best_pos = trial_pop[i].copy()
        
        # 7. Update History Memory (Weighted Lehmer Mean)
        if len(succ_scr) > 0:
            succ_scr = np.array(succ_scr)
            succ_sf = np.array(succ_sf)
            diff_vals = np.array(diff_vals)
            
            total_diff = np.sum(diff_vals)
            # Weights based on improvement magnitude
            if total_diff == 0:
                weights = np.ones(len(diff_vals)) / len(diff_vals)
            else:
                weights = diff_vals / total_diff
                
            # Update M_CR (Weighted Mean)
            m_cr_new = np.sum(weights * succ_scr)
            if np.isnan(m_cr_new): m_cr_new = 0.5
            
            # Update M_F (Weighted Lehmer Mean)
            num = np.sum(weights * (succ_sf ** 2))
            den = np.sum(weights * succ_sf)
            if den == 0:
                m_f_new = 0.5
            else:
                m_f_new = num / den
            if np.isnan(m_f_new): m_f_new = 0.5
            
            # Update Memory slot
            M_CR[k_mem] = np.clip(m_cr_new, 0, 1)
            M_F[k_mem] = np.clip(m_f_new, 0, 1)
            k_mem = (k_mem + 1) % mem_size
            
        # 8. Sort Population (Crucial for next generation's p-best selection)
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]

    return best_fit
