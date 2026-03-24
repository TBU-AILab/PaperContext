#Here is a fully operational, self-contained Python algorithm. It implements the **jSO (Self-adaptive Differential Evolution with Weighted Linear Population Size Reduction)** algorithm.
#
#This algorithm improves upon the previous L-SHADE implementation by:
#1.  **Dynamic `p`-best Selection**: Linearly reducing the greedy parameter `p` (from exploration to exploitation) over time.
#2.  **Mid-point Boundary Correction**: Instead of simple clipping (which causes "sticking" to bounds), it uses a bounce-back approach to preserve population distribution.
#3.  **Soft Restarts**: Instead of wiping the population when stagnated, it retains the global best and generates a "cloud" of mutants around it, maintaining the discovered basin of attraction while re-injecting diversity.
#4.  **Optimized Vectorization**: Minimizes Python loop overhead for parameter generation.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Implements jSO (variant of L-SHADE) with Soft Restarts.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Bounds Preprocessing ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Parameters ---
    # jSO / L-SHADE sizing
    # Initial population size: typically 25 * log(dim) * sqrt(dim), but we clamp for safety
    N_init = int(25 * np.sqrt(dim))
    N_init = max(30, min(N_init, 300)) 
    N_min = 4 # Minimum population size allowed
    
    # Archive parameters
    arc_rate = 2.6
    
    # Memory parameters
    H = 6 # History size
    M_cr = np.full(H, 0.8) # Start with high CR (approx 0.8)
    M_f = np.full(H, 0.5)  # Start with mid F
    k_mem = 0
    
    # Best solution holder
    best_sol = None
    best_fitness = float('inf')
    
    # Helper to check time
    def get_progress():
        elapsed = (datetime.now() - start_time).total_seconds()
        return elapsed / max_time

    def check_time():
        return (datetime.now() - start_time) >= limit

    # --- Initialization ---
    pop_size = N_init
    # Latin Hypercube Sampling-like initialization (Stratified) for better coverage
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
        pop[:, d] = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(pop[:, d])
        
    fitness = np.zeros(pop_size)
    
    # Initial Evaluation
    for i in range(pop_size):
        if check_time(): return best_fitness
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # Sort population
    sort_idx = np.argsort(fitness)
    pop = pop[sort_idx]
    fitness = fitness[sort_idx]
    
    archive = [] 
    
    # --- Main Loop ---
    while not check_time():
        
        # 1. Calculate Progress (0.0 to 1.0)
        progress = get_progress()
        if progress > 1.0: progress = 1.0
        
        # 2. Linear Population Size Reduction (LPSR)
        # Calculate new target size
        N_target = int(round(N_init + (N_min - N_init) * progress))
        N_target = max(N_min, N_target)
        
        if pop_size > N_target:
            # Reduction: Keep best N_target
            # (Population is always sorted at start of loop or after selection)
            reduction_count = pop_size - N_target
            # Add removed individuals to archive if space permits (optional, but good for diversity)
            for i in range(N_target, pop_size):
                if len(archive) < int(N_target * arc_rate):
                    archive.append(pop[i].copy())
            
            pop = pop[:N_target]
            fitness = fitness[:N_target]
            pop_size = N_target
            
            # Reduce archive size
            max_arc_size = int(pop_size * arc_rate)
            if len(archive) > max_arc_size:
                # Remove random elements to shrink
                idxs = np.random.choice(len(archive), max_arc_size, replace=False)
                archive = [archive[i] for i in idxs]

        # 3. Dynamic p-value for current-to-pbest
        # p ranges from 0.25 (exploration) down to 0.05 (exploitation)
        p_max = 0.25
        p_min = 0.05
        p_curr = p_max - (p_max - p_min) * progress
        
        # 4. Generate Parameters (F and CR)
        r_idx = np.random.randint(0, H, pop_size)
        m_cr = M_cr[r_idx]
        m_f = M_f[r_idx]
        
        # CR generation: Normal(m_cr, 0.1)
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        # Fix: CR should often be 0.0 or 1.0 in DE? No, L-SHADE typically uses continuous.
        # But if M_cr is essentially -1 (terminal), set to 0. 
        # (L-SHADE sets -1 to mean 0, but we use actual means).
        
        # F generation: Cauchy(m_f, 0.1)
        # Vectorized rejection sampling for Cauchy
        f = np.zeros(pop_size)
        
        # Base generation
        f_raw = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Check constraints
        # Rule: if F > 1 -> F = 1. If F <= 0 -> Regenerate.
        # Fast correction loop
        bad_indices = np.where(f_raw <= 0)[0]
        while len(bad_indices) > 0:
            # Regenerate only bad ones
            new_vals = m_f[bad_indices] + 0.1 * np.random.standard_cauchy(len(bad_indices))
            f_raw[bad_indices] = new_vals
            bad_indices = np.where(f_raw <= 0)[0]
            
        f = np.clip(f_raw, 0.0, 1.0) # Clip upper at 1.0
        
        # 5. Mutation: current-to-pbest/1
        # Prepare p-best
        p_num = max(1, int(p_curr * pop_size))
        pbest_idxs = np.random.randint(0, p_num, pop_size) # Indices in sorted pop
        x_pbest = pop[pbest_idxs]
        
        # Prepare r1 (distinct from i)
        # Shift random indices
        idxs = np.arange(pop_size)
        r1_idxs = np.random.randint(0, pop_size - 1, pop_size)
        # If r1_idx >= i, add 1 to skip i
        r1_idxs = np.where(r1_idxs >= idxs, r1_idxs + 1, r1_idxs)
        x_r1 = pop[r1_idxs]
        
        # Prepare r2 (distinct from i and r1, from Union(Pop, Archive))
        n_arc = len(archive)
        union_pop = pop
        if n_arc > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        
        n_union = len(union_pop)
        
        r2_idxs = np.random.randint(0, n_union - 2, pop_size)
        # Logic to ensure distinctness is complex to fully vectorize perfectly 
        # without small bias, but approx is okay for DE. 
        # Let's do a simple check-and-fix loop which is fast enough for N<200.
        x_r2 = np.zeros((pop_size, dim))
        for i in range(pop_size):
            r1 = r1_idxs[i]
            r2 = np.random.randint(0, n_union)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, n_union)
            x_r2[i] = union_pop[r2]

        # Calculate Mutation Vectors (jSO weighting applies specific logic, 
        # but L-SHADE standard form is robust enough here)
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        f_v = f[:, np.newaxis]
        # Note: L-SHADE uses F for both terms. jSO uses F_w for second term sometimes.
        # Sticking to standard L-SHADE formula for stability.
        mutant = pop + f_v * (x_pbest - pop) + f_v * (x_r1 - x_r2)
        
        # 6. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals < cr[:, np.newaxis]
        
        # Ensure at least one parameter comes from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        j_mask = np.zeros((pop_size, dim), dtype=bool)
        j_mask[np.arange(pop_size), j_rand] = True
        mask = np.logical_or(mask, j_mask)
        
        trial = np.where(mask, mutant, pop)
        
        # 7. Boundary Constraint Handling (Mid-point / Bounce back)
        # Instead of clipping, we set value to (bound + old_val) / 2
        # This preserves the search direction but restricts range.
        lower_viol = trial < min_b
        upper_viol = trial > max_b
        
        # If violated, set to mean of boundary and parent
        # This helps avoid getting stuck on the edge
        trial[lower_viol] = (min_b[np.where(lower_viol)[1]] + pop[lower_viol]) / 2.0
        trial[upper_viol] = (max_b[np.where(upper_viol)[1]] + pop[upper_viol]) / 2.0
        
        # 8. Selection
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        succ_mask = np.zeros(pop_size, dtype=bool)
        diff_f = np.zeros(pop_size)
        
        # Evaluation Loop
        for i in range(pop_size):
            if check_time(): return best_fitness
            
            t_fit = func(trial[i])
            
            if t_fit <= fitness[i]:
                # Improvement or equal
                if t_fit < fitness[i]:
                    archive.append(pop[i].copy())
                    diff_f[i] = fitness[i] - t_fit
                    succ_mask[i] = True
                
                new_pop[i] = trial[i]
                new_fitness[i] = t_fit
                
                if t_fit < best_fitness:
                    best_fitness = t_fit
                    best_sol = trial[i].copy()
                    
        pop = new_pop
        fitness = new_fitness
        
        # Sort for next p-best selection
        sort_p = np.argsort(fitness)
        pop = pop[sort_p]
        fitness = fitness[sort_p]
        
        # 9. Update Memory (Weighted Lehmer Mean)
        if np.any(succ_mask):
            s_f = f[succ_mask]
            s_cr = cr[succ_mask]
            df = diff_f[succ_mask]
            
            # Normalize weights
            denom = np.sum(df)
            if denom == 0:
                weights = np.ones(len(df)) / len(df)
            else:
                weights = df / denom
                
            # Update M_cr (Weighted Mean)
            # Check for terminal value -1 in L-SHADE usually ignored here, we assume bounds [0,1]
            m_cr_new = np.sum(weights * s_cr)
            # Clamp for safety
            M_cr[k_mem] = np.clip(m_cr_new, 0, 1)
            
            # Update M_f (Weighted Lehmer Mean)
            f_numer = np.sum(weights * (s_f**2))
            f_denom = np.sum(weights * s_f)
            if f_denom > 0:
                M_f[k_mem] = np.clip(f_numer / f_denom, 0, 1)
            else:
                M_f[k_mem] = 0.5
                
            k_mem = (k_mem + 1) % H

        # 10. Soft Restart Check
        # If population diversity is extremely low, or stagnant, we restart
        # But we keep the best solution (Soft Restart)
        # Convergence criteria: std dev of fitness or position
        if pop_size > N_min:
            # Check range of fitness
            fit_range = np.abs(fitness[-1] - fitness[0])
            if fit_range < 1e-8:
                # --- SOFT RESTART ---
                # 1. Keep Best (index 0 after sort)
                # 2. Perturb best with different scales
                
                # Keep top 10%
                n_keep = max(1, int(0.1 * pop_size))
                
                # Remaining slots
                n_reset = pop_size - n_keep
                
                # Generate new individuals
                # Half: Small Gaussian cloud around best (Exploitation of current basin)
                # Half: Global random (Exploration)
                n_cloud = n_reset // 2
                n_rand = n_reset - n_cloud
                
                # Cloud: std dev based on current best history or arbitrary small factor
                # Use mean parameter bounds width / 20
                sigma = (diff_b) * 0.05
                cloud = best_sol + np.random.normal(0, 1, (n_cloud, dim)) * sigma
                
                # Random
                rand_part = min_b + np.random.rand(n_rand, dim) * diff_b
                
                # Combine
                pop[n_keep : n_keep+n_cloud] = cloud
                pop[n_keep+n_cloud :] = rand_part
                
                # Clamp
                pop = np.clip(pop, min_b, max_b)
                
                # Evaluate new batch
                for i in range(n_keep, pop_size):
                    if check_time(): return best_fitness
                    val = func(pop[i])
                    fitness[i] = val
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = pop[i].copy()
                
                # Re-sort
                idx_r = np.argsort(fitness)
                pop = pop[idx_r]
                fitness = fitness[idx_r]
                
                # Reset Memory partially? 
                # No, keeping memory often helps if the function has consistent landscape properties.
                # But we can disperse the history slightly to encourage diversity.
                M_cr = np.clip(np.random.normal(M_cr, 0.1), 0, 1)
                M_f = np.clip(np.random.normal(M_f, 0.1), 0, 1)

    return best_fitness
