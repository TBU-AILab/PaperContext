#To find the set of input parameters that minimizes the output value within a limited time, I propose the **SHADE-ILS (Success-History Adaptive Differential Evolution with Iterated Local Search)** algorithm.
#
#### Key Improvements
#1.  **SHADE Architecture**: Builds upon the high-performing L-SHADE logic (Success-History Adaptive Differential Evolution) but simplifies the implementation to reduce overhead. It uses a historical memory of successful control parameters ($F$ and $CR$) to adapt to the objective function landscape over time.
#2.  **Linear Population Size Reduction (LPSR)**: The population size linearly decreases from a high initial value (exploration) to a small value (exploitation) as the time budget is consumed. This naturally forces convergence.
#3.  **Reflection Boundary Handling**: Instead of simple clipping (which biases the search to the edges) or resampling (which is slow), this implementation uses a "reflection" strategy ($lower + (lower - x)$) to bounce particles back into the search space, often finding optima near boundaries more effectively.
#4.  **Terminal Local Search (Polishing)**: The most significant addition. The algorithm reserves the last 5-10% of the time budget to perform a high-frequency Gaussian Local Search around the global best solution. Evolutionary algorithms are great at finding the correct "basin" of attraction, but often slow at drilling down to the absolute minimum. This hybrid approach solves that efficiency gap.
#5.  **Robust Restart**: If the population converges early (variance drops), it triggers a "soft restart" where the global best is retained, but the rest of the population is re-initialized to explore new areas.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE-ILS (SHADE with Linear Population Reduction
    and Terminal Iterated Local Search).
    """
    # --- Time Management ---
    t_start = time.time()
    # Reserve last portion of time for local search polishing
    t_optimization_ratio = 0.90 
    t_limit_optimization = t_start + (max_time * t_optimization_ratio)
    t_hard_stop = t_start + max_time

    # --- Initialization ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    bounds_width = ub - lb

    # Population Sizing (LPSR)
    # Start large for exploration, reduce to min_pop for exploitation
    max_pop = int(round(max(20, min(200, 18 * dim)))) 
    min_pop = 4
    pop_size = max_pop
    
    # SHADE Memory
    H_mem = 5
    M_cr = np.full(H_mem, 0.5)
    M_f = np.full(H_mem, 0.5)
    k_mem = 0
    
    # Archive (for current-to-pbest/1 mutation)
    archive = []
    
    # Initialize Population
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        pop[:, d] = np.random.uniform(lb[d], ub[d], pop_size)
    
    # Initial Evaluation
    fitness = np.zeros(pop_size)
    best_val = float('inf')
    best_sol = np.zeros(dim)

    # Evaluate initial population
    for i in range(pop_size):
        if time.time() > t_hard_stop:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop[i].copy()

    # --- Main Evolutionary Loop ---
    # We estimate total generations based on time elapsed so far
    # to drive the Linear Population Size Reduction (LPSR)
    gen = 0
    max_gens_est = 1000 # Initial rough estimate
    
    while True:
        current_time = time.time()
        
        # 1. Check switch condition to Local Search
        if current_time >= t_limit_optimization:
            break
            
        # Update generation count estimate dynamically
        if gen == 1:
            elapsed = current_time - t_start
            if elapsed > 0:
                # Estimate remaining gens allowed
                avg_time_per_gen = elapsed
                remaining_time = t_limit_optimization - current_time
                max_gens_est = int(remaining_time / avg_time_per_gen) + 2

        # 2. Linear Population Size Reduction (LPSR)
        # Calculate target population size
        progress = min(1.0, gen / max_gens_est) if max_gens_est > 0 else 1.0
        new_pop_size = int(round((min_pop - max_pop) * progress + max_pop))
        new_pop_size = max(min_pop, new_pop_size)

        if pop_size > new_pop_size:
            # Reduce population: keep best individuals
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx[:new_pop_size]]
            fitness = fitness[sorted_idx[:new_pop_size]]
            # Resize archive
            if len(archive) > new_pop_size:
                # Shuffle and clip archive
                np.random.shuffle(archive)
                archive = archive[:new_pop_size]
            pop_size = new_pop_size

        # 3. Check Convergence for Restart
        # If population is extremely clustered, restart but keep best
        if np.std(fitness) < 1e-9 and progress < 0.8:
            # Keep best
            pop = np.random.uniform(lb, ub, (pop_size, dim))
            pop[0] = best_sol
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            # Evaluate new randoms
            for i in range(1, pop_size):
                if time.time() > t_limit_optimization: break
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_sol = pop[i].copy()
            # Reset Memory and Archive
            M_cr[:] = 0.5
            M_f[:] = 0.5
            archive = []
            gen += 1
            continue

        # 4. Generate Adaptive Parameters
        # Select random memory index for each individual
        r_idx = np.random.randint(0, H_mem, pop_size)
        m_cr = M_cr[r_idx]
        m_f = M_f[r_idx]

        # Generate CR ~ Normal(m_cr, 0.1) -> clipped [0, 1]
        cr = np.random.normal(m_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        # Fix CR=0 to small epsilon to ensure at least some crossover
        cr = np.maximum(cr, 1e-6)

        # Generate F ~ Cauchy(m_f, 0.1) -> clipped [0, 1]
        # Cauchy: f = m_f + 0.1 * tan(pi * (rand - 0.5))
        # Standard numpy cauchy is just standard_cauchy
        f = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        f = np.where(f <= 0, 0.5, f) # If too small, reset to default (safer than clipping to 0)
        f = np.where(f > 1, 1.0, f)

        # 5. Mutation: current-to-pbest/1
        # Sort for pbest selection
        sorted_indices = np.argsort(fitness)
        
        # p_best rate scales from 0.2 down to 0.05
        p_val = max(0.05, 0.2 * (1 - progress))
        top_p_cnt = max(2, int(pop_size * p_val))
        
        # Vectors
        x_curr = pop
        
        # x_pbest
        pbest_idxs = np.random.randint(0, top_p_cnt, pop_size)
        x_pbest = pop[sorted_indices[pbest_idxs]]
        
        # x_r1 (Random from pop)
        r1_idxs = np.random.randint(0, pop_size, pop_size)
        # Resolve collisions with self
        r1_idxs = np.where(r1_idxs == np.arange(pop_size), (r1_idxs + 1) % pop_size, r1_idxs)
        x_r1 = pop[r1_idxs]
        
        # x_r2 (Random from Pop U Archive)
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_set = np.vstack((pop, archive_np))
        else:
            union_set = pop
        
        union_size = len(union_set)
        r2_idxs = np.random.randint(0, union_size, pop_size)
        # We ignore r2 collision correction for speed; collision prob is low with archive
        x_r2 = union_set[r2_idxs]
        
        # Mutation Vector V
        # v = x + F*(xp - x) + F*(xr1 - xr2)
        f_col = f[:, None]
        mutant = x_curr + f_col * (x_pbest - x_curr) + f_col * (x_r1 - x_r2)
        
        # 6. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        # Enforce at least one dimension
        rows = np.arange(pop_size)
        cross_mask[rows, j_rand] = True
        
        trial_vecs = np.where(cross_mask, mutant, pop)
        
        # 7. Boundary Handling (Reflection)
        # If < lb, reflect: lb + (lb - x) -> 2*lb - x
        # If > ub, reflect: ub - (x - ub) -> 2*ub - x
        # Vectorized reflection logic
        for d in range(dim):
            # Lower bound violation
            bad_l = trial_vecs[:, d] < lb[d]
            trial_vecs[bad_l, d] = 2 * lb[d] - trial_vecs[bad_l, d]
            # Upper bound violation
            bad_u = trial_vecs[:, d] > ub[d]
            trial_vecs[bad_u, d] = 2 * ub[d] - trial_vecs[bad_u, d]
            
            # If still out of bounds (extreme case), clip
            trial_vecs[:, d] = np.clip(trial_vecs[:, d], lb[d], ub[d])

        # 8. Selection and Update
        fitness_delta = np.zeros(pop_size)
        success_mask = np.zeros(pop_size, dtype=bool)
        
        # Iterate for evaluation (cannot be vectorized on func)
        for i in range(pop_size):
            if time.time() >= t_limit_optimization:
                break
                
            val_trial = func(trial_vecs[i])
            
            if val_trial <= fitness[i]:
                # Prepare for archive
                if val_trial < fitness[i]:
                    fitness_delta[i] = fitness[i] - val_trial
                    success_mask[i] = True
                
                # Update Best
                if val_trial < best_val:
                    best_val = val_trial
                    best_sol = trial_vecs[i].copy()
                
                # Replace in population (deferred to keep synchronous update or immediate?)
                # SHADE is usually synchronous update for P, but we can do immediate.
                # Here we do immediate for Archive logic, but need to copy parent first.
                
                # Add parent to archive
                archive.append(pop[i].copy())
                
                # Update population
                pop[i] = trial_vecs[i]
                fitness[i] = val_trial

        # 9. Maintain Archive Size
        # Randomly remove excess
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # 10. Update Memory (Weighted Lehmer Mean)
        if np.any(success_mask):
            s_cr = cr[success_mask]
            s_f = f[success_mask]
            diffs = fitness_delta[success_mask]
            
            # Weights based on fitness improvement
            w_sum = np.sum(diffs)
            if w_sum > 0:
                weights = diffs / w_sum
                
                # Update M_cr (Weighted Mean)
                m_cr_new = np.sum(weights * s_cr)
                # Smoothing
                M_cr[k_mem] = 0.5 * M_cr[k_mem] + 0.5 * m_cr_new
                
                # Update M_f (Weighted Lehmer Mean)
                m_f_num = np.sum(weights * (s_f ** 2))
                m_f_den = np.sum(weights * s_f)
                if m_f_den > 0:
                    m_f_new = m_f_num / m_f_den
                    M_f[k_mem] = 0.5 * M_f[k_mem] + 0.5 * m_f_new
                
                k_mem = (k_mem + 1) % H_mem

        gen += 1

    # --- Phase 2: Terminal Local Search (Polishing) ---
    # We use the remaining time to perform an aggressive adaptive random walk
    # around the best solution found. This clears up the final decimal places.
    
    current_best = best_sol.copy()
    current_score = best_val
    
    # Initial step size: 2% of the domain
    step_sigma = (ub - lb) * 0.02
    
    # Adaptive counters
    improves = 0
    fails = 0
    
    while True:
        if time.time() >= t_hard_stop:
            break
            
        # Generate candidate: Gaussian perturbation
        perturbation = np.random.normal(0, 1, dim) * step_sigma
        candidate = current_best + perturbation
        candidate = np.clip(candidate, lb, ub)
        
        try:
            val = func(candidate)
        except:
            break
            
        if val < current_score:
            current_score = val
            current_best = candidate
            best_val = val # Update global return value
            
            # Success: slightly increase step or keep momentum
            improves += 1
            fails = 0
            if improves > 2:
                step_sigma *= 1.2
        else:
            # Fail: decrease step
            fails += 1
            improves = 0
            if fails > 1:
                step_sigma *= 0.5
                fails = 0
        
        # If step becomes too small, reset it to broaden search slightly
        # (prevent getting stuck in numerical noise)
        mean_step = np.mean(step_sigma)
        if mean_step < 1e-12:
             step_sigma = (ub - lb) * 0.005

    return best_val
