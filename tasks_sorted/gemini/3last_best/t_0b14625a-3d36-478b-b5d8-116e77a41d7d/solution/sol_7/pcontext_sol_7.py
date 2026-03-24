#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 1.119894670088911
#2. output value is: 46.17972815079878
#3. output value is: 6.608363835134583
#
#The best result (1.11) was achieved using **SHADE with Archive, LHS, and Stagnation Restart** using **Bound Clipping**.
#The **L-SHADE** attempt (46.18) likely failed due to an incorrect restart mechanism that did not reset the linear reduction schedule, causing the population to be decimated too early.
#The **Midpoint Correction** attempt (6.61) likely performed worse than clipping because the optimal solutions for the test functions might lie on the boundaries.
#
#The algorithm below implements **L-SHADE-RSP (Linear Success-History Adaptive Differential Evolution with Restart and Schedule Reset)**, refined to fix previous issues.
#1.  **L-SHADE**: Uses Linear Population Size Reduction (LPSR) to balance exploration (early) and exploitation (late).
#2.  **Corrected Restart**: When stagnation or convergence is detected, the algorithm restarts the population (keeping the elite) and **recalculates the reduction schedule** based on the *remaining* time. This ensures the new population has a full lifecycle to converge.
#3.  **Bound Clipping**: Returns to using standard clipping (`np.clip`), which proved superior (1.11 vs 6.61).
#4.  **Robust Initialization**: Uses Latin Hypercube Sampling (LHS) with a population size clamped to `[30, 200]` to ensure speed.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Success-History Adaptive Differential Evolution)
    with Restart, Schedule Reset, and Bound Clipping.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population sizing
    # N_init: Initial population size. 18*dim is standard, but clamped to [30, 200]
    # to handle strict time limits and high dimensions.
    n_init = int(18 * dim)
    n_init = max(30, min(200, n_init))
    n_min = 4 # Minimum population size
    
    # SHADE Memory Parameters
    H = 6 # History size
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive Parameters
    arc_rate = 2.0
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Global Best Tracker
    best_fitness = float('inf')
    best_sol = np.zeros(dim)
    
    # --- Helper Functions ---
    def check_timeout():
        return (time.time() - start_time) >= max_time
    
    def get_lhs_population(size):
        # Latin Hypercube Sampling
        pop = np.zeros((size, dim))
        for d in range(dim):
            edges = np.linspace(lb[d], ub[d], size + 1)
            points = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points)
            pop[:, d] = points
        return pop

    # --- Initialization ---
    current_pop_size = n_init
    population = get_lhs_population(current_pop_size)
    fitness = np.full(current_pop_size, float('inf'))
    
    # Archive
    max_arc_size = int(n_init * arc_rate)
    archive = np.zeros((max_arc_size, dim))
    arc_count = 0
    
    # Evaluate Initial Population
    # Check time frequently enough to not overshoot max_time
    check_freq = max(1, int(current_pop_size / 5))
    
    for i in range(current_pop_size):
        if i % check_freq == 0 and check_timeout():
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()
            
    # --- Optimization Loop ---
    # We manage 'epochs'. A restart triggers a new epoch with the remaining time.
    epoch_start_time = start_time
    epoch_duration = max_time
    
    stagnation_counter = 0
    
    while not check_timeout():
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate progress relative to the current epoch
        now = time.time()
        elapsed_epoch = now - epoch_start_time
        
        if epoch_duration <= 1e-3: progress = 1.0
        else: progress = elapsed_epoch / epoch_duration
        
        if progress > 1.0: progress = 1.0
        
        # Calculate target population size
        target_size = int(round((n_min - n_init) * progress + n_init))
        target_size = max(n_min, target_size)
        
        # Apply Reduction
        if current_pop_size > target_size:
            # Sort by fitness
            sort_indices = np.argsort(fitness)
            population = population[sort_indices]
            fitness = fitness[sort_indices]
            
            # Truncate
            current_pop_size = target_size
            population = population[:current_pop_size]
            fitness = fitness[:current_pop_size]
            
            # Reduce Archive to maintain ratio
            curr_arc_cap = int(current_pop_size * arc_rate)
            if arc_count > curr_arc_cap:
                arc_count = curr_arc_cap
        
        # 2. Restart Mechanism
        # Trigger if:
        # a) Population converged (low std dev)
        # b) Stagnation (no improvement for 40 gens)
        do_restart = False
        if np.std(fitness) < 1e-8:
            do_restart = True
        elif stagnation_counter > 40:
            do_restart = True
            
        if do_restart:
            remaining = max_time - (time.time() - start_time)
            # Only restart if significant time remains (> 5%)
            if remaining > 0.05 * max_time:
                # Reset Epoch Schedule
                epoch_start_time = time.time()
                epoch_duration = remaining
                
                # Reset Population
                current_pop_size = n_init
                population = get_lhs_population(current_pop_size)
                fitness = np.full(current_pop_size, float('inf'))
                
                # Elitism: Keep global best
                population[0] = best_sol.copy()
                fitness[0] = best_fitness
                
                # Reset SHADE Internals
                mem_f.fill(0.5)
                mem_cr.fill(0.5)
                k_mem = 0
                arc_count = 0
                stagnation_counter = 0
                
                # Evaluate New Population (Skip elite at 0)
                for i in range(1, current_pop_size):
                    if i % check_freq == 0 and check_timeout(): return best_fitness
                    val = func(population[i])
                    fitness[i] = val
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = population[i].copy()
                continue
        
        # 3. SHADE Parameter Generation
        # Select memory slot
        r_idx = np.random.randint(0, H, current_pop_size)
        mu_cr = mem_cr[r_idx]
        mu_f = mem_f[r_idx]
        
        # CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, current_pop_size)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(mu_f, 0.1)
        f = np.random.standard_cauchy(current_pop_size) * 0.1 + mu_f
        # Repair F
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            f[mask_bad] = np.random.standard_cauchy(np.sum(mask_bad)) * 0.1 + mu_f[mask_bad]
        f = np.minimum(f, 1.0)
        
        # 4. Mutation: current-to-pbest/1
        # Sort for pbest selection
        sorted_indices = np.argsort(fitness)
        sorted_pop = population[sorted_indices]
        
        # Random p in [2/N, 0.2]
        p_vals = np.random.uniform(2.0/current_pop_size, 0.2, current_pop_size)
        n_pbest = (p_vals * current_pop_size).astype(int)
        n_pbest = np.maximum(2, n_pbest)
        
        # Select x_pbest
        rand_sel = (np.random.rand(current_pop_size) * n_pbest).astype(int)
        x_pbest = sorted_pop[rand_sel]
        
        # Select x_r1 != i
        idx_r1 = np.random.randint(0, current_pop_size, current_pop_size)
        mask_coll = idx_r1 == np.arange(current_pop_size)
        idx_r1[mask_coll] = (idx_r1[mask_coll] + 1) % current_pop_size
        x_r1 = population[idx_r1]
        
        # Select x_r2 from Union(Pop, Archive)
        union_size = current_pop_size + arc_count
        idx_r2 = np.random.randint(0, union_size, current_pop_size)
        
        mask_pop = idx_r2 < current_pop_size
        mask_arc = ~mask_pop
        
        x_r2 = np.empty((current_pop_size, dim))
        if np.any(mask_pop):
            x_r2[mask_pop] = population[idx_r2[mask_pop]]
        if np.any(mask_arc):
            x_r2[mask_arc] = archive[idx_r2[mask_arc] - current_pop_size]
            
        # Compute Mutant
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        f_vec = f[:, None]
        mutant = population + f_vec * (x_pbest - population) + f_vec * (x_r1 - x_r2)
        
        # Bound Clipping (Proven better than midpoint here)
        mutant = np.clip(mutant, lb, ub)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(current_pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, current_pop_size)
        mask_cross[np.arange(current_pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # 6. Evaluation and Update
        success_f = []
        success_cr = []
        diff_f = []
        
        improved_global = False
        
        for i in range(current_pop_size):
            if i % check_freq == 0 and check_timeout(): return best_fitness
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # If strictly better, archive parent and record success
                if f_trial < fitness[i]:
                    if arc_count < max_arc_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        r_a = np.random.randint(0, max_arc_size)
                        archive[r_a] = population[i].copy()
                        
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_f.append(fitness[i] - f_trial)
                
                fitness[i] = f_trial
                population[i] = trial[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
                    improved_global = True
                    
        # Update Stagnation
        if improved_global:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # 7. Update Memory (Weighted Lehmer Mean)
        if len(diff_f) > 0:
            w = np.array(diff_f)
            w = w / np.sum(w)
            
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            mean_f = np.sum(w * (s_f**2)) / np.sum(w * s_f)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            mean_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
    return best_fitness
