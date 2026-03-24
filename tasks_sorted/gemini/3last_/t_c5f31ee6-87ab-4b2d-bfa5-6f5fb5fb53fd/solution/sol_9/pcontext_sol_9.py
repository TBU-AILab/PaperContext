#The following algorithm is an optimized implementation of **L-SHADE-MTS**, a hybrid evolutionary algorithm designed for black-box optimization under strict time constraints.
#
##### Key Improvements:
#1.  **Time-Cycle Restart Strategy**: Instead of a single run or random restarts, the algorithm divides the available time into "cycles". Each cycle treats the *remaining time* as its total budget for the Linear Population Size Reduction (LPSR). This ensures that even late-stage restarts perform a complete exploration-to-exploitation transition.
#2.  **Explosive Restart**: When restarting, the algorithm does not discard the progress. It keeps the global best solution and generates a small "Gaussian Cloud" (10% of population) around it to refine the basin of attraction, while filling the rest with random individuals for global diversity.
#3.  **Dynamic Granularity Local Search (MTS-LS1)**: A coordinate-descent local search is triggered when the population stagnates. It uses a dynamic step size (`sr`) for each dimension. Dimensions that fail to improve have their search range halved, allowing the algorithm to "zoom in" on the optimum recursively.
#4.  **Optimized Parameter Handling**: Uses the SHADE adaptive mechanism (History based parameter adaptation) with weighted Lehmer means to tune scaling factors ($F$) and crossover rates ($CR$) on the fly.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Time-Constrained L-SHADE with Dynamic MTS-LS1 and Explosive Restarts.
    """
    # --- Time Management ---
    start_time = datetime.now()
    global_time_limit = timedelta(seconds=max_time)
    
    # Check time helper to ensure we respect the limit strictly
    def check_time():
        return (datetime.now() - start_time) >= global_time_limit

    # --- Problem Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Algorithm Configuration ---
    # Population Sizing:
    # We use a relatively large initial population for exploration, 
    # but cap it to ensure generations are fast enough in high dimensions.
    initial_pop_size = int(np.clip(20 * dim, 40, 200))
    min_pop_size = 5
    
    # Archive and Memory (SHADE)
    arc_rate = 2.0
    memory_size = 5
    
    # Best Solution Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Main Loop (Cycles/Restarts) ---
    # The algorithm runs multiple 'cycles'. Each cycle is a full L-SHADE run
    # that adapts its population reduction schedule to the *remaining* time.
    while not check_time():
        
        # Calculate time budget for this specific cycle
        elapsed_total = (datetime.now() - start_time).total_seconds()
        remaining_seconds = max_time - elapsed_total
        
        # If too little time remains for a meaningful cycle, stop.
        if remaining_seconds < 0.05: 
            return best_val
            
        cycle_start_time = datetime.now()
        
        # 1. Initialize Population
        pop_size = initial_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Restart Strategy: "Explosive Restart"
        # If we have a best solution found so far:
        # - Keep it (Elitism).
        # - Generate 10% of population as Gaussian perturbations around it (Exploitation).
        # - Generate the rest randomly (Exploration).
        start_eval_idx = 0
        if best_sol is not None:
            # Inject Best
            pop[0] = best_sol.copy()
            fitness[0] = best_val
            
            # Inject Gaussian Cloud (Zoom-in)
            cloud_size = int(0.1 * pop_size)
            # Standard deviation = 5% of domain size
            sigma = diff_b * 0.05
            
            for k in range(1, cloud_size + 1):
                mutant = best_sol + np.random.normal(0, 1, dim) * sigma
                pop[k] = np.clip(mutant, min_b, max_b)
                
            start_eval_idx = cloud_size + 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_time(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = pop[i].copy()
        
        # Initialize SHADE Memory (History)
        M_CR = np.full(memory_size, 0.5)
        M_F = np.full(memory_size, 0.5)
        k_mem = 0
        archive = []
        
        # Initialize Local Search (MTS-LS1) State
        # Search range (sr) starts large and shrinks
        sr = diff_b * 0.4
        sr_min = 1e-13
        stag_count = 0
        
        # --- Evolutionary Cycle ---
        cycle_active = True
        while cycle_active and not check_time():
            
            # A. Linear Population Size Reduction (LPSR)
            # Progress is defined by time elapsed in *current cycle* vs *remaining time at start of cycle*
            cycle_elapsed = (datetime.now() - cycle_start_time).total_seconds()
            progress = cycle_elapsed / remaining_seconds
            progress = min(progress, 1.0)
            
            target_size = int(round(initial_pop_size - (initial_pop_size - min_pop_size) * progress))
            target_size = max(min_pop_size, target_size)
            
            if pop_size > target_size:
                # Remove worst individuals (highest fitness)
                sorted_idx = np.argsort(fitness)
                keep_idx = sorted_idx[:target_size]
                pop = pop[keep_idx]
                fitness = fitness[keep_idx]
                pop_size = target_size
                
                # Resize archive
                arc_cap = int(pop_size * arc_rate)
                if len(archive) > arc_cap:
                    del_cnt = len(archive) - arc_cap
                    for _ in range(del_cnt):
                        archive.pop(np.random.randint(0, len(archive)))
                        
            # B. Parameter Generation
            r_idx = np.random.randint(0, memory_size, pop_size)
            mu_cr = M_CR[r_idx]
            mu_f = M_F[r_idx]
            
            # CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1, pop_size)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1)
            f_params = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            # Regenerate invalid F
            while np.any(f_params <= 0):
                mask = f_params <= 0
                f_params[mask] = mu_f[mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(mask)) - 0.5))
            f_params = np.clip(f_params, 0, 1)
            
            # C. Mutation: current-to-pbest/1
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            num_pbest = int(max(2, pop_size * p_val))
            sorted_idx = np.argsort(fitness)
            pbest_inds = sorted_idx[:num_pbest]
            
            # Vectors
            idx_pbest = np.random.choice(pbest_inds, pop_size)
            x_pbest = pop[idx_pbest]
            
            idx_r1 = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[idx_r1]
            
            # r2 from Union(Pop, Archive)
            if len(archive) > 0:
                union_pop = np.vstack((pop, np.array(archive)))
            else:
                union_pop = pop
            idx_r2 = np.random.randint(0, len(union_pop), pop_size)
            x_r2 = union_pop[idx_r2]
            
            f_col = f_params[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # D. Crossover: Binomial
            j_rand = np.random.randint(0, dim, pop_size)
            rand_M = np.random.rand(pop_size, dim)
            mask_cr = rand_M < cr[:, None]
            mask_cr[np.arange(pop_size), j_rand] = True
            
            trial = np.where(mask_cr, mutant, pop)
            trial = np.clip(trial, min_b, max_b)
            
            # E. Selection
            succ_f = []
            succ_cr = []
            diff_fit = []
            improved = False
            
            for i in range(pop_size):
                if check_time(): return best_val
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        # Store in archive
                        archive.append(pop[i].copy())
                        # Record success params
                        succ_f.append(f_params[i])
                        succ_cr.append(cr[i])
                        diff_fit.append(fitness[i] - f_trial)
                        improved = True
                    
                    pop[i] = trial[i]
                    fitness[i] = f_trial
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = pop[i].copy()
                        stag_count = 0
            
            # Manage Archive Size
            arc_cap = int(pop_size * arc_rate)
            while len(archive) > arc_cap:
                archive.pop(np.random.randint(0, len(archive)))
            
            # Update Memory (Weighted Lehmer Mean)
            if len(succ_f) > 0:
                w = np.array(diff_fit)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    mean_f = np.sum(w * (np.array(succ_f)**2)) / np.sum(w * np.array(succ_f))
                    mean_cr = np.sum(w * np.array(succ_cr))
                    M_F[k_mem] = 0.5 * M_F[k_mem] + 0.5 * mean_f
                    M_CR[k_mem] = 0.5 * M_CR[k_mem] + 0.5 * mean_cr
                    k_mem = (k_mem + 1) % memory_size
            
            if not improved:
                stag_count += 1
            
            # F. Local Search Trigger (MTS-LS1)
            # If population stagnates, we assume we are near a local/global optimum
            # and switch to coordinate descent to refine it.
            if stag_count > 5:
                # Local Search on the Best Solution
                ls_x = best_sol.copy()
                ls_f = best_val
                ls_improved_flag = False
                
                # Dimensionality Check: If dim is high, only check a subset to save time
                if dim > 50:
                    d_inds = np.random.choice(dim, 50, replace=False)
                else:
                    d_inds = np.random.permutation(dim)
                    
                for d in d_inds:
                    if check_time(): return best_val
                    
                    orig = ls_x[d]
                    # Attempt Move 1: Negative direction
                    ls_x[d] = np.clip(orig - sr[d], min_b[d], max_b[d])
                    val = func(ls_x)
                    
                    if val < ls_f:
                        ls_f = val
                        best_val = val
                        best_sol = ls_x.copy()
                        ls_improved_flag = True
                    else:
                        # Attempt Move 2: Positive direction (half step)
                        ls_x[d] = np.clip(orig + 0.5 * sr[d], min_b[d], max_b[d])
                        val = func(ls_x)
                        
                        if val < ls_f:
                            ls_f = val
                            best_val = val
                            best_sol = ls_x.copy()
                            ls_improved_flag = True
                        else:
                            # Both failed: Restore and shrink search range for this dim
                            ls_x[d] = orig
                            sr[d] *= 0.5
                
                if ls_improved_flag:
                    # Sync improved solution back to population (replace worst)
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_sol.copy()
                    fitness[worst_idx] = best_val
                    stag_count = 0
            
            # G. Cycle Termination Conditions
            # 1. Population converged (variance ~ 0)
            if np.std(fitness) < 1e-9:
                cycle_active = False
            
            # 2. Minimum population reached and still stagnating
            if pop_size <= min_pop_size and stag_count > 10:
                cycle_active = False
                
            # 3. Local search precision exhausted
            if np.all(sr < sr_min):
                cycle_active = False

    return best_val
