#The output value of the last generated algorithm is: 43.6522096413823
#
#The following algorithm implements **L-SHADE-MTS (Linear Success-History Adaptive Differential Evolution with Multiple Trajectory Search)**.
#
##### Key Improvements
#1.  **L-SHADE Architecture**: Utilizes Linear Population Size Reduction (LPSR) to transition from exploration (many diverse solutions) to exploitation (focused search) as the time budget depletes.
#2.  **Adaptive Parameters**: Maintains a historical memory of successful $F$ and $CR$ parameters, updating them using Weighted Lehmer Means to adapt to the objective function's landscape dynamically.
#3.  **Boundary Reflection**: Instead of simple clipping, particles reflecting off boundaries ($2 \times bound - x$) prevents stagnation at the edges of the search space.
#4.  **MTS Local Search (Polishing)**: Replaces the simple random walk of previous versions with **MTS-LS1 (Multiple Trajectory Search)**. This coordinate-based local search is highly effective at refining the best solution in the final phase, drilling down into the optimum with high precision.
#5.  **Robust Restart**: Monitors population variance. If the population collapses into a single point (stagnation) before time runs out, it triggers a soft restart—preserving the best solution while scattering the rest to find new basins of attraction.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE-MTS:
    L-SHADE (Linear Success-History Adaptive Differential Evolution) with
    Linear Population Size Reduction and Terminal MTS (Multiple Trajectory Search) polishing.
    """
    
    # --- 1. Initialization and Time Management ---
    t_start = time.time()
    t_end = t_start + max_time
    
    # Reserve the last 15% of the time budget for intensive local search (polishing)
    # This ensures we get the best possible precision on the found basin.
    ls_ratio = 0.15
    t_ls_start = t_start + (1.0 - ls_ratio) * max_time
    
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    
    # Population Size Management (LPSR)
    # Start with a large population for exploration (standard heuristic: 18 * dim)
    pop_size_init = int(round(18 * dim))
    # Reduce to a minimal population for fast convergence
    pop_size_min = 4
    pop_size = pop_size_init
    
    # SHADE Memory for Adaptive Parameters
    H = 5 # Memory size
    M_cr = np.full(H, 0.5) # Memory for Crossover Rate
    M_f = np.full(H, 0.5)  # Memory for Scaling Factor
    k_mem = 0
    
    archive = [] # External archive for diversity
    
    # Initialize Population using Latin Hypercube Sampling (LHS)
    # LHS ensures a more uniform initial distribution than pure random
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(lb[d], ub[d], pop_size + 1)
        samples = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(samples)
        pop[:, d] = samples
        
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Initial Evaluation
    for i in range(pop_size):
        if time.time() >= t_end: return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_sol = pop[i].copy()
            
    # --- 2. Evolutionary Phase (L-SHADE) ---
    gen = 0
    
    while True:
        curr_time = time.time()
        if curr_time >= t_ls_start:
            break
            
        # Calculate progress (0.0 to 1.0) based on time consumed vs evolutionary budget
        evo_time_total = t_ls_start - t_start
        if evo_time_total <= 0: break
        progress = (curr_time - t_start) / evo_time_total
        progress = np.clip(progress, 0, 1)
        
        # A. Linear Population Size Reduction (LPSR)
        new_pop_size = int(round(pop_size_init + (pop_size_min - pop_size_init) * progress))
        new_pop_size = max(pop_size_min, new_pop_size)
        
        if pop_size > new_pop_size:
            # Sort population by fitness and truncate the worst
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx[:new_pop_size]]
            fitness = fitness[sort_idx[:new_pop_size]]
            
            # Resize Archive to match new population size
            if len(archive) > new_pop_size:
                # Randomly remove elements to fit
                num_remove = len(archive) - new_pop_size
                for _ in range(num_remove):
                    archive.pop(np.random.randint(0, len(archive)))
            pop_size = new_pop_size
            
        # B. Parameter Adaptation
        # Select random memory slot for each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_cr = M_cr[r_idx]
        mu_f = M_f[r_idx]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        # Handle F constraints: if <= 0 resample, if > 1 clamp to 1
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        retry_count = 0
        while True:
            bad_f = f <= 0
            if not np.any(bad_f) or retry_count > 3:
                f[bad_f] = 0.5 # Fallback value
                break
            # Resample bad F values
            f[bad_f] = mu_f[bad_f] + 0.1 * np.random.standard_cauchy(np.sum(bad_f))
            retry_count += 1
        f = np.minimum(f, 1.0)
        
        # C. Mutation: current-to-pbest/1
        # p_best rate decreases linearly from 0.2 to 0.05 to increase greediness over time
        p_val = 0.2 - (0.15 * progress)
        p_val = max(0.05, p_val)
        top_p = max(2, int(pop_size * p_val))
        
        sorted_indices = np.argsort(fitness)
        
        # x_pbest: Randomly selected from top p%
        pbest_idxs = sorted_indices[np.random.randint(0, top_p, pop_size)]
        x_pbest = pop[pbest_idxs]
        
        # x_r1: Random distinct from x_i
        r1 = np.random.randint(0, pop_size, pop_size)
        col_r1 = r1 == np.arange(pop_size)
        r1[col_r1] = (r1[col_r1] + 1) % pop_size
        x_r1 = pop[r1]
        
        # x_r2: Random distinct from x_i and x_r1, from Union(Pop, Archive)
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
        union_size = len(union_pop)
        
        r2 = np.random.randint(0, union_size, pop_size)
        # Simple collision handling
        col_r2 = (r2 == np.arange(pop_size)) | (r2 == r1)
        r2[col_r2] = np.random.randint(0, union_size, np.sum(col_r2))
        x_r2 = union_pop[r2]
        
        # Compute Mutant Vector
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        f_vec = f[:, None]
        mutant = pop + f_vec * (x_pbest - pop) + f_vec * (x_r1 - x_r2)
        
        # D. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) < cr[:, None]
        mask[np.arange(pop_size), j_rand] = True
        trial = np.where(mask, mutant, pop)
        
        # E. Boundary Handling (Reflection)
        # Improves performance near bounds compared to clipping
        # Lower violation
        bl = trial < lb
        trial[bl] = 2*lb[np.where(bl)[1]] - trial[bl]
        # Upper violation
        bu = trial > ub
        trial[bu] = 2*ub[np.where(bu)[1]] - trial[bu]
        # Final safety clip
        trial = np.clip(trial, lb, ub)
        
        # F. Selection and Archive Update
        fitness_delta = np.zeros(pop_size)
        success_mask = np.zeros(pop_size, dtype=bool)
        
        for i in range(pop_size):
            if time.time() >= t_ls_start: break
            
            val_trial = func(trial[i])
            
            # Greedy selection
            if val_trial <= fitness[i]:
                if val_trial < fitness[i]:
                    fitness_delta[i] = fitness[i] - val_trial
                    success_mask[i] = True
                    
                    # Store inferior parent in archive
                    archive.append(pop[i].copy())
                
                fitness[i] = val_trial
                pop[i] = trial[i]
                
                if val_trial < best_val:
                    best_val = val_trial
                    best_sol = trial[i].copy()
        
        # Maintain Archive Size
        while len(archive) > pop_size:
            archive.pop(np.random.randint(0, len(archive)))
            
        # G. Update Historical Memory
        if np.any(success_mask):
            s_f = f[success_mask]
            s_cr = cr[success_mask]
            diffs = fitness_delta[success_mask]
            
            # Weights based on fitness improvement
            w = diffs / np.sum(diffs)
            
            # Update M_cr (Weighted Arithmetic Mean)
            mean_scr = np.sum(w * s_cr)
            M_cr[k_mem] = 0.5 * M_cr[k_mem] + 0.5 * mean_scr
            
            # Update M_f (Weighted Lehmer Mean)
            sum_wf = np.sum(w * s_f)
            if sum_wf > 1e-12:
                mean_sf = np.sum(w * (s_f**2)) / sum_wf
                M_f[k_mem] = 0.5 * M_f[k_mem] + 0.5 * mean_sf
                
            k_mem = (k_mem + 1) % H
            
        # H. Restart Mechanism
        # If population variance is negligible, we are stuck.
        # Restart population but keep the best solution.
        if gen % 20 == 0:
            # Check convergence
            if np.std(fitness) < 1e-9:
                idx_best = np.argmin(fitness)
                for i in range(pop_size):
                    if i != idx_best:
                        pop[i] = np.random.uniform(lb, ub)
                        if time.time() >= t_ls_start: break
                        v = func(pop[i])
                        fitness[i] = v
                        if v < best_val:
                            best_val = v
                            best_sol = pop[i].copy()
                # Clear archive on restart to avoid bias from previous basin
                archive = []
                M_cr.fill(0.5)
                M_f.fill(0.5)

        gen += 1
        
    # --- 3. Local Search Phase (MTS-LS1) ---
    # Polishing the best solution found so far using Multiple Trajectory Search
    # This coordinate-descent style search is very efficient for final convergence.
    
    curr_best = best_sol.copy()
    curr_val = best_val
    
    # Initial Search Range (approx 20% of domain)
    sr = (ub - lb) * 0.2
    
    while time.time() < t_end:
        
        # Optimize dimensions in random order
        dims_order = np.arange(dim)
        np.random.shuffle(dims_order)
        
        improved_any = False
        
        for i in dims_order:
            if time.time() >= t_end: break
            
            improved_dim = False
            
            # Try Negative Step
            x_test = curr_best.copy()
            x_test[i] = np.clip(x_test[i] - sr[i], lb[i], ub[i])
            val_test = func(x_test)
            
            if val_test < curr_val:
                curr_best = x_test
                curr_val = val_test
                best_val = curr_val
                improved_dim = True
            else:
                # Try Positive Step (Half size)
                # MTS logic: if -SR fails, try +0.5*SR
                x_test = curr_best.copy()
                x_test[i] = np.clip(x_test[i] + 0.5 * sr[i], lb[i], ub[i])
                val_test = func(x_test)
                
                if val_test < curr_val:
                    curr_best = x_test
                    curr_val = val_test
                    best_val = curr_val
                    improved_dim = True
            
            if not improved_dim:
                # If dimension didn't improve, reduce its search range
                sr[i] *= 0.5
            else:
                improved_any = True
                
        # If search range becomes too small due to stagnation, reset it slightly
        # to escape numerical precision traps
        if np.max(sr) < 1e-12:
             sr = (ub - lb) * 0.05
             
    return best_val
