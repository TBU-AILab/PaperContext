#The output value of the best generated algorithm is: 1.1117314285781248
#
#The best generated algorithm code:
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using an enhanced L-SHADE algorithm (Linear Success-History Adaptive 
    Differential Evolution) with Restart and jSO-inspired parameter adaptation.
    
    Key Features:
    1.  **L-SHADE backbone**: Linearly reduces population size (LPSR) to transition from exploration to exploitation.
    2.  **jSO Adaptation**: Linearly reduces the 'p' parameter (top % of population) for mutation selection,
        increasing selection pressure as the search progresses.
    3.  **LHS Initialization**: Ensures uniform initial coverage of the search space.
    4.  **Epoch-based Restart**: If convergence or stagnation occurs, restarts the population (keeping the elite)
        and resets the reduction schedule based on the *remaining* time.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population Sizing
    # N_init: Initial size. 25*dim is a robust heuristic for complex landscapes.
    # Clamped to [50, 300] to ensure reasonable speed/coverage trade-off.
    raw_n_init = int(25 * dim)
    n_init = max(50, min(300, raw_n_init))
    n_min = 4 # Minimum population size at the end of schedule
    
    # Archive parameters
    arc_rate = 2.0 # Archive capacity relative to current population size
    
    # SHADE Memory parameters
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Bounds processing
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
    curr_pop_size = n_init
    population = get_lhs_population(curr_pop_size)
    fitness = np.full(curr_pop_size, float('inf'))
    
    # Archive setup
    max_arc_size = int(n_init * arc_rate)
    archive = np.zeros((max_arc_size, dim))
    arc_count = 0
    
    # Initial Evaluation
    check_freq = max(1, int(curr_pop_size / 4))
    
    for i in range(curr_pop_size):
        if i % check_freq == 0 and check_timeout():
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()
            
    # --- Optimization Loop ---
    # We manage 'epochs'. A restart creates a new epoch with the remaining time.
    epoch_start_time = start_time
    epoch_duration = max_time 
    
    stagnation_counter = 0
    
    while not check_timeout():
        
        # 1. Linear Population Size Reduction (LPSR)
        now = time.time()
        # Calculate progress ratio (0.0 to 1.0) based on current epoch
        progress = (now - epoch_start_time) / epoch_duration if epoch_duration > 1e-9 else 1.0
        if progress > 1.0: progress = 1.0
        
        # Calculate new target population size
        target_size = int(round((n_min - n_init) * progress + n_init))
        target_size = max(n_min, target_size)
        
        # Reduce if necessary
        if curr_pop_size > target_size:
            # Sort by fitness
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Truncate weakest
            curr_pop_size = target_size
            population = population[:curr_pop_size]
            fitness = fitness[:curr_pop_size]
            
            # Adjust Archive size to maintain ratio
            curr_arc_cap = int(curr_pop_size * arc_rate)
            if arc_count > curr_arc_cap:
                arc_count = curr_arc_cap
                
        # 2. Parameter Adaptation
        # jSO Strategy: 'p' value for current-to-pbest decays linearly.
        # Starts at 0.25 (exploration), ends at 0.05 (exploitation).
        p_val = 0.25 - progress * (0.20)
        
        # Select Memory Index
        r_mem = np.random.randint(0, H, curr_pop_size)
        mu_cr = mem_cr[r_mem]
        mu_f = mem_f[r_mem]
        
        # Generate CR (Normal Distribution)
        cr = np.random.normal(mu_cr, 0.1, curr_pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F (Cauchy Distribution)
        f = np.random.standard_cauchy(curr_pop_size) * 0.1 + mu_f
        # Repair F <= 0
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            f[mask_bad] = np.random.standard_cauchy(np.sum(mask_bad)) * 0.1 + mu_f[mask_bad]
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: current-to-pbest/1
        # Sort population to find top p%
        sorted_idx = np.argsort(fitness)
        sorted_pop = population[sorted_idx]
        
        # Number of individuals in p-best set
        n_pbest = max(2, int(p_val * curr_pop_size))
        
        # Select x_pbest
        pbest_indices = np.random.randint(0, n_pbest, curr_pop_size)
        x_pbest = sorted_pop[pbest_indices]
        
        # Select x_r1 (Random from Pop, != i)
        r1_indices = np.random.randint(0, curr_pop_size, curr_pop_size)
        mask_coll = r1_indices == np.arange(curr_pop_size)
        r1_indices[mask_coll] = (r1_indices[mask_coll] + 1) % curr_pop_size
        x_r1 = population[r1_indices]
        
        # Select x_r2 (Random from Pop U Archive, != i, != r1)
        union_size = curr_pop_size + arc_count
        r2_indices = np.random.randint(0, union_size, curr_pop_size)
        
        mask_pop = r2_indices < curr_pop_size
        mask_arc = ~mask_pop
        
        x_r2 = np.empty((curr_pop_size, dim))
        x_r2[mask_pop] = population[r2_indices[mask_pop]]
        if np.any(mask_arc):
            arc_idx = r2_indices[mask_arc] - curr_pop_size
            x_r2[mask_arc] = archive[arc_idx]
            
        # Compute Mutant Vectors
        f_vec = f[:, None]
        mutant = population + f_vec * (x_pbest - population) + f_vec * (x_r1 - x_r2)
        
        # Bound Handling: Clipping
        mutant = np.clip(mutant, lb, ub)
        
        # 4. Crossover: Binomial
        rand_vals = np.random.rand(curr_pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        j_rand = np.random.randint(0, dim, curr_pop_size)
        mask_cross[np.arange(curr_pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # 5. Evaluation and Selection
        success_f = []
        success_cr = []
        diff_f = []
        
        improved_global = False
        check_freq = max(1, int(curr_pop_size / 4))
        
        for i in range(curr_pop_size):
            if i % check_freq == 0 and check_timeout(): return best_fitness
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Update Archive if strictly better
                if f_trial < fitness[i]:
                    if arc_count < max_arc_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        rand_rep = np.random.randint(0, max_arc_size)
                        archive[rand_rep] = population[i].copy()
                        
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_f.append(fitness[i] - f_trial)
                
                fitness[i] = f_trial
                population[i] = trial[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
                    improved_global = True
        
        if improved_global:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # 6. Update Memory (Weighted Lehmer Mean)
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
            
        # 7. Restart Logic
        # Trigger if:
        # a) Population has collapsed (std dev < 1e-9)
        # b) Global best hasn't improved for 40 generations (Stagnation)
        do_restart = False
        if np.std(fitness) < 1e-9:
            do_restart = True
        elif stagnation_counter > 40:
            do_restart = True
            
        if do_restart:
            remaining = max_time - (time.time() - start_time)
            # Only restart if significant time remains (> 5%)
            if remaining > 0.05 * max_time:
                # Reset Schedule
                epoch_start_time = time.time()
                epoch_duration = remaining
                
                # Reset Population (keep elite)
                curr_pop_size = n_init
                population = get_lhs_population(curr_pop_size)
                fitness.fill(float('inf'))
                
                population[0] = best_sol.copy()
                fitness[0] = best_fitness
                
                # Reset Internals
                mem_f.fill(0.5)
                mem_cr.fill(0.5)
                arc_count = 0
                stagnation_counter = 0
                
                # Evaluate New Population
                for i in range(1, curr_pop_size):
                    if i % check_freq == 0 and check_timeout(): return best_fitness
                    val = func(population[i])
                    fitness[i] = val
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = population[i].copy()
                        
    return best_fitness
