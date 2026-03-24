#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 6.039317548958536
#2. output value is: 41.706858739061985
#3. output value is: 1.119894670088911
#
#The best result (1.11) was achieved using a form of **SHADE** with Restart. The failure of the L-SHADE-RSP (41.7) was likely due to scheduling bugs in the reduction mechanism or restart logic.
#
#The algorithm below implements **iL-SHADE-R** (Improved L-SHADE with Robust Restart). It refines the best performing approaches by:
#1.  **Strict LPSR**: Implementing Linear Population Size Reduction based on the precise "epoch" time, ensuring the population size decreases smoothly from exploration to exploitation.
#2.  **LHS Initialization**: Using Latin Hypercube Sampling for initial diversity.
#3.  **Adaptive Restart**: Restarts the population if it converges (low standard deviation) or stagnates (no global improvement for `50 + dim` generations). Crucially, it resets the LPSR schedule to fit the **remaining** time, treating it as a new optimization epoch.
#4.  **jSO-style Adaptation**: Adapts the selection pressure ($p$) linearly over time, forcing greedy convergence in the final stages.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using iL-SHADE (Improved L-SHADE) with Restart.
    
    Key Features:
    1. Linear Population Size Reduction (LPSR) aligned with time epochs.
    2. SHADE adaptation for parameters F and CR.
    3. Latin Hypercube Sampling (LHS) for initialization.
    4. Robust Restart mechanism that resets the schedule based on remaining time.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population sizing: Start large for exploration, reduce to min_pop_size.
    # We scale initial size with dimension but clamp it for safety.
    initial_pop_size = int(20 * dim)
    initial_pop_size = max(50, min(300, initial_pop_size))
    min_pop_size = 4
    
    # SHADE Memory Parameters
    H = 6  # History memory size
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive Parameters
    # Archive stores displaced solutions to maintain diversity for mutation
    arc_rate = 2.0
    
    # Stagnation Parameters
    # Scale tolerance with dimension; higher dim needs more patience.
    max_stagnation_gens = 50 + dim
    
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
        """Latin Hypercube Sampling for uniform initial coverage."""
        pop = np.zeros((size, dim))
        for d in range(dim):
            # Divide dimension into 'size' intervals
            edges = np.linspace(lb[d], ub[d], size + 1)
            # Sample uniformly from each interval
            points = np.random.uniform(edges[:-1], edges[1:])
            # Shuffle points to break correlation
            np.random.shuffle(points)
            pop[:, d] = points
        return pop

    # --- Initialization ---
    pop_size = initial_pop_size
    population = get_lhs_population(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    # We check timeout periodically to ensure responsiveness
    check_freq = max(1, int(pop_size / 5))
    
    for i in range(pop_size):
        if i % check_freq == 0 and check_timeout():
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()
            
    # Initialize Archive
    max_arc_size = int(pop_size * arc_rate)
    archive = np.zeros((max_arc_size, dim))
    arc_count = 0
    
    # --- Optimization Loop ---
    # We use 'epochs'. A restart triggers a new epoch with the remaining time.
    epoch_start_time = start_time
    epoch_duration = max_time
    stagnation_counter = 0
    
    while not check_timeout():
        
        # 1. Linear Population Size Reduction (LPSR)
        now = time.time()
        # Calculate progress relative to current epoch
        time_in_epoch = now - epoch_start_time
        progress = time_in_epoch / epoch_duration if epoch_duration > 1e-5 else 1.0
        progress = np.clip(progress, 0, 1)
        
        # Calculate target population size
        target_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
        target_size = max(min_pop_size, target_size)
        
        # Resize if necessary
        if pop_size > target_size:
            # Sort population by fitness
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Truncate to new size
            pop_size = target_size
            population = population[:pop_size]
            fitness = fitness[:pop_size]
            
            # Resize Archive (Truncate)
            curr_arc_cap = int(pop_size * arc_rate)
            if arc_count > curr_arc_cap:
                arc_count = curr_arc_cap
        
        # 2. Adaptation (jSO-inspired)
        # p (top % for p-best) decays from 0.25 to 0.05
        p_val = 0.25 - progress * 0.20
        
        # Generate SHADE Parameters (F and CR)
        r_idx = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idx]
        mu_f = mem_f[r_idx]
        
        # CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # F ~ Cauchy(mu_f, 0.1)
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        # Repair F: if <= 0 retry, if > 1 clamp
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            f[mask_bad] = np.random.standard_cauchy(np.sum(mask_bad)) * 0.1 + mu_f[mask_bad]
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: current-to-pbest/1
        # Sort to find p-best
        sorted_idx = np.argsort(fitness)
        sorted_pop = population[sorted_idx]
        
        n_pbest = max(2, int(p_val * pop_size))
        pbest_indices = np.random.randint(0, n_pbest, pop_size)
        x_pbest = sorted_pop[pbest_indices]
        
        # Select x_r1 (random != i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        mask_coll = r1_indices == np.arange(pop_size)
        r1_indices[mask_coll] = (r1_indices[mask_coll] + 1) % pop_size
        x_r1 = population[r1_indices]
        
        # Select x_r2 (random from Population U Archive)
        union_size = pop_size + arc_count
        r2_indices = np.random.randint(0, union_size, pop_size)
        
        x_r2 = np.zeros((pop_size, dim))
        # Mask for checking if r2 is in Population or Archive
        mask_in_pop = r2_indices < pop_size
        mask_in_arc = ~mask_in_pop
        
        if np.any(mask_in_pop):
            x_r2[mask_in_pop] = population[r2_indices[mask_in_pop]]
        if np.any(mask_in_arc):
            # Index in archive is r2 - pop_size
            x_r2[mask_in_arc] = archive[r2_indices[mask_in_arc] - pop_size]
            
        # Calculate Mutant Vector
        # v = x + F*(xp - x) + F*(xr1 - xr2)
        f_vec = f[:, None]
        mutant = population + f_vec * (x_pbest - population) + f_vec * (x_r1 - x_r2)
        
        # Bound Handling: Clip
        mutant = np.clip(mutant, lb, ub)
        
        # 4. Crossover: Binomial
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        # Force at least one dimension to be from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # 5. Evaluation and Selection
        success_f = []
        success_cr = []
        diff_f = []
        
        improved_global = False
        
        # Check time more frequently in small loops
        check_freq = max(1, int(pop_size / 5))
        
        for i in range(pop_size):
            if i % check_freq == 0 and check_timeout(): return best_fitness
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Improvement or Neutral: Update
                
                # If strictly better, add parent to archive
                if f_trial < fitness[i]:
                    curr_arc_cap = int(pop_size * arc_rate)
                    if arc_count < curr_arc_cap:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Replace random archive member
                        if curr_arc_cap > 0:
                            rand_rep = np.random.randint(0, curr_arc_cap)
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
                    
        # Update Stagnation
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
            
            # Mean F (Lehmer)
            mean_f = np.sum(w * (s_f**2)) / np.sum(w * s_f)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            # Mean CR (Weighted Arithmetic)
            mean_cr = np.sum(w * s_cr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
        # 7. Restart Logic
        # Condition A: Convergence (Low diversity)
        # Condition B: Stagnation (No global improvement for long time)
        do_restart = False
        if np.std(fitness) < 1e-8:
            do_restart = True
        elif stagnation_counter > max_stagnation_gens:
            do_restart = True
            
        if do_restart:
            remaining = max_time - (time.time() - start_time)
            # Only restart if >5% time remains to make it worthwhile
            if remaining > 0.05 * max_time:
                # Reset Epoch
                epoch_start_time = time.time()
                epoch_duration = remaining
                
                # Reset Population
                pop_size = initial_pop_size
                population = get_lhs_population(pop_size)
                fitness = np.full(pop_size, float('inf'))
                
                # Elitism: Inject best solution found so far into new population
                population[0] = best_sol.copy()
                fitness[0] = best_fitness
                
                # Reset SHADE Internals
                mem_f.fill(0.5)
                mem_cr.fill(0.5)
                k_mem = 0
                arc_count = 0
                stagnation_counter = 0
                
                # Evaluate New Population (Skip elite at index 0)
                for i in range(1, pop_size):
                    if i % check_freq == 0 and check_timeout(): return best_fitness
                    val = func(population[i])
                    fitness[i] = val
                    if val < best_fitness:
                        best_fitness = val
                        best_sol = population[i].copy()
                continue

    return best_fitness
