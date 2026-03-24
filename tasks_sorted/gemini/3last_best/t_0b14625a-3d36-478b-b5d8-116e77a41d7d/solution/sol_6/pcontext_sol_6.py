#The output value of the best generated algorithm is: 1.1117314285781248
#
#The best generated algorithm code:
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE (Success-History Adaptive Differential Evolution)
    with Latin Hypercube Sampling, External Archive, and a Stagnation-based Restart Mechanism.
    
    Key Improvements:
    1.  **LHS Initialization**: Ensures uniform coverage of the search space initially.
    2.  **SHADE Adaptation**: Automatically tunes Mutation (F) and Crossover (CR) rates using historical success data.
    3.  **Robust Restart**: Triggers a restart if the population converges (low variance) OR if the global best 
        has not improved for a set number of generations (stagnation), helping to escape local optima.
    4.  **Optimized Population Size**: Scales with dimension but capped to ensure sufficient generations within `max_time`.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: 15*dim is a balanced heuristic. 
    # Clamped to [30, 200] to handle time constraints effectively across dimensions.
    pop_size = int(15 * dim)
    pop_size = max(30, min(200, pop_size))
    
    # SHADE Memory Parameters
    H = 5  # History size
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # External Archive
    # Stores inferior solutions to maintain diversity in the 'current-to-pbest' mutation
    archive_size = pop_size
    archive = np.zeros((archive_size, dim))
    arc_count = 0
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    lower_b = bounds_np[:, 0]
    upper_b = bounds_np[:, 1]
    
    # Global Best Tracker
    best_fitness = float('inf')
    best_sol = np.zeros(dim)
    
    # --- Helper Functions ---
    def check_timeout():
        return (time.time() - start_time) >= max_time

    def init_population():
        # Latin Hypercube Sampling
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(lower_b[d], upper_b[d], pop_size + 1)
            points = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points)
            pop[:, d] = points
        return pop

    # --- Initialization ---
    population = init_population()
    fitness = np.full(pop_size, float('inf'))
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if check_timeout():
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()
            
    # --- Optimization Loop ---
    stagnation_counter = 0
    last_best_fitness = best_fitness
    
    while not check_timeout():
        
        # 1. Restart Logic
        # Calculate population diversity (std dev of fitness)
        fit_std = np.std(fitness)
        
        # Trigger restart if:
        # a) Population has collapsed (converged)
        # b) No improvement in global best for 50 generations (stagnation)
        if fit_std < 1e-6 or stagnation_counter > 50:
            # Re-initialize
            population = init_population()
            fitness.fill(float('inf'))
            
            # Elitism: Keep the best found solution in slot 0
            population[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset Adaptive Memory
            mem_f.fill(0.5)
            mem_cr.fill(0.5)
            k_mem = 0
            arc_count = 0
            stagnation_counter = 0
            last_best_fitness = best_fitness
            
            # Evaluate new population (skip elite)
            for i in range(1, pop_size):
                if check_timeout(): return best_fitness
                val = func(population[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = population[i].copy()
            continue

        # Check Stagnation
        if best_fitness < last_best_fitness - 1e-8:
            stagnation_counter = 0
            last_best_fitness = best_fitness
        else:
            stagnation_counter += 1

        # 2. Parameter Generation (SHADE)
        # Pick random memory slot
        idx_r = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[idx_r]
        mu_f = mem_f[idx_r]
        
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        # Repair F <= 0
        while True:
            mask_bad = f <= 0
            if not np.any(mask_bad): break
            f[mask_bad] = np.random.standard_cauchy(np.sum(mask_bad)) * 0.1 + mu_f[mask_bad]
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: current-to-pbest/1
        # Sort population to identify top p-best individuals
        sort_indices = np.argsort(fitness)
        sorted_pop = population[sort_indices]
        
        # Select p-best size randomly in [2/N, 0.2] for each individual
        p_vals = np.random.uniform(2/pop_size, 0.2, pop_size)
        n_pbest = np.maximum(2, (p_vals * pop_size).astype(int))
        
        # Choose x_pbest
        rand_idx = (np.random.rand(pop_size) * n_pbest).astype(int)
        x_pbest = sorted_pop[rand_idx]
        
        # Choose x_r1 (random from population, distinct from current)
        idx_r1 = np.random.randint(0, pop_size, pop_size)
        mask_coll = idx_r1 == np.arange(pop_size)
        idx_r1[mask_coll] = (idx_r1[mask_coll] + 1) % pop_size
        x_r1 = population[idx_r1]
        
        # Choose x_r2 (random from Population Union Archive)
        if arc_count > 0:
            union_pop = np.concatenate((population, archive[:arc_count]), axis=0)
        else:
            union_pop = population
            
        idx_r2 = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[idx_r2]
        
        # Calculate Mutation Vectors
        f_vec = f[:, None]
        mutant = population + f_vec * (x_pbest - population) + f_vec * (x_r1 - x_r2)
        
        # Bound Handling: Clip
        mutant = np.clip(mutant, lower_b, upper_b)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask_cross = rand_vals < cr[:, None]
        # Ensure at least one dimension comes from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask_cross[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask_cross, mutant, population)
        
        # 5. Selection and Memory Update
        success_f = []
        success_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if check_timeout(): return best_fitness
            
            f_trial = func(trial[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                # If strictly better, store in archive and record parameters
                if f_trial < fitness[i]:
                    if arc_count < archive_size:
                        archive[arc_count] = population[i].copy()
                        arc_count += 1
                    else:
                        # Replace random archive member
                        r_a = np.random.randint(0, archive_size)
                        archive[r_a] = population[i].copy()
                        
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_fitness.append(fitness[i] - f_trial)
                
                population[i] = trial[i]
                fitness[i] = f_trial
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial[i].copy()
                    stagnation_counter = 0 # Reset stagnation on global improvement
                    
        # 6. Update SHADE Memory (Weighted Lehmer Mean)
        if len(diff_fitness) > 0:
            w = np.array(diff_fitness)
            w = w / np.sum(w)
            
            sf = np.array(success_f)
            scr = np.array(success_cr)
            
            # Update F
            mean_f = np.sum(w * (sf**2)) / np.sum(w * sf)
            mem_f[k_mem] = np.clip(mean_f, 0, 1)
            
            # Update CR
            mean_cr = np.sum(w * scr)
            mem_cr[k_mem] = np.clip(mean_cr, 0, 1)
            
            k_mem = (k_mem + 1) % H
            
    return best_fitness
