The output values and codes for the last generated algorithms are as follows:
1. output value is: 15.230174087252792

 1. algorithm code is:
#The following is a fully operational and self-contained algorithm using a **Success-History based Adaptive Differential Evolution (SHADE)** strategy with **Restarts**.
#
#### Algorithm Description
#To achieve the minimum output value within a limited time, this algorithm implements **SHADE with Linear Population Sizing and Restarts**. This is an advanced variant of Differential Evolution that self-adapts its control parameters ($F$ and $CR$) based on the success of previous generations, making it highly effective for diverse and unknown fitness landscapes.
#
#1.  **Strategy (DE/current-to-pbest/1/bin)**: The mutation strategy moves individuals towards a solution selected from the top $p\%$ best individuals ($p$-best). This balances exploitation (convergence) and exploration better than targeting the single best solution.
#2.  **External Archive**: An archive stores recently replaced inferior solutions. These are used in the mutation step (as the second difference vector) to maintain population diversity and prevent premature convergence.
#3.  **Self-Adaptive Parameters (SHADE)**: Instead of fixed or random parameters, mutation factor $F$ and crossover rate $CR$ are adapted using a historical memory ($M_F, M_{CR}$). Successful parameters that lead to fitness improvements are used to update this memory, guiding the algorithm towards optimal settings automatically.
#4.  **Restart Mechanism**: The algorithm detects stagnation (no improvement in best fitness) or convergence (low population variance). Upon detection, it restarts with a fresh population but carries over the global best solution (Elitism) to ensure monotonic improvement.
#5.  **Adaptive Population**: The population size is heuristically determined based on the dimension ($15\sqrt{D}$) to ensure adequate search density without exceeding the time budget.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Success-History based Adaptive 
    Differential Evolution (SHADE) with Restarts.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracker across all restarts
    global_best_fit = float('inf')
    global_best_sol = None
    
    # Helper to check time budget
    def is_time_left():
        return (time.time() - start_time) < max_time

    # --- Main Restart Loop ---
    # Restarts the population if it converges or stagnates.
    while is_time_left():
        
        # Adaptive Population Size
        # Heuristic: 15 * sqrt(D), clamped to [30, 80] for efficiency
        pop_size = int(15 * np.sqrt(dim))
        pop_size = max(30, min(80, pop_size))
        
        # --- SHADE Memory Initialization ---
        # Memory stores successful F and CR values to guide future generation.
        # H (history size) = 5
        mem_size = 5
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.5)
        k_mem = 0  # Memory index pointer
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous restarts
        start_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_fit
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if not is_time_left(): return global_best_fit
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = pop[i].copy()
                
        # External Archive for 'current-to-pbest' diversity
        # Stores inferior solutions replaced by better offspring
        archive = []
        
        # Stagnation tracking
        stag_counter = 0
        prev_min_fit = np.min(fitness)
        
        # --- Evolution Loop ---
        while is_time_left():
            # 1. Sort Population
            # Necessary for 'current-to-pbest' to easily pick top p%
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            # 2. Convergence / Stagnation Check
            current_min = fitness[0]
            
            # Check if population variance is zero or no improvement seen
            if np.std(fitness) < 1e-8 or abs(current_min - prev_min_fit) < 1e-8:
                stag_counter += 1
            else:
                stag_counter = 0
                prev_min_fit = current_min
                
            if stag_counter > 30: # Trigger restart if stuck
                break
                
            # 3. Parameter Generation (SHADE Strategy)
            # Pick random memory slot for each individual
            r_idx = np.random.randint(0, mem_size, pop_size)
            mu_f = mem_F[r_idx]
            mu_cr = mem_CR[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr_g = np.random.normal(mu_cr, 0.1)
            cr_g = np.clip(cr_g, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1), clipped (0, 1]
            # Cauchy = mu + scale * tan(pi * (rand - 0.5))
            f_g = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f_g = np.clip(f_g, 0.1, 1.0) 
            
            # 4. Mutation: DE/current-to-pbest/1
            # Select p-best (top 15% or at least 2)
            p_limit = max(2, int(0.15 * pop_size))
            p_best_idxs = np.random.randint(0, p_limit, pop_size)
            x_pbest = pop[p_best_idxs]
            
            # Select r1 (from Population)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_idxs]
            
            # Select r2 (from Population U Archive)
            if len(archive) > 0:
                arc_arr = np.array(archive)
                n_arc = len(archive)
                # Generate random indices covering both pop and archive
                r2_raw_idxs = np.random.randint(0, pop_size + n_arc, pop_size)
                
                # Construct union (concatenation is efficient for these sizes)
                union_pop = np.concatenate((pop, arc_arr), axis=0)
                x_r2 = union_pop[r2_raw_idxs]
            else:
                r2_idxs = np.random.randint(0, pop_size, pop_size)
                x_r2 = pop[r2_idxs]
            
            # Mutation Equation
            # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            # Note: 'pop' is sorted, so pop[i] is current 'x'
            mutant = pop + f_g[:, None] * (x_pbest - pop) + f_g[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 5. Crossover
            # Binomial crossover
            cross_points = np.random.rand(pop_size, dim) < cr_g[:, None]
            # Ensure at least one parameter comes from mutant (DE constraint)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_points[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_points, mutant, pop)
            
            # 6. Evaluation & Selection
            succ_F = []
            succ_CR = []
            diff_fitness = []
            
            for i in range(pop_size):
                if not is_time_left(): return global_best_fit
                
                f_trial = func(trial[i])
                f_old = fitness[i]
                
                if f_trial <= f_old:
                    # Successful Update
                    
                    # Add parent to archive (if strictly better)
                    if f_trial < f_old:
                        if len(archive) < pop_size:
                            archive.append(pop[i].copy())
                        else:
                            # If archive full, replace random solution
                            rem = np.random.randint(0, pop_size)
                            archive[rem] = pop[i].copy()
                        
                        # Record success for memory update
                        succ_F.append(f_g[i])
                        succ_CR.append(cr_g[i])
                        diff_fitness.append(f_old - f_trial)
                    
                    # Update Population
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    # Update Global Best
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial
                        global_best_sol = trial[i].copy()
            
            # 7. Update SHADE Memory
            if len(succ_F) > 0:
                succ_F = np.array(succ_F)
                succ_CR = np.array(succ_CR)
                weights = np.array(diff_fitness)
                total_w = np.sum(weights)
                
                if total_w > 0:
                    weights /= total_w
                    
                    # Weighted Lehmer Mean for F
                    mean_F = np.sum(weights * (succ_F ** 2)) / np.sum(weights * succ_F)
                    
                    # Weighted Mean for CR
                    mean_CR = np.sum(weights * succ_CR)
                    
                    # Update Memory at current pointer
                    mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * mean_F
                    mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * mean_CR
                    
                    k_mem = (k_mem + 1) % mem_size

    return global_best_fit


2. output value is: 8.627763194400856

 2. algorithm code is:
#The following is a fully operational and self-contained algorithm using an improved **Differential Evolution (DE)** strategy optimized for limited-time execution.
#
##### Algorithm Description
#To achieve a lower minimum value (better fitness) than previous iterations, this algorithm implements **DE/current-to-pbest/1/bin with Adaptive Restarts**. 
#1.  **Strategy (DE/current-to-pbest/1/bin)**: Unlike `current-to-best`, which can converge prematurely to local optima, `current-to-pbest` guides the population towards a random individual selected from the top $p\%$ best candidates. This balances convergence speed with population diversity.
#2.  **Adaptive Population Sizing**: The population size is heuristically derived from the dimension ($20\sqrt{D}$) to ensure sufficient search density for higher dimensions while remaining fast for lower ones.
#3.  **Randomized Control Parameters**: Instead of fixed values, mutation factor $F$ and crossover rate $CR$ are randomized per individual for every generation. This mimics self-adaptive behaviors (like SHADE), allowing the algorithm to dynamically handle different fitness landscape features.
#4.  **Stagnation Detection & Restarts**: The algorithm monitors fitness improvements. If the population standard deviation drops (convergence) or the best fitness doesn't improve for a set number of generations (stagnation), it restarts.
#5.  **Elitism**: On restart, the global best solution is carried over to the new population to prevent regression.
#
##### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Differential Evolution with the 
    DE/current-to-pbest/1/bin strategy and adaptive restarts.
    """
    # Initialize timing
    start_time = time.time()
    
    # Precompute bounds for vectorization efficiency
    bounds_arr = np.array(bounds)
    min_bound = bounds_arr[:, 0]
    max_bound = bounds_arr[:, 1]
    diff_bound = max_bound - min_bound
    
    # --- Adaptive Configuration ---
    # Population size scaling: slightly larger than standard to support p-best diversity
    # Heuristic: pop_size = 20 * sqrt(dim), clamped between 20 and 100 for efficiency
    pop_size = int(20 * np.sqrt(dim))
    pop_size = max(20, min(100, pop_size))
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    best_solution = None
    
    # Helper to enforce time limit
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    while not is_time_up():
        
        # 1. Initialize Population
        # Uniform random initialization within bounds
        population = min_bound + np.random.rand(pop_size, dim) * diff_bound
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        start_idx = 0
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
            start_idx = 1 # Skip re-evaluating the injected best
        
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if is_time_up(): return best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
                
        # 2. Differential Evolution Loop
        stagnation_count = 0
        prev_min_fit = np.min(fitness)
        
        while not is_time_up():
            # Sort population by fitness (required for current-to-pbest strategy)
            # This moves the best individuals to the top of the arrays
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            current_min_fit = fitness[0]
            
            # --- Convergence & Stagnation Checks ---
            # If population variance is negligible, we have converged -> Restart
            if np.std(fitness) < 1e-8:
                break
            
            # If best fitness hasn't improved significantly, increment stagnation counter
            if abs(current_min_fit - prev_min_fit) < 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_min_fit = current_min_fit
            
            # Trigger restart if stuck in a local optimum for too long
            if stagnation_count > 30:
                break
            
            # --- Strategy: DE/current-to-pbest/1/bin ---
            # 'p-best': Select target from the top p% (p_val) individuals
            # This maintains diversity better than targeting solely the single best
            p_val = max(2, int(0.15 * pop_size))
            
            # Vectorized Indices Generation
            # 1. p-best indices: random integer from [0, p_val) for each individual
            idxs_pbest = np.random.randint(0, p_val, pop_size)
            x_pbest = population[idxs_pbest]
            
            # 2. r1, r2 indices: random integers from whole population
            idxs_r1 = np.random.randint(0, pop_size, pop_size)
            idxs_r2 = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[idxs_r1]
            x_r2 = population[idxs_r2]
            
            # Randomized Control Parameters (Dithering)
            # F ~ Uniform(0.5, 0.95), CR ~ Uniform(0.8, 1.0)
            # Randomizing per individual helps escape local basins
            F = np.random.uniform(0.5, 0.95, (pop_size, 1))
            CR = np.random.uniform(0.8, 1.0, (pop_size, 1))
            
            # Mutation Equation: v = x + F(x_pbest - x) + F(x_r1 - x_r2)
            # Note: 'population' is the current x because we sorted it
            mutant = population + F * (x_pbest - population) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(pop_size, dim) < CR
            # Constraint: Ensure at least one dimension is inherited from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            # Construct Trial Population
            trial = np.where(cross_mask, mutant, population)
            
            # Bounds Constraint (Clipping)
            trial = np.clip(trial, min_bound, max_bound)
            
            # --- Selection (Evaluation) ---
            # Evaluate trial vectors and apply greedy selection
            for i in range(pop_size):
                if is_time_up(): return best_fitness
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial[i]
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial[i].copy()
                        
    return best_fitness


3. output value is: 44.9935846844537

 3. algorithm code is:
#The following is a fully operational and self-contained algorithm using an improved **Differential Evolution (DE)** strategy.
#
#### Algorithm Description
#To minimize the output value within the limited time, this algorithm implements **Differential Evolution with Restarts and Dithering**.
#1.  **Strategy (DE/current-to-best/1/bin)**: This variant balances exploration and exploitation better than the standard `rand/1` strategy by guiding individuals towards the best solution found so far while maintaining diversity using difference vectors.
#2.  **Restarts**: If the population converges (standard deviation of fitness becomes negligible) before the time limit, the algorithm restarts with a fresh random population. This prevents the algorithm from getting stuck in local optima.
#3.  **Elitism**: During restarts, the global best solution found is injected into the new population to ensure the search never regresses.
#4.  **Dithering**: The mutation factor $F$ is randomized for each generation to handle various fitness landscape scales dynamically.
#5.  **Vectorization**: NumPy operations are used for mutation and crossover to maximize computational efficiency, allowing more function evaluations within `max_time`.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Differential Evolution with Restarts 
    and the DE/current-to-best/1/bin strategy.
    """
    start_time = time.time()
    
    # --- Algorithm Configuration ---
    # Population size: Adaptive based on dimension.
    # Scaled to ensure enough diversity for high dims, but fast enough for low dims.
    pop_size = int(10 * np.sqrt(dim))
    pop_size = max(10, pop_size)
    pop_size = min(50, pop_size)
    
    # Crossover Probability
    CR = 0.9

    # Pre-process bounds for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    best_solution = None

    # Helper function to check if time budget is exhausted
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    # Loop continues restarting the population until time runs out
    while not is_time_up():
        
        # 1. Initialize Population
        # Uniform random distribution within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        # This ensures we don't lose the best candidate during a restart.
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if is_time_up(): return best_fitness
            
            # Skip re-evaluation of the injected best solution
            if best_solution is not None and i == 0:
                continue

            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
        
        # 2. Evolution Generations Loop
        # Run until the population converges or time runs out
        while not is_time_up():
            
            # Check for Convergence
            # If the population variance is extremely low, we are likely in a local optimum.
            # Break the inner loop to trigger a restart.
            if np.std(fitness) < 1e-8:
                break
            
            # Dithering: Randomize Mutation Factor F [0.5, 1.0] per generation
            F = 0.5 + 0.5 * np.random.rand()
            
            # Find the best individual in the current population for the mutation strategy
            current_best_idx = np.argmin(fitness)
            x_best = population[current_best_idx]
            
            # --- Vectorized Mutation & Crossover ---
            
            # Generate random indices for difference vectors
            # Note: We allow minor collisions (r1==r2 or r1==i) for computational speed.
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Strategy: DE/current-to-best/1/bin
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This moves every individual towards the best, plus a random difference for diversity.
            mutant = population + F * (x_best - population) + F * (population[r1] - population[r2])
            
            # Binary Crossover
            cross_mask = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one parameter is taken from the mutant (DE constraint)
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            # Create trial population
            trial_pop = np.where(cross_mask, mutant, population)
            
            # Apply Bounds Constraints (Clipping)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection (Evaluation) ---
            for i in range(pop_size):
                if is_time_up(): return best_fitness
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection: Accept if better or equal
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_pop[i].copy()

    return best_fitness


The output values and codes for the best generated algorithms are as follows:
1. output value is: 8.627763194400856

 1. algorithm code is:
#The following is a fully operational and self-contained algorithm using an improved **Differential Evolution (DE)** strategy optimized for limited-time execution.
#
##### Algorithm Description
#To achieve a lower minimum value (better fitness) than previous iterations, this algorithm implements **DE/current-to-pbest/1/bin with Adaptive Restarts**. 
#1.  **Strategy (DE/current-to-pbest/1/bin)**: Unlike `current-to-best`, which can converge prematurely to local optima, `current-to-pbest` guides the population towards a random individual selected from the top $p\%$ best candidates. This balances convergence speed with population diversity.
#2.  **Adaptive Population Sizing**: The population size is heuristically derived from the dimension ($20\sqrt{D}$) to ensure sufficient search density for higher dimensions while remaining fast for lower ones.
#3.  **Randomized Control Parameters**: Instead of fixed values, mutation factor $F$ and crossover rate $CR$ are randomized per individual for every generation. This mimics self-adaptive behaviors (like SHADE), allowing the algorithm to dynamically handle different fitness landscape features.
#4.  **Stagnation Detection & Restarts**: The algorithm monitors fitness improvements. If the population standard deviation drops (convergence) or the best fitness doesn't improve for a set number of generations (stagnation), it restarts.
#5.  **Elitism**: On restart, the global best solution is carried over to the new population to prevent regression.
#
##### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Differential Evolution with the 
    DE/current-to-pbest/1/bin strategy and adaptive restarts.
    """
    # Initialize timing
    start_time = time.time()
    
    # Precompute bounds for vectorization efficiency
    bounds_arr = np.array(bounds)
    min_bound = bounds_arr[:, 0]
    max_bound = bounds_arr[:, 1]
    diff_bound = max_bound - min_bound
    
    # --- Adaptive Configuration ---
    # Population size scaling: slightly larger than standard to support p-best diversity
    # Heuristic: pop_size = 20 * sqrt(dim), clamped between 20 and 100 for efficiency
    pop_size = int(20 * np.sqrt(dim))
    pop_size = max(20, min(100, pop_size))
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    best_solution = None
    
    # Helper to enforce time limit
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    while not is_time_up():
        
        # 1. Initialize Population
        # Uniform random initialization within bounds
        population = min_bound + np.random.rand(pop_size, dim) * diff_bound
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        start_idx = 0
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
            start_idx = 1 # Skip re-evaluating the injected best
        
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if is_time_up(): return best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
                
        # 2. Differential Evolution Loop
        stagnation_count = 0
        prev_min_fit = np.min(fitness)
        
        while not is_time_up():
            # Sort population by fitness (required for current-to-pbest strategy)
            # This moves the best individuals to the top of the arrays
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            current_min_fit = fitness[0]
            
            # --- Convergence & Stagnation Checks ---
            # If population variance is negligible, we have converged -> Restart
            if np.std(fitness) < 1e-8:
                break
            
            # If best fitness hasn't improved significantly, increment stagnation counter
            if abs(current_min_fit - prev_min_fit) < 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_min_fit = current_min_fit
            
            # Trigger restart if stuck in a local optimum for too long
            if stagnation_count > 30:
                break
            
            # --- Strategy: DE/current-to-pbest/1/bin ---
            # 'p-best': Select target from the top p% (p_val) individuals
            # This maintains diversity better than targeting solely the single best
            p_val = max(2, int(0.15 * pop_size))
            
            # Vectorized Indices Generation
            # 1. p-best indices: random integer from [0, p_val) for each individual
            idxs_pbest = np.random.randint(0, p_val, pop_size)
            x_pbest = population[idxs_pbest]
            
            # 2. r1, r2 indices: random integers from whole population
            idxs_r1 = np.random.randint(0, pop_size, pop_size)
            idxs_r2 = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[idxs_r1]
            x_r2 = population[idxs_r2]
            
            # Randomized Control Parameters (Dithering)
            # F ~ Uniform(0.5, 0.95), CR ~ Uniform(0.8, 1.0)
            # Randomizing per individual helps escape local basins
            F = np.random.uniform(0.5, 0.95, (pop_size, 1))
            CR = np.random.uniform(0.8, 1.0, (pop_size, 1))
            
            # Mutation Equation: v = x + F(x_pbest - x) + F(x_r1 - x_r2)
            # Note: 'population' is the current x because we sorted it
            mutant = population + F * (x_pbest - population) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(pop_size, dim) < CR
            # Constraint: Ensure at least one dimension is inherited from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            # Construct Trial Population
            trial = np.where(cross_mask, mutant, population)
            
            # Bounds Constraint (Clipping)
            trial = np.clip(trial, min_bound, max_bound)
            
            # --- Selection (Evaluation) ---
            # Evaluate trial vectors and apply greedy selection
            for i in range(pop_size):
                if is_time_up(): return best_fitness
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial[i]
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial[i].copy()
                        
    return best_fitness


2. output value is: 15.230174087252792

 2. algorithm code is:
#The following is a fully operational and self-contained algorithm using a **Success-History based Adaptive Differential Evolution (SHADE)** strategy with **Restarts**.
#
#### Algorithm Description
#To achieve the minimum output value within a limited time, this algorithm implements **SHADE with Linear Population Sizing and Restarts**. This is an advanced variant of Differential Evolution that self-adapts its control parameters ($F$ and $CR$) based on the success of previous generations, making it highly effective for diverse and unknown fitness landscapes.
#
#1.  **Strategy (DE/current-to-pbest/1/bin)**: The mutation strategy moves individuals towards a solution selected from the top $p\%$ best individuals ($p$-best). This balances exploitation (convergence) and exploration better than targeting the single best solution.
#2.  **External Archive**: An archive stores recently replaced inferior solutions. These are used in the mutation step (as the second difference vector) to maintain population diversity and prevent premature convergence.
#3.  **Self-Adaptive Parameters (SHADE)**: Instead of fixed or random parameters, mutation factor $F$ and crossover rate $CR$ are adapted using a historical memory ($M_F, M_{CR}$). Successful parameters that lead to fitness improvements are used to update this memory, guiding the algorithm towards optimal settings automatically.
#4.  **Restart Mechanism**: The algorithm detects stagnation (no improvement in best fitness) or convergence (low population variance). Upon detection, it restarts with a fresh population but carries over the global best solution (Elitism) to ensure monotonic improvement.
#5.  **Adaptive Population**: The population size is heuristically determined based on the dimension ($15\sqrt{D}$) to ensure adequate search density without exceeding the time budget.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Success-History based Adaptive 
    Differential Evolution (SHADE) with Restarts.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracker across all restarts
    global_best_fit = float('inf')
    global_best_sol = None
    
    # Helper to check time budget
    def is_time_left():
        return (time.time() - start_time) < max_time

    # --- Main Restart Loop ---
    # Restarts the population if it converges or stagnates.
    while is_time_left():
        
        # Adaptive Population Size
        # Heuristic: 15 * sqrt(D), clamped to [30, 80] for efficiency
        pop_size = int(15 * np.sqrt(dim))
        pop_size = max(30, min(80, pop_size))
        
        # --- SHADE Memory Initialization ---
        # Memory stores successful F and CR values to guide future generation.
        # H (history size) = 5
        mem_size = 5
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.5)
        k_mem = 0  # Memory index pointer
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous restarts
        start_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_fit
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if not is_time_left(): return global_best_fit
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = pop[i].copy()
                
        # External Archive for 'current-to-pbest' diversity
        # Stores inferior solutions replaced by better offspring
        archive = []
        
        # Stagnation tracking
        stag_counter = 0
        prev_min_fit = np.min(fitness)
        
        # --- Evolution Loop ---
        while is_time_left():
            # 1. Sort Population
            # Necessary for 'current-to-pbest' to easily pick top p%
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            # 2. Convergence / Stagnation Check
            current_min = fitness[0]
            
            # Check if population variance is zero or no improvement seen
            if np.std(fitness) < 1e-8 or abs(current_min - prev_min_fit) < 1e-8:
                stag_counter += 1
            else:
                stag_counter = 0
                prev_min_fit = current_min
                
            if stag_counter > 30: # Trigger restart if stuck
                break
                
            # 3. Parameter Generation (SHADE Strategy)
            # Pick random memory slot for each individual
            r_idx = np.random.randint(0, mem_size, pop_size)
            mu_f = mem_F[r_idx]
            mu_cr = mem_CR[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr_g = np.random.normal(mu_cr, 0.1)
            cr_g = np.clip(cr_g, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1), clipped (0, 1]
            # Cauchy = mu + scale * tan(pi * (rand - 0.5))
            f_g = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f_g = np.clip(f_g, 0.1, 1.0) 
            
            # 4. Mutation: DE/current-to-pbest/1
            # Select p-best (top 15% or at least 2)
            p_limit = max(2, int(0.15 * pop_size))
            p_best_idxs = np.random.randint(0, p_limit, pop_size)
            x_pbest = pop[p_best_idxs]
            
            # Select r1 (from Population)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_idxs]
            
            # Select r2 (from Population U Archive)
            if len(archive) > 0:
                arc_arr = np.array(archive)
                n_arc = len(archive)
                # Generate random indices covering both pop and archive
                r2_raw_idxs = np.random.randint(0, pop_size + n_arc, pop_size)
                
                # Construct union (concatenation is efficient for these sizes)
                union_pop = np.concatenate((pop, arc_arr), axis=0)
                x_r2 = union_pop[r2_raw_idxs]
            else:
                r2_idxs = np.random.randint(0, pop_size, pop_size)
                x_r2 = pop[r2_idxs]
            
            # Mutation Equation
            # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            # Note: 'pop' is sorted, so pop[i] is current 'x'
            mutant = pop + f_g[:, None] * (x_pbest - pop) + f_g[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 5. Crossover
            # Binomial crossover
            cross_points = np.random.rand(pop_size, dim) < cr_g[:, None]
            # Ensure at least one parameter comes from mutant (DE constraint)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_points[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_points, mutant, pop)
            
            # 6. Evaluation & Selection
            succ_F = []
            succ_CR = []
            diff_fitness = []
            
            for i in range(pop_size):
                if not is_time_left(): return global_best_fit
                
                f_trial = func(trial[i])
                f_old = fitness[i]
                
                if f_trial <= f_old:
                    # Successful Update
                    
                    # Add parent to archive (if strictly better)
                    if f_trial < f_old:
                        if len(archive) < pop_size:
                            archive.append(pop[i].copy())
                        else:
                            # If archive full, replace random solution
                            rem = np.random.randint(0, pop_size)
                            archive[rem] = pop[i].copy()
                        
                        # Record success for memory update
                        succ_F.append(f_g[i])
                        succ_CR.append(cr_g[i])
                        diff_fitness.append(f_old - f_trial)
                    
                    # Update Population
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    # Update Global Best
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial
                        global_best_sol = trial[i].copy()
            
            # 7. Update SHADE Memory
            if len(succ_F) > 0:
                succ_F = np.array(succ_F)
                succ_CR = np.array(succ_CR)
                weights = np.array(diff_fitness)
                total_w = np.sum(weights)
                
                if total_w > 0:
                    weights /= total_w
                    
                    # Weighted Lehmer Mean for F
                    mean_F = np.sum(weights * (succ_F ** 2)) / np.sum(weights * succ_F)
                    
                    # Weighted Mean for CR
                    mean_CR = np.sum(weights * succ_CR)
                    
                    # Update Memory at current pointer
                    mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * mean_F
                    mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * mean_CR
                    
                    k_mem = (k_mem + 1) % mem_size

    return global_best_fit


3. output value is: 44.9935846844537

 3. algorithm code is:
#The following is a fully operational and self-contained algorithm using an improved **Differential Evolution (DE)** strategy.
#
#### Algorithm Description
#To minimize the output value within the limited time, this algorithm implements **Differential Evolution with Restarts and Dithering**.
#1.  **Strategy (DE/current-to-best/1/bin)**: This variant balances exploration and exploitation better than the standard `rand/1` strategy by guiding individuals towards the best solution found so far while maintaining diversity using difference vectors.
#2.  **Restarts**: If the population converges (standard deviation of fitness becomes negligible) before the time limit, the algorithm restarts with a fresh random population. This prevents the algorithm from getting stuck in local optima.
#3.  **Elitism**: During restarts, the global best solution found is injected into the new population to ensure the search never regresses.
#4.  **Dithering**: The mutation factor $F$ is randomized for each generation to handle various fitness landscape scales dynamically.
#5.  **Vectorization**: NumPy operations are used for mutation and crossover to maximize computational efficiency, allowing more function evaluations within `max_time`.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Differential Evolution with Restarts 
    and the DE/current-to-best/1/bin strategy.
    """
    start_time = time.time()
    
    # --- Algorithm Configuration ---
    # Population size: Adaptive based on dimension.
    # Scaled to ensure enough diversity for high dims, but fast enough for low dims.
    pop_size = int(10 * np.sqrt(dim))
    pop_size = max(10, pop_size)
    pop_size = min(50, pop_size)
    
    # Crossover Probability
    CR = 0.9

    # Pre-process bounds for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    best_solution = None

    # Helper function to check if time budget is exhausted
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    # Loop continues restarting the population until time runs out
    while not is_time_up():
        
        # 1. Initialize Population
        # Uniform random distribution within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        # This ensures we don't lose the best candidate during a restart.
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if is_time_up(): return best_fitness
            
            # Skip re-evaluation of the injected best solution
            if best_solution is not None and i == 0:
                continue

            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
        
        # 2. Evolution Generations Loop
        # Run until the population converges or time runs out
        while not is_time_up():
            
            # Check for Convergence
            # If the population variance is extremely low, we are likely in a local optimum.
            # Break the inner loop to trigger a restart.
            if np.std(fitness) < 1e-8:
                break
            
            # Dithering: Randomize Mutation Factor F [0.5, 1.0] per generation
            F = 0.5 + 0.5 * np.random.rand()
            
            # Find the best individual in the current population for the mutation strategy
            current_best_idx = np.argmin(fitness)
            x_best = population[current_best_idx]
            
            # --- Vectorized Mutation & Crossover ---
            
            # Generate random indices for difference vectors
            # Note: We allow minor collisions (r1==r2 or r1==i) for computational speed.
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Strategy: DE/current-to-best/1/bin
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This moves every individual towards the best, plus a random difference for diversity.
            mutant = population + F * (x_best - population) + F * (population[r1] - population[r2])
            
            # Binary Crossover
            cross_mask = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one parameter is taken from the mutant (DE constraint)
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            # Create trial population
            trial_pop = np.where(cross_mask, mutant, population)
            
            # Apply Bounds Constraints (Clipping)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection (Evaluation) ---
            for i in range(pop_size):
                if is_time_up(): return best_fitness
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection: Accept if better or equal
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_pop[i].copy()

    return best_fitness

Improve.
The output values and codes for the last generated algorithms are as follows:
1. output value is: 15.230174087252792

 1. algorithm code is:
#The following is a fully operational and self-contained algorithm using a **Success-History based Adaptive Differential Evolution (SHADE)** strategy with **Restarts**.
#
#### Algorithm Description
#To achieve the minimum output value within a limited time, this algorithm implements **SHADE with Linear Population Sizing and Restarts**. This is an advanced variant of Differential Evolution that self-adapts its control parameters ($F$ and $CR$) based on the success of previous generations, making it highly effective for diverse and unknown fitness landscapes.
#
#1.  **Strategy (DE/current-to-pbest/1/bin)**: The mutation strategy moves individuals towards a solution selected from the top $p\%$ best individuals ($p$-best). This balances exploitation (convergence) and exploration better than targeting the single best solution.
#2.  **External Archive**: An archive stores recently replaced inferior solutions. These are used in the mutation step (as the second difference vector) to maintain population diversity and prevent premature convergence.
#3.  **Self-Adaptive Parameters (SHADE)**: Instead of fixed or random parameters, mutation factor $F$ and crossover rate $CR$ are adapted using a historical memory ($M_F, M_{CR}$). Successful parameters that lead to fitness improvements are used to update this memory, guiding the algorithm towards optimal settings automatically.
#4.  **Restart Mechanism**: The algorithm detects stagnation (no improvement in best fitness) or convergence (low population variance). Upon detection, it restarts with a fresh population but carries over the global best solution (Elitism) to ensure monotonic improvement.
#5.  **Adaptive Population**: The population size is heuristically determined based on the dimension ($15\sqrt{D}$) to ensure adequate search density without exceeding the time budget.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Success-History based Adaptive 
    Differential Evolution (SHADE) with Restarts.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracker across all restarts
    global_best_fit = float('inf')
    global_best_sol = None
    
    # Helper to check time budget
    def is_time_left():
        return (time.time() - start_time) < max_time

    # --- Main Restart Loop ---
    # Restarts the population if it converges or stagnates.
    while is_time_left():
        
        # Adaptive Population Size
        # Heuristic: 15 * sqrt(D), clamped to [30, 80] for efficiency
        pop_size = int(15 * np.sqrt(dim))
        pop_size = max(30, min(80, pop_size))
        
        # --- SHADE Memory Initialization ---
        # Memory stores successful F and CR values to guide future generation.
        # H (history size) = 5
        mem_size = 5
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.5)
        k_mem = 0  # Memory index pointer
        
        # Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous restarts
        start_idx = 0
        if global_best_sol is not None:
            pop[0] = global_best_sol
            fitness[0] = global_best_fit
            start_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if not is_time_left(): return global_best_fit
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_fit:
                global_best_fit = val
                global_best_sol = pop[i].copy()
                
        # External Archive for 'current-to-pbest' diversity
        # Stores inferior solutions replaced by better offspring
        archive = []
        
        # Stagnation tracking
        stag_counter = 0
        prev_min_fit = np.min(fitness)
        
        # --- Evolution Loop ---
        while is_time_left():
            # 1. Sort Population
            # Necessary for 'current-to-pbest' to easily pick top p%
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            # 2. Convergence / Stagnation Check
            current_min = fitness[0]
            
            # Check if population variance is zero or no improvement seen
            if np.std(fitness) < 1e-8 or abs(current_min - prev_min_fit) < 1e-8:
                stag_counter += 1
            else:
                stag_counter = 0
                prev_min_fit = current_min
                
            if stag_counter > 30: # Trigger restart if stuck
                break
                
            # 3. Parameter Generation (SHADE Strategy)
            # Pick random memory slot for each individual
            r_idx = np.random.randint(0, mem_size, pop_size)
            mu_f = mem_F[r_idx]
            mu_cr = mem_CR[r_idx]
            
            # Generate CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr_g = np.random.normal(mu_cr, 0.1)
            cr_g = np.clip(cr_g, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1), clipped (0, 1]
            # Cauchy = mu + scale * tan(pi * (rand - 0.5))
            f_g = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            f_g = np.clip(f_g, 0.1, 1.0) 
            
            # 4. Mutation: DE/current-to-pbest/1
            # Select p-best (top 15% or at least 2)
            p_limit = max(2, int(0.15 * pop_size))
            p_best_idxs = np.random.randint(0, p_limit, pop_size)
            x_pbest = pop[p_best_idxs]
            
            # Select r1 (from Population)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            x_r1 = pop[r1_idxs]
            
            # Select r2 (from Population U Archive)
            if len(archive) > 0:
                arc_arr = np.array(archive)
                n_arc = len(archive)
                # Generate random indices covering both pop and archive
                r2_raw_idxs = np.random.randint(0, pop_size + n_arc, pop_size)
                
                # Construct union (concatenation is efficient for these sizes)
                union_pop = np.concatenate((pop, arc_arr), axis=0)
                x_r2 = union_pop[r2_raw_idxs]
            else:
                r2_idxs = np.random.randint(0, pop_size, pop_size)
                x_r2 = pop[r2_idxs]
            
            # Mutation Equation
            # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
            # Note: 'pop' is sorted, so pop[i] is current 'x'
            mutant = pop + f_g[:, None] * (x_pbest - pop) + f_g[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 5. Crossover
            # Binomial crossover
            cross_points = np.random.rand(pop_size, dim) < cr_g[:, None]
            # Ensure at least one parameter comes from mutant (DE constraint)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_points[np.arange(pop_size), j_rand] = True
            
            trial = np.where(cross_points, mutant, pop)
            
            # 6. Evaluation & Selection
            succ_F = []
            succ_CR = []
            diff_fitness = []
            
            for i in range(pop_size):
                if not is_time_left(): return global_best_fit
                
                f_trial = func(trial[i])
                f_old = fitness[i]
                
                if f_trial <= f_old:
                    # Successful Update
                    
                    # Add parent to archive (if strictly better)
                    if f_trial < f_old:
                        if len(archive) < pop_size:
                            archive.append(pop[i].copy())
                        else:
                            # If archive full, replace random solution
                            rem = np.random.randint(0, pop_size)
                            archive[rem] = pop[i].copy()
                        
                        # Record success for memory update
                        succ_F.append(f_g[i])
                        succ_CR.append(cr_g[i])
                        diff_fitness.append(f_old - f_trial)
                    
                    # Update Population
                    fitness[i] = f_trial
                    pop[i] = trial[i]
                    
                    # Update Global Best
                    if f_trial < global_best_fit:
                        global_best_fit = f_trial
                        global_best_sol = trial[i].copy()
            
            # 7. Update SHADE Memory
            if len(succ_F) > 0:
                succ_F = np.array(succ_F)
                succ_CR = np.array(succ_CR)
                weights = np.array(diff_fitness)
                total_w = np.sum(weights)
                
                if total_w > 0:
                    weights /= total_w
                    
                    # Weighted Lehmer Mean for F
                    mean_F = np.sum(weights * (succ_F ** 2)) / np.sum(weights * succ_F)
                    
                    # Weighted Mean for CR
                    mean_CR = np.sum(weights * succ_CR)
                    
                    # Update Memory at current pointer
                    mem_F[k_mem] = 0.5 * mem_F[k_mem] + 0.5 * mean_F
                    mem_CR[k_mem] = 0.5 * mem_CR[k_mem] + 0.5 * mean_CR
                    
                    k_mem = (k_mem + 1) % mem_size

    return global_best_fit


2. output value is: 8.627763194400856

 2. algorithm code is:
#The following is a fully operational and self-contained algorithm using an improved **Differential Evolution (DE)** strategy optimized for limited-time execution.
#
##### Algorithm Description
#To achieve a lower minimum value (better fitness) than previous iterations, this algorithm implements **DE/current-to-pbest/1/bin with Adaptive Restarts**. 
#1.  **Strategy (DE/current-to-pbest/1/bin)**: Unlike `current-to-best`, which can converge prematurely to local optima, `current-to-pbest` guides the population towards a random individual selected from the top $p\%$ best candidates. This balances convergence speed with population diversity.
#2.  **Adaptive Population Sizing**: The population size is heuristically derived from the dimension ($20\sqrt{D}$) to ensure sufficient search density for higher dimensions while remaining fast for lower ones.
#3.  **Randomized Control Parameters**: Instead of fixed values, mutation factor $F$ and crossover rate $CR$ are randomized per individual for every generation. This mimics self-adaptive behaviors (like SHADE), allowing the algorithm to dynamically handle different fitness landscape features.
#4.  **Stagnation Detection & Restarts**: The algorithm monitors fitness improvements. If the population standard deviation drops (convergence) or the best fitness doesn't improve for a set number of generations (stagnation), it restarts.
#5.  **Elitism**: On restart, the global best solution is carried over to the new population to prevent regression.
#
##### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Differential Evolution with the 
    DE/current-to-pbest/1/bin strategy and adaptive restarts.
    """
    # Initialize timing
    start_time = time.time()
    
    # Precompute bounds for vectorization efficiency
    bounds_arr = np.array(bounds)
    min_bound = bounds_arr[:, 0]
    max_bound = bounds_arr[:, 1]
    diff_bound = max_bound - min_bound
    
    # --- Adaptive Configuration ---
    # Population size scaling: slightly larger than standard to support p-best diversity
    # Heuristic: pop_size = 20 * sqrt(dim), clamped between 20 and 100 for efficiency
    pop_size = int(20 * np.sqrt(dim))
    pop_size = max(20, min(100, pop_size))
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    best_solution = None
    
    # Helper to enforce time limit
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    while not is_time_up():
        
        # 1. Initialize Population
        # Uniform random initialization within bounds
        population = min_bound + np.random.rand(pop_size, dim) * diff_bound
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        start_idx = 0
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
            start_idx = 1 # Skip re-evaluating the injected best
        
        # Evaluate Initial Population
        for i in range(start_idx, pop_size):
            if is_time_up(): return best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
                
        # 2. Differential Evolution Loop
        stagnation_count = 0
        prev_min_fit = np.min(fitness)
        
        while not is_time_up():
            # Sort population by fitness (required for current-to-pbest strategy)
            # This moves the best individuals to the top of the arrays
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            current_min_fit = fitness[0]
            
            # --- Convergence & Stagnation Checks ---
            # If population variance is negligible, we have converged -> Restart
            if np.std(fitness) < 1e-8:
                break
            
            # If best fitness hasn't improved significantly, increment stagnation counter
            if abs(current_min_fit - prev_min_fit) < 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_min_fit = current_min_fit
            
            # Trigger restart if stuck in a local optimum for too long
            if stagnation_count > 30:
                break
            
            # --- Strategy: DE/current-to-pbest/1/bin ---
            # 'p-best': Select target from the top p% (p_val) individuals
            # This maintains diversity better than targeting solely the single best
            p_val = max(2, int(0.15 * pop_size))
            
            # Vectorized Indices Generation
            # 1. p-best indices: random integer from [0, p_val) for each individual
            idxs_pbest = np.random.randint(0, p_val, pop_size)
            x_pbest = population[idxs_pbest]
            
            # 2. r1, r2 indices: random integers from whole population
            idxs_r1 = np.random.randint(0, pop_size, pop_size)
            idxs_r2 = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[idxs_r1]
            x_r2 = population[idxs_r2]
            
            # Randomized Control Parameters (Dithering)
            # F ~ Uniform(0.5, 0.95), CR ~ Uniform(0.8, 1.0)
            # Randomizing per individual helps escape local basins
            F = np.random.uniform(0.5, 0.95, (pop_size, 1))
            CR = np.random.uniform(0.8, 1.0, (pop_size, 1))
            
            # Mutation Equation: v = x + F(x_pbest - x) + F(x_r1 - x_r2)
            # Note: 'population' is the current x because we sorted it
            mutant = population + F * (x_pbest - population) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(pop_size, dim) < CR
            # Constraint: Ensure at least one dimension is inherited from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            # Construct Trial Population
            trial = np.where(cross_mask, mutant, population)
            
            # Bounds Constraint (Clipping)
            trial = np.clip(trial, min_bound, max_bound)
            
            # --- Selection (Evaluation) ---
            # Evaluate trial vectors and apply greedy selection
            for i in range(pop_size):
                if is_time_up(): return best_fitness
                
                f_trial = func(trial[i])
                
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial[i]
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial[i].copy()
                        
    return best_fitness


3. output value is: 44.9935846844537

 3. algorithm code is:
#The following is a fully operational and self-contained algorithm using an improved **Differential Evolution (DE)** strategy.
#
#### Algorithm Description
#To minimize the output value within the limited time, this algorithm implements **Differential Evolution with Restarts and Dithering**.
#1.  **Strategy (DE/current-to-best/1/bin)**: This variant balances exploration and exploitation better than the standard `rand/1` strategy by guiding individuals towards the best solution found so far while maintaining diversity using difference vectors.
#2.  **Restarts**: If the population converges (standard deviation of fitness becomes negligible) before the time limit, the algorithm restarts with a fresh random population. This prevents the algorithm from getting stuck in local optima.
#3.  **Elitism**: During restarts, the global best solution found is injected into the new population to ensure the search never regresses.
#4.  **Dithering**: The mutation factor $F$ is randomized for each generation to handle various fitness landscape scales dynamically.
#5.  **Vectorization**: NumPy operations are used for mutation and crossover to maximize computational efficiency, allowing more function evaluations within `max_time`.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes the function 'func' using Differential Evolution with Restarts 
    and the DE/current-to-best/1/bin strategy.
    """
    start_time = time.time()
    
    # --- Algorithm Configuration ---
    # Population size: Adaptive based on dimension.
    # Scaled to ensure enough diversity for high dims, but fast enough for low dims.
    pop_size = int(10 * np.sqrt(dim))
    pop_size = max(10, pop_size)
    pop_size = min(50, pop_size)
    
    # Crossover Probability
    CR = 0.9

    # Pre-process bounds for vectorized operations
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Track the global best solution found across all restarts
    best_fitness = float('inf')
    best_solution = None

    # Helper function to check if time budget is exhausted
    def is_time_up():
        return (time.time() - start_time) >= max_time

    # --- Main Optimization Loop (Restarts) ---
    # Loop continues restarting the population until time runs out
    while not is_time_up():
        
        # 1. Initialize Population
        # Uniform random distribution within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject the best solution found so far into the new population
        # This ensures we don't lose the best candidate during a restart.
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if is_time_up(): return best_fitness
            
            # Skip re-evaluation of the injected best solution
            if best_solution is not None and i == 0:
                continue

            val = func(population[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
        
        # 2. Evolution Generations Loop
        # Run until the population converges or time runs out
        while not is_time_up():
            
            # Check for Convergence
            # If the population variance is extremely low, we are likely in a local optimum.
            # Break the inner loop to trigger a restart.
            if np.std(fitness) < 1e-8:
                break
            
            # Dithering: Randomize Mutation Factor F [0.5, 1.0] per generation
            F = 0.5 + 0.5 * np.random.rand()
            
            # Find the best individual in the current population for the mutation strategy
            current_best_idx = np.argmin(fitness)
            x_best = population[current_best_idx]
            
            # --- Vectorized Mutation & Crossover ---
            
            # Generate random indices for difference vectors
            # Note: We allow minor collisions (r1==r2 or r1==i) for computational speed.
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Strategy: DE/current-to-best/1/bin
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This moves every individual towards the best, plus a random difference for diversity.
            mutant = population + F * (x_best - population) + F * (population[r1] - population[r2])
            
            # Binary Crossover
            cross_mask = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one parameter is taken from the mutant (DE constraint)
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            # Create trial population
            trial_pop = np.where(cross_mask, mutant, population)
            
            # Apply Bounds Constraints (Clipping)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection (Evaluation) ---
            for i in range(pop_size):
                if is_time_up(): return best_fitness
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection: Accept if better or equal
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_pop[i].copy()

    return best_fitness