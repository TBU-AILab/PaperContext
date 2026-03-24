#The following algorithm implements **Self-Adaptive Differential Evolution (JADE-variant) with LHS Initialization and Coordinate Descent Local Search**.
#
#### Algorithm Description
#
#To improve upon the previous best solution (score ~2.05), this algorithm introduces a **Hybrid Memetic Strategy**:
#
#1.  **LHS (Latin Hypercube Sampling) Initialization**: Ensures the initial population covers the search space more evenly than uniform random sampling, providing a better starting point for the evolutionary process.
#2.  **Adaptive DE (JADE-based)**:
#    *   **Strategy**: Uses `DE/current-to-pbest/1/bin` with an external archive. This balances convergence (by following the top $p\%$ individuals) with diversity (using archived difference vectors).
#    *   **Parameter Adaptation**: Instead of fixed or purely random parameters, $F$ (Mutation) and $CR$ (Crossover) are sampled from Cauchy and Normal distributions, respectively. The means of these distributions ($\mu_F, \mu_{CR}$) are adapted based on the success of previous generations (Lehmer mean), allowing the algorithm to "learn" the optimal step sizes and crossover probabilities for the specific function landscape.
#3.  **Coordinate Descent Local Search**:
#    *   DE is excellent at finding the basin of attraction but can be slow to refine the solution to high precision (oscillating around the optimum).
#    *   **Polishing Step**: When the population stagnates or is about to restart, a fast **Coordinate Descent** (axis-parallel search) is applied to the best solution found. This greedily optimizes each dimension with shrinking step sizes to extract maximum precision from the identified basin.
#4.  **Restart Mechanism**: If the population converges or stagnates, the algorithm restarts with a fresh LHS population (keeping the global best via elitism) to explore other potential basins.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes 'func' using Self-Adaptive Differential Evolution (JADE)
    combined with Latin Hypercube Sampling and Coordinate Descent Local Search.
    """
    start_time = time.time()
    
    # --- Helper: Check Time ---
    def check_time():
        return (time.time() - start_time) >= max_time

    # --- Pre-computation ---
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    best_fitness = float('inf')
    best_solution = None
    
    # --- Configuration ---
    # Population size: 20 * sqrt(dim) is a balanced heuristic for DE
    pop_size = int(20 * np.sqrt(dim))
    pop_size = max(30, min(100, pop_size))
    
    # Archive size to maintain diversity
    archive_size = int(pop_size * 1.5)
    
    # --- Local Search: Coordinate Descent ---
    # Polishes the best solution found to extract maximum precision
    # Triggered before a restart or at the end of a run
    def local_search(start_sol, start_fit):
        nonlocal best_fitness, best_solution
        
        current_sol = start_sol.copy()
        current_fit = start_fit
        
        # Start step size at 1% of domain width
        step_size = diff_b * 0.01
        min_step = 1e-9
        
        # Max iterations to prevent excessive time usage
        # This budget allows for a quick polish without blocking restarts
        ls_iter = 0
        max_ls_iter = 50 * dim 

        while np.max(step_size) > min_step and ls_iter < max_ls_iter:
            if check_time(): return
            
            improved_any = False
            # Randomize dimension order to avoid directional bias
            for d_idx in np.random.permutation(dim):
                if check_time(): return
                
                ls_iter += 1
                
                # Try positive move
                temp_val = current_sol[d_idx]
                step = step_size[d_idx]
                
                current_sol[d_idx] = np.clip(temp_val + step, min_b[d_idx], max_b[d_idx])
                f_new = func(current_sol)
                
                if f_new < current_fit:
                    current_fit = f_new
                    improved_any = True
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = current_sol.copy()
                else:
                    # Try negative move
                    current_sol[d_idx] = np.clip(temp_val - step, min_b[d_idx], max_b[d_idx])
                    f_new = func(current_sol)
                    
                    if f_new < current_fit:
                        current_fit = f_new
                        improved_any = True
                        if f_new < best_fitness:
                            best_fitness = f_new
                            best_solution = current_sol.copy()
                    else:
                        # Revert if neither improved
                        current_sol[d_idx] = temp_val
            
            # If no dimension improved, reduce step sizes to fine-tune
            if not improved_any:
                step_size *= 0.5
            
    # --- Main Optimization Loop (Restarts) ---
    while not check_time():
        
        # 1. LHS Initialization
        # Generate Latin Hypercube Samples
        lhs_raw = np.random.rand(pop_size, dim)
        for d in range(dim):
            # Stratify samples in each dimension
            lhs_raw[:, d] = (np.random.permutation(pop_size) + lhs_raw[:, d]) / pop_size
        population = min_b + lhs_raw * diff_b
        
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous runs
        start_eval = 0
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
            start_eval = 1
            
        # Evaluate Initial Population
        for i in range(start_eval, pop_size):
            if check_time(): return best_fitness
            val = func(population[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
                
        # 2. Setup Run State
        archive = np.empty((archive_size, dim))
        n_arc = 0
        
        # Adaptive Parameters (JADE initialization)
        mu_F = 0.5
        mu_CR = 0.5
        c_adapt = 0.1 # Learning rate
        
        stag_counter = 0
        last_best = np.min(fitness)
        
        # 3. Evolution Loop
        while not check_time():
            # Sort population (Best at index 0)
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Check Stagnation
            curr_best = fitness[0]
            if abs(curr_best - last_best) < 1e-10:
                stag_counter += 1
            else:
                stag_counter = 0
                last_best = curr_best
                
            # Restart Trigger
            # If stagnated for 30 gens or variance collapsed
            if stag_counter > 30 or np.std(fitness) < 1e-9:
                # Perform Local Search on the best solution of this run
                local_search(population[0], fitness[0])
                break
            
            # --- Generate Parameters (JADE) ---
            # F ~ Cauchy(mu_F, 0.1)
            rand_u = np.random.rand(pop_size)
            F = mu_F + 0.1 * np.tan(np.pi * (rand_u - 0.5))
            F = np.where(F <= 0, 0.1, F) # Truncate negative to 0.1
            F = np.clip(F, 0.0, 1.0)     # Clip > 1.0
            
            # CR ~ Normal(mu_CR, 0.1)
            CR = np.random.normal(mu_CR, 0.1, pop_size)
            CR = np.clip(CR, 0.0, 1.0)
            
            # --- Mutation: DE/current-to-pbest/1 ---
            # Select p-best vectors (top 15%)
            p_size = max(2, int(0.15 * pop_size))
            p_idxs = np.random.randint(0, p_size, pop_size)
            x_pbest = population[p_idxs]
            
            # Select r1 vectors (random from population)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[r1_idxs]
            
            # Select r2 vectors (random from Union(Population, Archive))
            if n_arc > 0:
                total_cnt = pop_size + n_arc
                r2_raw = np.random.randint(0, total_cnt, pop_size)
                
                x_r2 = np.empty_like(population)
                
                # Indices < pop_size map to population
                mask_pop = r2_raw < pop_size
                if np.any(mask_pop):
                    x_r2[mask_pop] = population[r2_raw[mask_pop]]
                
                # Indices >= pop_size map to archive
                mask_arc = ~mask_pop
                if np.any(mask_arc):
                    arc_map_idxs = r2_raw[mask_arc] - pop_size
                    x_r2[mask_arc] = archive[arc_map_idxs]
            else:
                r2_idxs = np.random.randint(0, pop_size, pop_size)
                x_r2 = population[r2_idxs]
            
            # Compute Mutant: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            mutant = population + F[:, None] * (x_pbest - population) + F[:, None] * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- Crossover ---
            rand_cross = np.random.rand(pop_size, dim)
            mask_cross = rand_cross < CR[:, None]
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            mask_cross = mask_cross | j_rand_mask
            
            trial = np.where(mask_cross, mutant, population)
            
            # --- Selection and Adaptation ---
            succ_F = []
            succ_CR = []
            
            for i in range(pop_size):
                if check_time(): return best_fitness
                
                f_tri = func(trial[i])
                f_old = fitness[i]
                
                if f_tri <= f_old:
                    # Successful Update
                    
                    # Add replaced parent to archive if strictly better
                    if f_tri < f_old:
                        if n_arc < archive_size:
                            archive[n_arc] = population[i].copy()
                            n_arc += 1
                        else:
                            # Random replacement if full
                            archive[np.random.randint(0, archive_size)] = population[i].copy()
                        
                        # Store parameters for adaptation
                        succ_F.append(F[i])
                        succ_CR.append(CR[i])
                        
                    population[i] = trial[i]
                    fitness[i] = f_tri
                    
                    if f_tri < best_fitness:
                        best_fitness = f_tri
                        best_solution = trial[i].copy()
            
            # --- Parameter Adaptation (Lehmer Mean) ---
            if len(succ_F) > 0:
                sF = np.array(succ_F)
                sCR = np.array(succ_CR)
                
                # Weighted Lehmer mean for F (favors higher successful F)
                mean_pow = np.sum(sF**2) / (np.sum(sF) + 1e-15)
                mu_F = (1 - c_adapt) * mu_F + c_adapt * mean_pow
                
                # Arithmetic mean for CR
                mean_ari = np.mean(sCR)
                mu_CR = (1 - c_adapt) * mu_CR + c_adapt * mean_ari
                
    return best_fitness
