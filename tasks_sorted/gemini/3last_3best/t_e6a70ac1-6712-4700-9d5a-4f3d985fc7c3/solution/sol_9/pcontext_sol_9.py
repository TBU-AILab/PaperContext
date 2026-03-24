#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 8.627763194400856
#   algorithm code is: [DE with Adaptive Restarts]
#2. output value is: 11.61933730018302
#   algorithm code is: [L-SHADE with jDE]
#3. output value is: 15.230174087252792
#   algorithm code is: [SHADE with Restarts]
#
#The best performance was achieved by **DE with Adaptive Restarts**, which utilized a `current-to-pbest` strategy with randomized control parameters ($F, CR$) and aggressive restarts. The SHADE variants likely suffered from Python's interpreter overhead or converged prematurely due to learning low mutation rates.
#
#The following algorithm, **LHS-DE with Archive and Restarts**, improves upon the winner by:
#1.  **Adding an External Archive**: This significantly enhances the diversity of the difference vectors ($x_{r1} - x_{r2}$) in the `current-to-pbest` strategy without requiring additional function evaluations. This helps escape local optima.
#2.  **Latin Hypercube Initialization (LHS)**: Instead of uniform random, LHS is used to seed the population, ensuring a more even coverage of the search space for the initial generation of each restart.
#3.  **Optimized Vectorization**: The archive selection logic is implemented using vectorized indexing to minimize overhead, avoiding costly array concatenations inside the loop.
#4.  **Robust Parameters**: It retains the high-energy parameter distributions ($F \in [0.5, 0.95]$) that proved successful in the previous iteration.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Optimized Differential Evolution (DE) utilizing Latin Hypercube Sampling,
    an External Archive, and Adaptive Restarts to minimize the function value.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Heuristic based on dimension (25*sqrt(D))
    # Clamped to [30, 100] to balance search density with iteration speed
    pop_size = int(25 * np.sqrt(dim))
    pop_size = max(30, min(100, pop_size))
    
    # Archive size: Stores inferior solutions to maintain diversity
    archive_size = int(pop_size * 1.5)
    
    # Precompute bounds for speed
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking across restarts
    best_fitness = float('inf')
    best_solution = None
    
    # Helper: Check remaining time
    def check_time():
        return (time.time() - start_time) >= max_time

    # Helper: Latin Hypercube Sampling (LHS)
    # Generates a more fastidiously distributed initial population than random
    def init_lhs(n, d):
        rng = np.random.rand(n, d)
        for j in range(d):
            # Permute each dimension to ensure one sample per interval
            rng[:, j] = (np.random.permutation(n) + rng[:, j]) / n
        return min_b + rng * diff_b

    # --- Main Restart Loop ---
    while not check_time():
        
        # 1. Initialize Population using LHS
        population = init_lhs(pop_size, dim)
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous runs to ensure monotonicity
        start_eval_idx = 0
        if best_solution is not None:
            population[0] = best_solution
            fitness[0] = best_fitness
            start_eval_idx = 1
            
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if check_time(): return best_fitness
            val = func(population[i])
            fitness[i] = val
            if val < best_fitness:
                best_fitness = val
                best_solution = population[i].copy()
                
        # 2. Setup Archive and Stagnation Tracking
        archive = np.empty((archive_size, dim))
        n_arc = 0  # Current number of items in archive
        
        stag_count = 0
        last_best_fit = np.min(fitness)
        
        # --- Evolutionary Loop ---
        while not check_time():
            # Sort population (Best at index 0) - Required for p-best selection
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Check Stagnation
            current_best_fit = fitness[0]
            if abs(current_best_fit - last_best_fit) < 1e-10:
                stag_count += 1
            else:
                stag_count = 0
                last_best_fit = current_best_fit
            
            # Restart if stagnated (40 gens) or converged (zero variance)
            if stag_count > 40 or np.std(fitness) < 1e-9:
                break
            
            # --- Mutation Strategy: DE/current-to-pbest/1/bin with Archive ---
            
            # 1. Parameter Dithering
            # Randomized F [0.5, 0.95] and CR [0.8, 1.0] per individual.
            # These "High Energy" parameters prevent premature convergence.
            F = np.random.uniform(0.5, 0.95, (pop_size, 1))
            CR = np.random.uniform(0.8, 1.0, (pop_size, 1))
            
            # 2. Select p-best vectors (from top 15%)
            p_limit = max(2, int(0.15 * pop_size))
            p_idxs = np.random.randint(0, p_limit, pop_size)
            x_pbest = population[p_idxs]
            
            # 3. Select r1 vectors (random from population)
            r1_idxs = np.random.randint(0, pop_size, pop_size)
            x_r1 = population[r1_idxs]
            
            # 4. Select r2 vectors (random from Union(Population, Archive))
            # Vectorized selection to avoid concatenating large arrays
            if n_arc > 0:
                total_choices = pop_size + n_arc
                r2_raw = np.random.randint(0, total_choices, pop_size)
                
                x_r2 = np.empty((pop_size, dim))
                
                # Indices < pop_size take from population
                mask_pop = r2_raw < pop_size
                if np.any(mask_pop):
                    x_r2[mask_pop] = population[r2_raw[mask_pop]]
                
                # Indices >= pop_size take from archive
                mask_arc = ~mask_pop
                if np.any(mask_arc):
                    arc_idxs = r2_raw[mask_arc] - pop_size
                    x_r2[mask_arc] = archive[arc_idxs]
            else:
                r2_idxs = np.random.randint(0, pop_size, pop_size)
                x_r2 = population[r2_idxs]
            
            # Compute Mutant: v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
            mutant = population + F * (x_pbest - population) + F * (x_r1 - x_r2)
            
            # --- Crossover ---
            rand_cross = np.random.rand(pop_size, dim)
            cross_mask = rand_cross < CR
            
            # Ensure at least one dimension is inherited from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            j_rand_mask = np.zeros((pop_size, dim), dtype=bool)
            j_rand_mask[np.arange(pop_size), j_rand] = True
            cross_mask = cross_mask | j_rand_mask
            
            trial_pop = np.where(cross_mask, mutant, population)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection ---
            for i in range(pop_size):
                if check_time(): return best_fitness
                
                f_trial = func(trial_pop[i])
                f_old = fitness[i]
                
                if f_trial <= f_old:
                    # Successful Update
                    
                    # Store inferior parent in archive if offspring is strictly better
                    if f_trial < f_old:
                        if n_arc < archive_size:
                            archive[n_arc] = population[i].copy()
                            n_arc += 1
                        else:
                            # Random replacement if full
                            replace_idx = np.random.randint(0, archive_size)
                            archive[replace_idx] = population[i].copy()
                            
                    population[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    # Update Global Best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_pop[i].copy()
                        
    return best_fitness
