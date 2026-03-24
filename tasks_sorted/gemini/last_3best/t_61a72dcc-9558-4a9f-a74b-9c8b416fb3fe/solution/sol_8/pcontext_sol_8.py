#The previous best result (12.61) was achieved by a Differential Evolution (DE) algorithm with a "current-to-best" mutation strategy and random restarts. The subsequent attempts to add complex local search (Coordinate Descent) or self-adaptation (jDE/OBL) degraded performance, likely due to the overhead or inefficient use of the limited evaluation budget.
#
#The following improved algorithm builds upon the successful "current-to-best" DE framework but introduces a **"Soft Restart" mechanism with Elitism and Gaussian Perturbation**.
#
#**Key Improvements:**
#1.  **Soft Restart vs. Hard Restart:** Instead of discarding the entire population upon convergence (which risks losing the best basin found so far), this algorithm retains the global best solution.
#2.  **Multi-Scale Initialization:** Upon restarting, the new population is composed of three tiers:
#    *   **The Global Best:** Preserved to ensure non-regression.
#    *   **Gaussian Clouds:** A portion of the population is initialized as small Gaussian perturbations around the global best. This acts as an implicit, high-dimensional local search to refine the solution (addressing the precision issue).
#    *   **Random Individuals:** The rest of the population is randomized to explore the search space for completely new basins.
#3.  **Aggressive Convergence:** Uses the proven `current-to-best/1/bin` strategy with high crossover (CR=0.9) to rapidly collapse onto local optima, relying on the restart mechanism to escape.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Restarting Differential Evolution with Soft-Restarts.
    
    Features:
    - Strategy: DE current-to-best/1/bin (Greedy convergence).
    - Restarts: Triggered by population stagnation or low variance.
    - Soft Restart: Retains global best and seeds new population with 
      multi-scale Gaussian perturbations around it for local refinement,
      combined with global random exploration.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    lower_bound = bounds_arr[:, 0]
    upper_bound = bounds_arr[:, 1]
    bound_diff = upper_bound - lower_bound
    
    # --- Configuration ---
    # Population size: Capped to balance speed and diversity.
    # Scaled with sqrt(dim) to handle higher dimensions efficiently.
    # Example: dim=10 -> pop~30, dim=100 -> pop~100
    pop_size = int(np.clip(10 * np.sqrt(dim), 30, 100))
    
    # Trackers for the global best solution
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # Initialize First Population (Random)
    population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
    fitnesses = np.full(pop_size, float('inf'))
    
    # Evaluate initial population
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return global_best_fitness
        val = func(population[i])
        fitnesses[i] = val
        if val < global_best_fitness:
            global_best_fitness = val
            global_best_sol = population[i].copy()
            
    # --- Main Loop (Restarts) ---
    while True:
        # Check time budget
        if time.time() - start_time >= max_time:
            return global_best_fitness
        
        # --- Evolution Phase ---
        # Run DE until the population stagnates or converges
        stagnation_counter = 0
        last_gen_best = global_best_fitness
        
        # Inner Generation Loop
        while True:
            # Check time budget per generation
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            # --- DE Strategy: current-to-best/1/bin ---
            # 1. Sort population to put best at index 0 (Elitism logic)
            sorted_indices = np.argsort(fitnesses)
            population = population[sorted_indices]
            fitnesses = fitnesses[sorted_indices]
            
            x_best = population[0] # Current best in population
            
            # Update global best if current best is better
            if fitnesses[0] < global_best_fitness:
                global_best_fitness = fitnesses[0]
                global_best_sol = x_best.copy()
            
            # 2. Convergence & Stagnation Check
            current_best = fitnesses[0]
            if abs(current_best - last_gen_best) < 1e-9:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_gen_best = current_best
                
            pop_std = np.std(fitnesses)
            
            # Trigger Restart if:
            # - Population has collapsed (low variance)
            # - No improvement for 20 generations
            if pop_std < 1e-6 or stagnation_counter > 20:
                break
            
            # 3. Parameters
            # F: Dithered [0.5, 1.0] - Encourages exploration around difference vector
            F = 0.5 + 0.5 * np.random.rand(pop_size, 1)
            # CR: 0.9 - High crossover preserves good parameter combinations
            CR = 0.9
            
            # 4. Mutation & Crossover (Vectorized)
            # Indices for difference vectors
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Mutation: V = X + F*(X_best - X) + F*(X_r1 - X_r2)
            # This "current-to-best" strategy pulls the population aggressively towards the optimum.
            mutant = population + F * (x_best - population) + F * (population[r1] - population[r2])
            
            # Crossover (Binomial)
            mask = np.random.rand(pop_size, dim) < CR
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask, mutant, population)
            
            # Boundary Constraint (Clip)
            trial_pop = np.clip(trial_pop, lower_bound, upper_bound)
            
            # 5. Selection
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return global_best_fitness
                
                t_val = func(trial_pop[i])
                
                if t_val <= fitnesses[i]:
                    fitnesses[i] = t_val
                    population[i] = trial_pop[i]
                    if t_val < global_best_fitness:
                        global_best_fitness = t_val
                        global_best_sol = trial_pop[i].copy()
                        
        # --- Restart Phase (Soft Restart) ---
        # Instead of random restart, we mix exploration and exploitation.
        
        # 1. Base: Random population (Exploration)
        population = lower_bound + np.random.rand(pop_size, dim) * bound_diff
        
        # 2. Inject Global Best (Elitism)
        if global_best_sol is not None:
            population[0] = global_best_sol
            
            # 3. Inject Gaussian Clouds (Local Refinement)
            # Create mutants around the best solution to refine precision 
            # without explicit local search overhead.
            
            # Cloud 1: Very tight precision (Scale: 0.01%) - 10% of pop
            n_fine = int(0.1 * pop_size)
            for k in range(1, n_fine + 1):
                if k >= pop_size: break
                noise = np.random.randn(dim) * (bound_diff * 1e-4)
                population[k] = np.clip(global_best_sol + noise, lower_bound, upper_bound)
                
            # Cloud 2: Moderate local search (Scale: 1%) - 10% of pop
            n_med = int(0.1 * pop_size)
            for k in range(n_fine + 1, n_fine + n_med + 1):
                if k >= pop_size: break
                noise = np.random.randn(dim) * (bound_diff * 1e-2)
                population[k] = np.clip(global_best_sol + noise, lower_bound, upper_bound)
                
        # Evaluate new population (Skip index 0 as it's already known)
        fitnesses[0] = global_best_fitness
        for i in range(1, pop_size):
            if time.time() - start_time >= max_time:
                return global_best_fitness
            val = func(population[i])
            fitnesses[i] = val
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_sol = population[i].copy()

    return global_best_fitness
