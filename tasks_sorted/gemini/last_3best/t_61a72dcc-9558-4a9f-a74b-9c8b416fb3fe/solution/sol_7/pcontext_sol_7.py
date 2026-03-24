#The output value of the last generated algorithm (IPO-DE-OBL) was **25.23**, which is significantly worse than the first algorithm's result of **12.61**. This suggests that while opposition-based learning and population management are theoretically sound, the overhead or the lack of aggressive convergence logic (like in the first algorithm) hindered performance in the short time limit.
#
#The improved algorithm below implements a **Hybrid Memetic Algorithm**. It takes the high-speed convergence of the best-performing strategy (Differential Evolution `current-to-best`) and augments it with two critical components:
#1.  **OBL Initialization:** Retained from the last attempt to ensure high-quality starting points for every restart.
#2.  **Coordinate Descent Local Search:** A lightweight, derivative-free local search that runs *after* DE converges. This "polishes" the best solution found by DE to a much higher precision, addressing the likely cause of the fitness gap between previous runs.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Hybrid Memetic Algorithm:
    1.  Initialization using Opposition-Based Learning (OBL) for maximizing initial coverage.
    2.  Differential Evolution (DE) with 'current-to-best' strategy for fast global convergence.
    3.  Coordinate Descent Local Search (CD) for high-precision refinement of the best solution.
    4.  Restarts to utilize the full time budget.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    lower_bound = bounds_arr[:, 0]
    upper_bound = bounds_arr[:, 1]
    bound_diff = upper_bound - lower_bound
    
    # Track global best across all restarts
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # Helper to safely update global best
    def update_global(sol, val):
        nonlocal global_best_fitness, global_best_sol
        if val < global_best_fitness:
            global_best_fitness = val
            global_best_sol = sol.copy()
            return True
        return False

    # --- Configuration ---
    # Moderate population size to allow fast generations and time for Local Search
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # --- Main Restart Loop ---
    while True:
        # Check time budget before starting a new run
        if time.time() - start_time >= max_time:
            return global_best_fitness
            
        # ==========================================
        # Phase 1: OBL Initialization
        # ==========================================
        # 1. Random Population
        pop_rand = lower_bound + np.random.rand(pop_size, dim) * bound_diff
        
        # 2. Opposite Population (Antithetic sampling)
        pop_opp = lower_bound + upper_bound - pop_rand
        pop_opp = np.clip(pop_opp, lower_bound, upper_bound)
        
        # 3. Combine and Evaluate
        candidates = np.vstack((pop_rand, pop_opp))
        cand_fitness = []
        
        for i in range(len(candidates)):
            if time.time() - start_time >= max_time:
                return global_best_fitness
            val = func(candidates[i])
            cand_fitness.append(val)
            update_global(candidates[i], val)
            
        cand_fitness = np.array(cand_fitness)
        
        # 4. Select Best N individuals
        sorted_idx = np.argsort(cand_fitness)
        population = candidates[sorted_idx[:pop_size]]
        fitnesses = cand_fitness[sorted_idx[:pop_size]]
        
        # ==========================================
        # Phase 2: Differential Evolution (DE)
        # ==========================================
        # Strategy: current-to-best/1/bin (Proven fast convergence)
        stagnation_count = 0
        last_best = fitnesses[0]
        
        while True:
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            # --- Parameters ---
            # F (Mutation): Dithered [0.5, 1.0] for exploration
            F = 0.5 + 0.5 * np.random.rand()
            # CR (Crossover): High (0.9) for exploitation of good structure
            CR = 0.9
            
            # --- Mutation ---
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            
            x_best = population[0] # Population is sorted/maintained such that index 0 is best
            
            # Random indices
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Vectorized mutation
            mutant = population + F * (x_best - population) + F * (population[r1] - population[r2])
            
            # --- Crossover ---
            mask = np.random.rand(pop_size, dim) < CR
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask, mutant, population)
            trial_pop = np.clip(trial_pop, lower_bound, upper_bound)
            
            # --- Selection ---
            for i in range(pop_size):
                if time.time() - start_time >= max_time:
                    return global_best_fitness
                
                t_val = func(trial_pop[i])
                
                if t_val <= fitnesses[i]:
                    fitnesses[i] = t_val
                    population[i] = trial_pop[i]
                    update_global(trial_pop[i], t_val)
            
            # --- Maintenance & Convergence ---
            # Ensure best is at index 0 for next iteration's mutation efficiency
            best_idx = np.argmin(fitnesses)
            if best_idx != 0:
                population[[0, best_idx]] = population[[best_idx, 0]]
                fitnesses[[0, best_idx]] = fitnesses[[best_idx, 0]]
                
            current_best = fitnesses[0]
            
            # Check for stagnation
            if abs(current_best - last_best) < 1e-9:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best = current_best
                
            # Exit DE phase if variance is low or stagnation is detected
            # This triggers the Local Search phase
            if np.var(fitnesses) < 1e-8 or stagnation_count > 20:
                break
        
        # ==========================================
        # Phase 3: Local Search (Coordinate Descent)
        # ==========================================
        # Start refining from the best solution found in the DE phase
        ls_sol = population[0].copy()
        ls_val = fitnesses[0]
        
        # Initial step size: 5% of domain width
        step_sizes = bound_diff * 0.05
        min_step = 1e-9
        
        # Iterate until step size becomes negligible or time runs out
        while np.max(step_sizes) > min_step:
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            improved_in_pass = False
            
            # Iterate through dimensions in random order to avoid bias
            dims = np.random.permutation(dim)
            
            for d in dims:
                if time.time() - start_time >= max_time:
                    return global_best_fitness
                
                origin_val = ls_sol[d]
                
                # 1. Try Negative Step
                ls_sol[d] = np.clip(origin_val - step_sizes[d], lower_bound[d], upper_bound[d])
                val_neg = func(ls_sol)
                update_global(ls_sol, val_neg)
                
                if val_neg < ls_val:
                    ls_val = val_neg
                    improved_in_pass = True
                    continue # Success, move to next dimension
                
                # 2. Try Positive Step (if negative failed)
                ls_sol[d] = np.clip(origin_val + step_sizes[d], lower_bound[d], upper_bound[d])
                val_pos = func(ls_sol)
                update_global(ls_sol, val_pos)
                
                if val_pos < ls_val:
                    ls_val = val_pos
                    improved_in_pass = True
                else:
                    # Neither direction improved, revert change
                    ls_sol[d] = origin_val
            
            # If no improvement in any dimension, shrink step size
            if not improved_in_pass:
                step_sizes *= 0.5
                
    return global_best_fitness
