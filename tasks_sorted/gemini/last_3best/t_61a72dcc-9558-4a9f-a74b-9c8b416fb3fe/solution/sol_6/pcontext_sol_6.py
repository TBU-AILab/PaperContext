#Here is the improved algorithm. It implements **IPO-DE-OBL (Increasing Population Size Differential Evolution with Opposition-Based Learning)**.
#
#**Key Improvements:**
#1.  **IPO (Increasing Population Size) Strategy:** Instead of a fixed population size, the algorithm starts with a small population to ensure fast convergence for simple landscapes. On each restart, the population size doubles. This allows the algorithm to switch from "fast exploitation" to "thorough exploration" as time permits, maximizing the probability of finding the global optimum.
#2.  **Opposition-Based Learning (OBL) Initialization:** At the start of each run, the algorithm generates a random population *and* its opposite (symmetric within bounds). It evaluates both and keeps the fittest half. This guarantees a better initial coverage of the search space and acts as a powerful antithetic sampling technique.
#3.  **Greedy Mutation (`current-to-best/1`):** This strategy (identified as the best performer in previous iterations) is retained. It drives the population quickly towards the best known solution.
#4.  **Robust Restart Logic:** The algorithm detects convergence through fitness variance and stagnation counters, triggering a restart with a larger population to escape local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using IPO-DE-OBL:
    Differential Evolution with Increasing Population Size,
    Opposition-Based Learning Initialization, and Restarts.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    lower_bound = bounds_arr[:, 0]
    upper_bound = bounds_arr[:, 1]
    bound_diff = upper_bound - lower_bound
    
    global_best_fitness = float('inf')
    
    # --- IPO Configuration ---
    # Start with a small population for fast initial convergence.
    # Heuristic: 10 * dim, but clamped to [10, 40] for the first run.
    current_pop_size = int(np.clip(10 * dim, 10, 40))
    
    # Cap for population growth to prevent iterations becoming too slow
    max_pop_size = 200 

    # --- Main Loop (Restarts) ---
    while True:
        # Check time budget before starting a new run
        if time.time() - start_time >= max_time:
            return global_best_fitness

        # --- 1. OBL Initialization ---
        # Generate Random Population
        pop_rand = lower_bound + np.random.rand(current_pop_size, dim) * bound_diff
        
        # Generate Opposite Population (Antithetic sampling)
        # O_i = Lower + Upper - X_i
        pop_opp = lower_bound + upper_bound - pop_rand
        
        # Combine and ensure bounds
        combined_pop = np.vstack((pop_rand, pop_opp))
        combined_pop = np.clip(combined_pop, lower_bound, upper_bound)
        
        # Evaluate Combined Population (2 * pop_size)
        total_candidates = 2 * current_pop_size
        fitnesses = np.full(total_candidates, float('inf'))
        
        for i in range(total_candidates):
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            val = func(combined_pop[i])
            fitnesses[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val
        
        # Select best N individuals to start the evolution
        # This provides a high-quality initial population
        sorted_indices = np.argsort(fitnesses)
        population = combined_pop[sorted_indices[:current_pop_size]]
        pop_fitness = fitnesses[sorted_indices[:current_pop_size]]
        
        # --- 2. Evolution Loop ---
        last_best_fit = pop_fitness[0]
        stagnation_count = 0
        
        while True:
            # Check time budget per generation
            if time.time() - start_time >= max_time:
                return global_best_fitness
            
            # --- Parameters ---
            # F (Mutation Factor): Dithered between 0.5 and 1.0. 
            # High F encourages exploration which complements the greedy mutation strategy.
            F = 0.5 + 0.5 * np.random.rand()
            # CR (Crossover Rate): High rate (0.9) works well for current-to-best.
            CR = 0.9
            
            # --- Mutation: current-to-best/1 ---
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            
            # Identify Best
            # Since we maintain local population updates, we find the current best index
            best_idx = np.argmin(pop_fitness)
            x_best = population[best_idx]
            
            # Random Indices r1, r2
            r1 = np.random.randint(0, current_pop_size, current_pop_size)
            r2 = np.random.randint(0, current_pop_size, current_pop_size)
            
            # Compute Mutant Vector (Vectorized)
            # F is scalar, broadcasting applies
            mutant = population + F * (x_best - population) + F * (population[r1] - population[r2])
            
            # --- Crossover: Binomial ---
            rand_vals = np.random.rand(current_pop_size, dim)
            cross_mask = rand_vals < CR
            
            # Force at least one dimension from mutant (DE standard)
            j_rand = np.random.randint(0, dim, current_pop_size)
            cross_mask[np.arange(current_pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # --- Boundary Handling ---
            trial_pop = np.clip(trial_pop, lower_bound, upper_bound)
            
            # --- Selection ---
            # Evaluate trials and replace parents if better
            for i in range(current_pop_size):
                if time.time() - start_time >= max_time:
                    return global_best_fitness
                
                val = func(trial_pop[i])
                
                if val <= pop_fitness[i]:
                    pop_fitness[i] = val
                    population[i] = trial_pop[i]
                    
                    if val < global_best_fitness:
                        global_best_fitness = val
            
            # --- Convergence & Restart Logic ---
            current_best = np.min(pop_fitness)
            
            # Check for Stagnation (Best value not changing)
            if np.abs(current_best - last_best_fit) < 1e-9:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = current_best
            
            # Check for Convergence (Population variance is low)
            pop_std = np.std(pop_fitness)
            
            # Trigger Restart if:
            # 1. Population has collapsed (std < 1e-8)
            # 2. Search is stuck (stagnation > 30 gens)
            if pop_std < 1e-8 or stagnation_count > 30:
                break
        
        # --- IPO: Increase Population for next restart ---
        # Double population to search more thoroughly in the next run
        current_pop_size = int(current_pop_size * 2)
        if current_pop_size > max_pop_size:
            current_pop_size = max_pop_size

    return global_best_fitness
