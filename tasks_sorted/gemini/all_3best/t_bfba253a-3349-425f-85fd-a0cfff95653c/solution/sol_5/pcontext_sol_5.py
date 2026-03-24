#The following algorithm improves upon the previous best result (10.53) by implementing **Differential Evolution with Opposition-Based Learning (OBL), Restarts, and Gaussian Local Search**.
#
##### Rationale for Improvements:
#1.  **Opposition-Based Learning (OBL) Initialization**: Retained from the best-performing algorithm. It generates a pool of random solutions and their "opposites" ($min + max - x$), selecting the best half. This significantly improves the quality of the initial basins of attraction.
#2.  **Restart Mechanism with Stagnation Detection**: To prevent the algorithm from wasting time in local optima, it triggers a restart if the population variance drops below a threshold **or** if the best fitness hasn't improved for a set number of generations. It preserves the Global Best (Elitism) to seed the next run.
#3.  **Gaussian Local Search (Exploitation)**: The primary addition. Standard DE mutation (rand/1) is excellent for exploration but can be slow to refine the final decimal places of the optimum. This algorithm adds an "opportunistic" local search: whenever a new **Global Best** is found, it samples a few neighbors using a Gaussian distribution scaled to the current population spread. This allows for rapid gradient-like descent in promising regions without the cost of gradient calculation.
#4.  **Robust Parameters**: Uses the proven **DE/rand/1/bin** strategy with dithered mutation factors and reflection-based boundary handling, which maintains population diversity better than clipping.
#
##### Algorithm Code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) with 
    Opposition-Based Learning (OBL) initialization, Gaussian Local Search,
    and a Restart mechanism with Elitism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Capped to ensure iteration speed while maintaining diversity.
    # 10*dim is standard, clipped to [20, 60] to fit time constraints.
    pop_size = int(np.clip(10 * dim, 20, 60))
    
    # DE Parameters
    CR = 0.9  # Crossover probability
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track Global Best across restarts
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # --- Helper: Time Check ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: OBL Initialization ---
    def generate_initial_pool(size):
        # 1. Random half
        p1 = min_b + np.random.rand(size, dim) * diff_b
        # 2. Opposite half (Opposition-Based Learning)
        # x_opp = min + max - x
        p2 = min_b + max_b - p1
        # Clip ensures opposite points stay within bounds
        p2 = np.clip(p2, min_b, max_b)
        return np.vstack((p1, p2))

    # --- Main Optimization Loop (Handles Restarts) ---
    while True:
        if check_time(): return global_best_fitness
        
        # 1. Initialize Population (OBL)
        # Generate 2 * pop_size candidates and pick the best pop_size
        candidates = generate_initial_pool(pop_size)
        cand_fitness = np.full(len(candidates), float('inf'))
        
        valid_count = 0
        for i in range(len(candidates)):
            if check_time(): return global_best_fitness
            val = func(candidates[i])
            cand_fitness[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_sol = candidates[i].copy()
            valid_count += 1
            
        # Select best 'pop_size' individuals
        sorted_idx = np.argsort(cand_fitness[:valid_count])
        current_size = min(pop_size, len(sorted_idx))
        population = candidates[sorted_idx[:current_size]]
        fitness = cand_fitness[sorted_idx[:current_size]]
        
        # If we have a previous global best (from restart), inject it (Elitism)
        if global_best_sol is not None and len(population) > 0:
            population[-1] = global_best_sol
            fitness[-1] = global_best_fitness

        # Reset stagnation monitoring for this restart
        stagnation_counter = 0
        best_in_run = np.min(fitness) if len(fitness) > 0 else float('inf')

        # --- Evolution Loop ---
        while True:
            if check_time(): return global_best_fitness
            
            # 2. Check Restart Conditions
            # Condition A: Convergence (Low Variance)
            if len(fitness) > 1:
                fit_std = np.std(fitness)
                fit_range = np.max(fitness) - np.min(fitness)
                if fit_std < 1e-6 or fit_range < 1e-6:
                    break # Trigger Restart
            
            # Condition B: Stagnation (No improvement for N generations)
            current_best = np.min(fitness)
            if current_best < best_in_run - 1e-8:
                best_in_run = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter > 40: # Stagnation threshold
                break # Trigger Restart
            
            # 3. Dynamic Parameters
            # Dither F in [0.5, 0.9] to maintain diversity and avoid parameter stagnation
            F = 0.5 + 0.4 * np.random.rand()
            
            # Calculate population scale for Local Search
            # Mean standard deviation across dimensions gives a sense of "search radius"
            pop_std = np.mean(np.std(population, axis=0))
            
            # 4. Iterate Population
            for i in range(len(population)):
                if check_time(): return global_best_fitness
                
                # Mutation: DE/rand/1/bin
                idxs = [x for x in range(len(population)) if x != i]
                if len(idxs) < 3: break 
                
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                
                # Mutant Vector
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: Binomial
                mask = np.random.rand(dim) < CR
                j_rand = np.random.randint(dim)
                mask[j_rand] = True
                
                u = np.where(mask, v, population[i])
                
                # Boundary Handling: Reflection
                # Reflects particles back into bounds to preserve distribution
                under_b = u < min_b
                if np.any(under_b):
                    u[under_b] = 2 * min_b[under_b] - u[under_b]
                over_b = u > max_b
                if np.any(over_b):
                    u[over_b] = 2 * max_b[over_b] - u[over_b]
                u = np.clip(u, min_b, max_b)
                
                # Selection
                f_u = func(u)
                
                if f_u <= fitness[i]:
                    population[i] = u
                    fitness[i] = f_u
                    
                    if f_u < global_best_fitness:
                        global_best_fitness = f_u
                        global_best_sol = u.copy()
                        
                        # --- Local Search (Exploitation) ---
                        # If we found a new global best, attempt to refine it
                        # using a small Gaussian walk proportional to population spread.
                        # This helps refine the solution faster than random mutation.
                        ls_sigma = max(1e-8, pop_std * 0.5)
                        
                        # Try 2 random neighbors
                        for _ in range(2):
                            if check_time(): return global_best_fitness
                            
                            ls_cand = u + np.random.normal(0, ls_sigma, dim)
                            
                            # Boundary (Reflection for LS too)
                            ls_under = ls_cand < min_b
                            if np.any(ls_under): 
                                ls_cand[ls_under] = 2*min_b[ls_under] - ls_cand[ls_under]
                            ls_over = ls_cand > max_b
                            if np.any(ls_over): 
                                ls_cand[ls_over] = 2*max_b[ls_over] - ls_cand[ls_over]
                            ls_cand = np.clip(ls_cand, min_b, max_b)
                            
                            f_ls = func(ls_cand)
                            
                            if f_ls < global_best_fitness:
                                global_best_fitness = f_ls
                                global_best_sol = ls_cand.copy()
                                # Adopt the improved LS solution into the population
                                population[i] = ls_cand
                                fitness[i] = f_ls
                                u = ls_cand # Update center for potential next LS step

    return global_best_fitness
