#The output values indicate that **Algorithm 2 (DE + OBL + Restart)** achieved the best performance (10.53), significantly outperforming the standard restart strategy (Result 1), SHADE (Result 3), and jDE (Result 4).
#
#The success of Algorithm 2 is attributed to:
#1.  **OBL Initialization**: Starting with a higher quality pool by checking opposite points.
#2.  **Reflection Boundaries**: Preserving population diversity at the edges better than clipping.
#3.  **Restarts**: preventing stagnation.
#
#However, a **Full Restart** (as used in Algorithm 2) can be destructive, as it discards the entire population covariance structure except for one elite.
#
#To improve upon this, I propose a **Partial Restart Strategy** integrated into the DE+OBL framework. Instead of discarding the whole population upon stagnation, we retain the top 20% (elites) and replenish the remaining 80% using **OBL-based generation**. This preserves the best genetic material found so far while injecting high-quality diversity to escape local optima.
#
#**Algorithm Highlights:**
#1.  **Partial Restart**: Retains elites, refills the rest with OBL candidates.
#2.  **OBL Initialization/Refill**: Generates $N$ random points and their $N$ opposites, selecting the best $N$.
#3.  **Reflective Boundary Handling**: Mirrors particles back into the search space.
#4.  **Robust DE Parameters**: Uses DE/rand/1/bin with dithered $F$ for stable exploration.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) with 
    Opposition-Based Learning (OBL) initialization and a Partial Restart mechanism.
    
    This improves upon standard restarts by retaining the top portion of the 
    population (Elitism) and refilling the rest with OBL-generated candidates,
    combining exploitation of known good regions with exploration of new ones.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Capped to ensure iteration speed while maintaining diversity.
    # 10*dim is standard, clipped to [20, 50] for efficiency.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # DE Parameters
    CR = 0.9  # Crossover probability
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Track Global Best
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # --- Helper: Time Check ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: OBL Generation ---
    # Generates 'count' individuals using Opposition-Based Learning
    # Returns: (selected_population, fitness_values)
    def generate_obl_pool(count):
        nonlocal global_best_fitness, global_best_sol
        
        # 1. Random candidates
        p1 = min_b + np.random.rand(count, dim) * diff_b
        
        # 2. Opposite candidates (Opposition-Based Learning)
        # x_opp = min + max - x
        p2 = min_b + max_b - p1
        p2 = np.clip(p2, min_b, max_b) # Keep within bounds
        
        # Combine
        candidates = np.vstack((p1, p2))
        cand_fitness = []
        valid_candidates = []
        
        # Evaluate
        for i in range(len(candidates)):
            if check_time(): break
            val = func(candidates[i])
            cand_fitness.append(val)
            valid_candidates.append(candidates[i])
            
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_sol = candidates[i].copy()
                
        if not cand_fitness:
            return np.array([]), np.array([])

        cand_fitness = np.array(cand_fitness)
        valid_candidates = np.array(valid_candidates)
        
        # Select best 'count' from the 2*count pool
        sorted_idx = np.argsort(cand_fitness)
        best_n = min(count, len(sorted_idx))
        
        return valid_candidates[sorted_idx[:best_n]], cand_fitness[sorted_idx[:best_n]]

    # --- Initialization ---
    population, fitness = generate_obl_pool(pop_size)
    if len(population) == 0: return global_best_fitness # Timeout check

    # Restart monitoring
    stagnation_counter = 0
    best_in_run = np.min(fitness)
    
    # --- Main Optimization Loop ---
    while True:
        if check_time(): return global_best_fitness
        
        # 1. Check Restart Conditions
        do_restart = False
        
        # Metric A: Stagnation (No improvement for N generations)
        current_best = np.min(fitness)
        if current_best >= best_in_run - 1e-8:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_in_run = current_best
            
        if stagnation_counter > 30:
            do_restart = True
            
        # Metric B: Convergence (Low Variance)
        if not do_restart and len(fitness) > 1:
            fit_std = np.std(fitness)
            fit_range = np.max(fitness) - np.min(fitness)
            if fit_std < 1e-6 or fit_range < 1e-6:
                do_restart = True
                
        if do_restart:
            # --- Partial Restart Strategy ---
            # Keep top 20% (Elitism), replace rest with fresh OBL candidates.
            # This is better than full restart as it preserves the best modes found.
            
            # Sort current population
            sort_idx = np.argsort(fitness)
            elite_count = max(1, int(pop_size * 0.2))
            
            elites = population[sort_idx[:elite_count]]
            elite_fit = fitness[sort_idx[:elite_count]]
            
            # Generate rest using OBL
            needed = pop_size - elite_count
            new_pop, new_fit = generate_obl_pool(needed)
            
            if len(new_pop) > 0:
                population = np.vstack((elites, new_pop))
                fitness = np.concatenate((elite_fit, new_fit))
            
            # Reset counters
            stagnation_counter = 0
            best_in_run = np.min(fitness) if len(fitness) > 0 else float('inf')
            
            if check_time(): return global_best_fitness
            continue # Start fresh generation immediately

        # 2. Dynamic Parameters
        # Dither F in [0.5, 0.8] to prevent parameter-based stagnation
        F = 0.5 + 0.3 * np.random.rand()
        
        # 3. Evolution Cycle (DE/rand/1/bin)
        # Using arrays for 'new' ensures we don't mix generations in-place during the loop
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(len(population)):
            if check_time(): return global_best_fitness
            
            # Mutation indices selection
            idxs = [x for x in range(len(population)) if x != i]
            if len(idxs) < 3: break
            
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
            
            # Mutant Vector
            v = x_r1 + F * (x_r2 - x_r3)
            
            # Crossover: Binomial
            mask = np.random.rand(dim) < CR
            j_rand = np.random.randint(dim)
            mask[j_rand] = True # Ensure at least one dimension changes
            
            u = np.where(mask, v, population[i])
            
            # Boundary Handling: Reflection
            # Reflects particles back into bounds to preserve distribution density at edges
            under_b = u < min_b
            if np.any(under_b):
                u[under_b] = 2 * min_b[under_b] - u[under_b]
            over_b = u > max_b
            if np.any(over_b):
                u[over_b] = 2 * max_b[over_b] - u[over_b]
            
            # Safety clip (in case of extreme reflection)
            u = np.clip(u, min_b, max_b)
            
            # Selection (Greedy)
            f_u = func(u)
            
            if f_u <= fitness[i]:
                new_population[i] = u
                new_fitness[i] = f_u
                
                if f_u < global_best_fitness:
                    global_best_fitness = f_u
                    global_best_sol = u.copy()
        
        population = new_population
        fitness = new_fitness

    return global_best_fitness
