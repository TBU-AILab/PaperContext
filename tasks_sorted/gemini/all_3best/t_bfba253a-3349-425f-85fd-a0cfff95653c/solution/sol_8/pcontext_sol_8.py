#The following algorithm improves upon the best previous result (~10.53) by incorporating an **External Archive** and a **'current-to-pbest' mutation strategy** into the successful OBL-Restart framework.
#
#**Rationale for Improvements:**
#1.  **Current-to-pBest Mutation with Archive**: The previous best algorithm used `DE/rand/1`, which is robust but slow to converge. `DE/current-to-best` (used in other attempts) converges too fast and gets stuck. **`DE/current-to-pbest/1`** balances these by steering towards a random top-tier solution ($p$-best). Crucially, it uses an **External Archive** of recently replaced solutions. This archive maintains population diversity by allowing the difference vector to draw from "older" positions, preventing the search step size from vanishing too quickly. This is the core engine of state-of-the-art algorithms like L-SHADE.
#2.  **OBL & Restart**: We retain the **Opposition-Based Learning** initialization and **Restart** mechanism (keeping elites) from the best-performing algorithm, as these proved essential for exploring the landscape and escaping local optima within the time limit.
#3.  **Parameter Distributions**: Instead of complex history-based adaptation (which failed in previous SHADE attempts likely due to time overhead), we use robust probability distributions (Cauchy for $F$, Normal for $CR$) to generate diverse control parameters for each individual, ensuring the algorithm doesn't get stuck with a poor parameter set.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution with:
    1. Opposition-Based Learning (OBL) for initialization and restarts.
    2. 'current-to-pbest' mutation strategy with an External Archive (inspired by JADE/SHADE).
    3. Restart mechanism with Elitism to escape local optima.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Balance between diversity and speed.
    # 10*dim is standard, clipped to [20, 50] to ensure reasonable generation count.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # External Archive (stores displaced individuals to maintain diversity)
    archive = []
    max_archive_size = int(pop_size * 2.0)
    
    # Strategy Parameters
    p_best_rate = 0.11  # Top 11% used for 'p-best' selection
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global tracking
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # --- Helpers ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    def get_obl_pool(count):
        """Generates 'count' optimized individuals using OBL."""
        nonlocal global_best_fitness, global_best_sol
        
        # 1. Random Generation
        p_rand = min_b + np.random.rand(count, dim) * diff_b
        
        # 2. Opposite Generation (min + max - x)
        p_opp = min_b + max_b - p_rand
        p_opp = np.clip(p_opp, min_b, max_b)
        
        # 3. Combine and Evaluate
        candidates = np.vstack((p_rand, p_opp))
        cand_fit = []
        valid_cand = []
        
        for i in range(len(candidates)):
            if check_time(): break
            val = func(candidates[i])
            cand_fit.append(val)
            valid_cand.append(candidates[i])
            
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_sol = candidates[i].copy()
                
        if not cand_fit: return np.array([]), np.array([])
        
        cand_fit = np.array(cand_fit)
        valid_cand = np.array(valid_cand)
        
        # Select best 'count'
        idx = np.argsort(cand_fit)
        n = min(count, len(idx))
        return valid_cand[idx[:n]], cand_fit[idx[:n]]

    # --- Initialization ---
    population, fitness = get_obl_pool(pop_size)
    if len(population) == 0: return global_best_fitness

    # Run Statistics for Restart
    stagnation_counter = 0
    best_in_run = np.min(fitness)
    
    # --- Main Loop ---
    while True:
        if check_time(): return global_best_fitness
        
        # 1. Restart Logic
        do_restart = False
        
        # Check Stagnation (Fitness improvement)
        curr_min = np.min(fitness)
        if curr_min < best_in_run - 1e-8:
            best_in_run = curr_min
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # Trigger: No improvement for 35 gens OR Low Variance (Convergence)
        if stagnation_counter > 35:
            do_restart = True
        elif len(fitness) > 1 and np.std(fitness) < 1e-6:
            do_restart = True
            
        if do_restart:
            # Elitism: Keep top 10%
            elite_count = max(1, int(pop_size * 0.1))
            sort_idx = np.argsort(fitness)
            elites = population[sort_idx[:elite_count]]
            elite_fit = fitness[sort_idx[:elite_count]]
            
            # Refill remainder with OBL
            needed = pop_size - elite_count
            new_pop, new_fit = get_obl_pool(needed)
            
            if len(new_pop) > 0:
                population = np.vstack((elites, new_pop))
                fitness = np.concatenate((elite_fit, new_fit))
            
            # Reset Archive and counters for new basin search
            archive = []
            stagnation_counter = 0
            best_in_run = np.min(fitness)
            continue

        # 2. Preparation for Generation
        # Sort population to easily pick p-best
        sort_idx = np.argsort(fitness)
        sorted_pop = population[sort_idx]
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Union Pool (Population + Archive) for mutation
        if len(archive) > 0:
            pool = np.vstack((population, np.array(archive)))
        else:
            pool = population
            
        # 3. Evolution Cycle
        for i in range(pop_size):
            if check_time(): return global_best_fitness
            
            x_i = population[i]
            
            # --- Parameter Generation ---
            # F: Cauchy(0.5, 0.1) -> Allows occasional large jumps (exploration)
            F = 0.5 + 0.1 * np.random.standard_cauchy()
            while F <= 0: # Regenerate if negative
                F = 0.5 + 0.1 * np.random.standard_cauchy()
            F = min(1.0, F)
            
            # CR: Normal(0.9, 0.05) -> Favor high crossover for component dependence
            CR = np.random.normal(0.9, 0.05)
            CR = np.clip(CR, 0.0, 1.0)
            
            # --- Mutation: current-to-pbest/1 ---
            # V = X_i + F*(X_pbest - X_i) + F*(X_r1 - X_r2)
            
            # Select X_pbest (Top p%)
            p_idx = np.random.randint(0, max(1, int(pop_size * p_best_rate)))
            x_pbest = sorted_pop[p_idx]
            
            # Select X_r1 (from Population, != i)
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select X_r2 (from Pool (Pop+Archive), != i, != r1)
            # Note: Pool indices 0..pop_size-1 correspond to population
            r2 = -1
            while True:
                r2 = np.random.randint(0, len(pool))
                if r2 < pop_size:
                    if r2 != i and r2 != r1: break
                else:
                    break # Archive members are safe
            x_r2 = pool[r2]
            
            # Compute Mutant
            mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            # --- Crossover ---
            mask = np.random.rand(dim) < CR
            j_rand = np.random.randint(dim)
            mask[j_rand] = True # Ensure at least one dimension changes
            u = np.where(mask, mutant, x_i)
            
            # --- Boundary Handling: Reflection ---
            under = u < min_b
            if np.any(under): u[under] = 2*min_b[under] - u[under]
            over = u > max_b
            if np.any(over): u[over] = 2*max_b[over] - u[over]
            u = np.clip(u, min_b, max_b)
            
            # --- Selection ---
            f_u = func(u)
            
            if f_u <= fitness[i]:
                # Update Population
                new_population[i] = u
                new_fitness[i] = f_u
                
                # Update Archive (Store replaced parent)
                archive.append(x_i.copy())
                if len(archive) > max_archive_size:
                    archive.pop(np.random.randint(0, len(archive)))
                
                # Update Global Best
                if f_u < global_best_fitness:
                    global_best_fitness = f_u
                    global_best_sol = u.copy()
        
        population = new_population
        fitness = new_fitness

    return global_best_fitness
