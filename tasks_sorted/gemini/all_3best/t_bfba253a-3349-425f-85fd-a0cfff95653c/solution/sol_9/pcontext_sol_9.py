#The following algorithm improves upon the best previous result (~10.53) by incorporating a **Coordinate Descent "Polishing" Phase** into the successful OBL-Restart DE framework.
#
#**Rationale for Improvements:**
#1.  **DE/rand/1 + OBL + Restarts**: This combination proved most robust (Result 5, 10.53). More complex strategies like SHADE or Gaussian Local Search failed due to overhead or premature convergence. We retain this core engine.
#2.  **Polishing Phase**: Differential Evolution is excellent at finding the basin of attraction but can be slow to refine the solution to high precision due to its stochastic nature (random mutation). To bridge the gap between ~10.53 and the global minimum, we add a deterministic **Coordinate Descent** step. This is triggered only when the population stagnates, just before a restart. It "polishes" the best solution found in the current basin, ensuring we hit the true local minimum before abandoning the population.
#3.  **Elite Preservation**: Unlike Partial Restarts (which kept 20% and got 45.21), we strictly keep only **one elite** (the global best) during restarts. This ensures maximum diversity for the new OBL-generated population to find different basins.
#4.  **Reflective Boundaries**: Preserved from the best algorithm to maintain population density at the edges.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE/rand/1/bin) with:
    1. Opposition-Based Learning (OBL) for initialization and restarts.
    2. A Polishing Phase (Coordinate Descent) triggered on stagnation to refine the best solution.
    3. A Restart mechanism that preserves the single global best elite.
    4. Reflection-based boundary handling.
    """
    start_time = datetime.now()
    # Use 95% of the time budget to ensure safe return
    time_limit = timedelta(seconds=max_time * 0.95)
    
    # --- Configuration ---
    # Population size: Standard 10*dim, clipped to [20, 50] for speed vs diversity balance.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # DE Parameters
    # CR = 0.9 favors modifying many parameters (good for non-separable functions)
    CR = 0.9 
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_fitness = float('inf')
    global_best_sol = None
    
    # --- Helper: Strict Time Check ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Helper: OBL Generation ---
    # Generates 2*size candidates (Random + Opposite) and selects the best 'size'
    def generate_obl_pool(size):
        nonlocal global_best_fitness, global_best_sol
        
        # 1. Random Generation
        p_rand = min_b + np.random.rand(size, dim) * diff_b
        
        # 2. Opposite Generation (x' = min + max - x)
        p_opp = min_b + max_b - p_rand
        p_opp = np.clip(p_opp, min_b, max_b)
        
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
        
        # Select best 'size' individuals
        cand_fit = np.array(cand_fit)
        valid_cand = np.array(valid_cand)
        idx = np.argsort(cand_fit)
        n = min(size, len(idx))
        return valid_cand[idx[:n]], cand_fit[idx[:n]]

    # --- Helper: Polishing Phase (Coordinate Descent) ---
    # Deterministic local search to refine a solution
    def polish_solution(sol, val):
        nonlocal global_best_fitness, global_best_sol
        
        current_sol = sol.copy()
        current_val = val
        
        # Step sizes relative to the domain range
        # We try 1% then 0.1% of the domain width
        steps = [0.01, 0.001]
        
        for step_scale in steps:
            for d in range(dim):
                if check_time(): return current_sol, current_val
                
                step_size = step_scale * diff_b[d]
                
                # Search in both positive and negative directions for this dimension
                for direction in [1, -1]:
                    temp_sol = current_sol.copy()
                    temp_sol[d] += direction * step_size
                    
                    # Reflection for local search boundaries
                    if temp_sol[d] > max_b[d]:
                        temp_sol[d] = 2*max_b[d] - temp_sol[d]
                    elif temp_sol[d] < min_b[d]:
                        temp_sol[d] = 2*min_b[d] - temp_sol[d]
                    temp_sol[d] = np.clip(temp_sol[d], min_b[d], max_b[d])
                    
                    f_new = func(temp_sol)
                    
                    if f_new < current_val:
                        current_val = f_new
                        current_sol = temp_sol
                        if f_new < global_best_fitness:
                            global_best_fitness = f_new
                            global_best_sol = temp_sol.copy()
                        # If improved, keep the change and try next dimension
                        # (Greedy approach)
                        break 
                        
        return current_sol, current_val

    # --- Initialization ---
    population, fitness = generate_obl_pool(pop_size)
    if len(population) == 0: return global_best_fitness
    
    stagnation_counter = 0
    best_in_run = np.min(fitness)
    
    # --- Main Optimization Loop ---
    while True:
        if check_time(): return global_best_fitness
        
        # 1. Restart Logic
        do_restart = False
        
        # Check Stagnation (No improvement in best fitness)
        curr_min = np.min(fitness)
        if curr_min < best_in_run - 1e-8:
            best_in_run = curr_min
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # Trigger conditions:
        # A. Stagnation counter exceeds limit
        # B. Population variance is extremely low (Convergence)
        if stagnation_counter > 30:
            do_restart = True
        elif len(fitness) > 1:
            if np.std(fitness) < 1e-6 or (np.max(fitness) - np.min(fitness)) < 1e-6:
                do_restart = True
                
        if do_restart:
            # --- Polishing Phase ---
            # Before wiping the population, polish the best individual
            # This extracts the absolute minimum from the current basin
            idx_best = np.argmin(fitness)
            if fitness[idx_best] <= global_best_fitness:
                 _, _ = polish_solution(population[idx_best], fitness[idx_best])
            
            # --- Restart ---
            # Re-initialize population with OBL
            new_pop, new_fit = generate_obl_pool(pop_size)
            if len(new_pop) == 0: return global_best_fitness
            
            population = new_pop
            fitness = new_fit
            
            # Elitism: Inject the single global best into the new population
            if global_best_sol is not None:
                population[0] = global_best_sol.copy()
                fitness[0] = global_best_fitness
                
            # Reset counters
            stagnation_counter = 0
            best_in_run = np.min(fitness)
            continue

        # 2. Differential Evolution Step (DE/rand/1/bin)
        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)
        
        # Dither F per generation: [0.4, 0.9] to allow both exploration and exploitation
        F = 0.4 + 0.5 * np.random.rand()
        
        for i in range(pop_size):
            if check_time(): return global_best_fitness
            
            # Select r1, r2, r3 distinct from i
            idxs = [x for x in range(pop_size) if x != i]
            # Simple rejection sampling for small arrays is fast enough
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
            x_i = population[i]
            
            # Mutant Vector
            v = x_r1 + F * (x_r2 - x_r3)
            
            # Crossover (Binomial)
            mask = np.random.rand(dim) < CR
            j_rand = np.random.randint(dim)
            mask[j_rand] = True # Ensure at least one dimension changes
            u = np.where(mask, v, x_i)
            
            # Boundary Handling: Reflection
            # Bouncing back is better than clipping for maintaining diversity
            under = u < min_b
            if np.any(under): u[under] = 2*min_b[under] - u[under]
            over = u > max_b
            if np.any(over): u[over] = 2*max_b[over] - u[over]
            # Clip safe-guard
            u = np.clip(u, min_b, max_b)
            
            # Selection
            f_u = func(u)
            
            if f_u <= fitness[i]:
                new_population[i] = u
                new_fitness[i] = f_u
                if f_u < global_best_fitness:
                    global_best_fitness = f_u
                    global_best_sol = u.copy()
            else:
                new_population[i] = x_i
                new_fitness[i] = fitness[i]
                
        population = new_population
        fitness = new_fitness

    return global_best_fitness
