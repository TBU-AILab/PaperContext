#The following algorithm improves upon the previous best results by implementing **Differential Evolution with Opposition-Based Learning (OBL) Initialization and a Restart Mechanism**.
#
#### Rationale for Improvements:
#1.  **Opposition-Based Learning (OBL) Initialization**: Instead of starting with purely random points, the algorithm generates a random population and its "opposite" population ($x' = min + max - x$) within the bounds. It then selects the best individuals from this combined pool. This increases the probability of starting in a promising basin of attraction without extra function evaluations later on.
#2.  **Restart Mechanism**: Standard DE can sometimes stagnate (converge to a local optimum) before the time limit expires. This algorithm monitors population variance. If diversity drops below a threshold, it triggers a restart (re-initializing the population) while injecting the global best solution found so far (Elitism). This allows the algorithm to use the remaining time to explore other regions.
#3.  **Reflection-based Boundary Handling**: Simple clipping ($x = \text{clip}(x, min, max)$) can bias the search towards the edges. This algorithm uses reflection ($x_{new} = 2 \cdot min - x_{old}$), which bounces values back into the search space, preserving the statistical distribution of the population.
#4.  **Robust Parameters**: It uses the robust **DE/rand/1/bin** strategy which performed best in previous tests (result 15.94), but enhances it with per-generation dithered mutation factors ($F$) to avoid parameter-specific stagnation.
#
#### Algorithm Code:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) with 
    Opposition-Based Learning (OBL) initialization and a restart mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Capped to ensure speed while maintaining diversity.
    # A size of ~10*D is standard, but we cap it between 20 and 50 
    # to ensure many generations can run within the time limit.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
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
        # Clip to ensure numerical stability within bounds
        p2 = np.clip(p2, min_b, max_b)
        return np.vstack((p1, p2))

    # --- Main Optimization Loop (Handles Restarts) ---
    while True:
        if check_time(): return global_best_fitness
        
        # 1. Initialize Population (OBL)
        # We generate 2 * pop_size candidates and pick the best pop_size
        candidates = generate_initial_pool(pop_size)
        cand_fitness = np.full(len(candidates), float('inf'))
        
        # Evaluate initial pool
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
        # Note: If time ran out during loop, we just use what we have so far
        sorted_idx = np.argsort(cand_fitness[:valid_count])
        # Take top pop_size (or fewer if we didn't finish)
        current_size = min(pop_size, len(sorted_idx))
        population = candidates[sorted_idx[:current_size]]
        fitness = cand_fitness[sorted_idx[:current_size]]
        
        # If we have a previous global best (from a restart), inject it (Elitism)
        # We replace the worst individual of the current OBL population
        if global_best_sol is not None and len(population) > 0:
            population[-1] = global_best_sol
            fitness[-1] = global_best_fitness

        # --- Evolution Loop ---
        while True:
            if check_time(): return global_best_fitness
            
            # 2. Check Stagnation/Convergence
            # If population variance is negligible, restart to explore elsewhere
            if len(fitness) > 1:
                fit_std = np.std(fitness)
                fit_range = np.max(fitness) - np.min(fitness)
                if fit_std < 1e-6 or fit_range < 1e-6:
                    break # Break inner loop -> Trigger Restart
            
            # 3. Dynamic Parameter Dithering
            # Randomize F slightly per generation to prevent search stagnation
            # F in [0.5, 0.8]
            F = 0.5 + 0.3 * np.random.rand()
            
            # 4. Iterate Population
            for i in range(len(population)):
                if check_time(): return global_best_fitness
                
                # Mutation: DE/rand/1/bin
                # Select r1, r2, r3 distinct from i
                idxs = [x for x in range(len(population)) if x != i]
                # Need at least 3 other individuals
                if len(idxs) < 3: break 
                
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                
                x_i = population[i]
                x_r1 = population[r1]
                x_r2 = population[r2]
                x_r3 = population[r3]
                
                # Mutant Vector
                v = x_r1 + F * (x_r2 - x_r3)
                
                # Crossover: Binomial
                mask = np.random.rand(dim) < CR
                # Ensure at least one dimension is changed
                j_rand = np.random.randint(dim)
                mask[j_rand] = True
                
                u = np.where(mask, v, x_i)
                
                # Boundary Handling: Reflection
                # If a particle hits a wall, it bounces back rather than sticking (clipping)
                # This preserves population distribution better
                
                # Reflect Lower Bounds
                under_b = u < min_b
                if np.any(under_b):
                    u[under_b] = 2 * min_b[under_b] - u[under_b]
                
                # Reflect Upper Bounds
                over_b = u > max_b
                if np.any(over_b):
                    u[over_b] = 2 * max_b[over_b] - u[over_b]
                
                # Final safeguard clip (in case of double reflection out of bounds)
                u = np.clip(u, min_b, max_b)
                
                # Selection (Greedy)
                f_u = func(u)
                
                if f_u <= fitness[i]:
                    population[i] = u
                    fitness[i] = f_u
                    
                    if f_u < global_best_fitness:
                        global_best_fitness = f_u
                        global_best_sol = u.copy()

    return global_best_fitness
