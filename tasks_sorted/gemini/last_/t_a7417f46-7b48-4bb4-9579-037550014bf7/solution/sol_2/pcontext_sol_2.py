#Here is an improved optimization algorithm.
#
#It implements a **Self-Adaptive Differential Evolution (jDE)** algorithm combined with a **Coordinate Descent Polishing** strategy.
#
#### Key Improvements:
#1.  **Self-Adaptation (jDE)**: Instead of fixed parameters, each individual adapts its own mutation factor ($F$) and crossover rate ($CR$). This allows the algorithm to automatically tune itself to the specific function landscape.
#2.  **Exploration Strategy**: Uses `DE/rand/1/bin` instead of the greedy `current-to-best`. While slightly slower to converge, it is much more robust against getting trapped in local optima (which likely caused the score of ~25 previously).
#3.  **Coordinate Descent Polishing**: When the population stagnates, the algorithm switches to a fast local search (Coordinate Descent) on the best candidate. This "polishes" the solution to high precision, drilling down into the bottom of the basin where DE might struggle.
#4.  **Restart Mechanism**: If the population converges or stalls, the algorithm restarts with new random candidates (keeping the global best), ensuring efficient use of time to explore multiple basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restart Self-Adaptive Differential Evolution (jDE)
    hybridized with Coordinate Descent Polishing.
    """
    # 1. Setup and Time Management
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def check_timeout():
        return datetime.now() - start_time >= time_limit

    # 2. Pre-process Bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # 3. Algorithm Parameters
    # Population size: 10*dim is standard, but capped for speed to allow restarts
    pop_size = int(10 * dim)
    pop_size = np.clip(pop_size, 20, 60)
    
    # Global Best Tracking
    best_val = float('inf')
    best_vec = None
    
    # --- Helper: Coordinate Descent Polishing ---
    # Used to refine the solution when DE stagnates
    def polish(candidate_x, candidate_val):
        nonlocal best_val, best_vec
        
        current_x = candidate_x.copy()
        current_val = candidate_val
        
        # Grid of step sizes (relative to bounds)
        # We start coarse and go very fine
        step_sizes = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7]
        
        for step_scale in step_sizes:
            if check_timeout(): break
            
            # Hill climbing loop at current resolution
            improved = True
            while improved:
                if check_timeout(): break
                improved = False
                
                # Randomize dimension order to avoid bias
                dims = np.random.permutation(dim)
                
                for d in dims:
                    if check_timeout(): break
                    
                    step = step_scale * diff_b[d]
                    
                    # Try positive move
                    temp_x = current_x.copy()
                    temp_x[d] += step
                    # Simple clamp
                    if temp_x[d] > max_b[d]: temp_x[d] = max_b[d]
                    
                    val = func(temp_x)
                    if val < current_val:
                        current_val = val
                        current_x = temp_x
                        improved = True
                        # Update global immediately
                        if val < best_val:
                            best_val = val
                            best_vec = current_x.copy()
                    else:
                        # Try negative move
                        temp_x = current_x.copy()
                        temp_x[d] -= step
                        if temp_x[d] < min_b[d]: temp_x[d] = min_b[d]
                        
                        val = func(temp_x)
                        if val < current_val:
                            current_val = val
                            current_x = temp_x
                            improved = True
                            if val < best_val:
                                best_val = val
                                best_vec = current_x.copy()
        
        return current_val

    # --- Main Restart Loop ---
    while not check_timeout():
        # Initialize Population (Random)
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_timeout(): return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_vec = pop[i].copy()
                
        # jDE Initialization: F and CR stored per individual
        # F ~ 0.5, CR ~ 0.9 initially
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        stall_count = 0
        local_best = np.min(fitness)
        
        # --- Inner Evolution Loop ---
        while True:
            if check_timeout(): return best_val
            
            # 1. jDE Parameter Adaptation
            # With prob 0.1, generate new F/CR, otherwise keep old
            # We generate potential new parameters first
            mask_F = np.random.rand(pop_size) < 0.1
            mask_CR = np.random.rand(pop_size) < 0.1
            
            F_new = F.copy()
            CR_new = CR.copy()
            
            # F in [0.1, 1.0], CR in [0.0, 1.0]
            F_new[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
            CR_new[mask_CR] = np.random.rand(np.sum(mask_CR))
            
            # 2. Mutation: DE/rand/1/bin
            # Indices: r1 != r2 != r3 != i
            idxs = np.arange(pop_size)
            
            # Efficient vectorized index shifting
            s1 = np.random.randint(1, pop_size)
            s2 = np.random.randint(1, pop_size)
            s3 = np.random.randint(1, pop_size)
            
            # Ensure uniqueness
            while s2 == s1: s2 = np.random.randint(1, pop_size)
            while s3 == s1 or s3 == s2: s3 = np.random.randint(1, pop_size)
            
            r1 = pop[np.roll(idxs, s1)]
            r2 = pop[np.roll(idxs, s2)]
            r3 = pop[np.roll(idxs, s3)]
            
            # Mutation Vector: v = r1 + F * (r2 - r3)
            # Use F_new for calculation
            v = r1 + F_new[:, None] * (r2 - r3)
            
            # 3. Crossover
            rand_vals = np.random.rand(pop_size, dim)
            cross_mask = rand_vals < CR_new[:, None]
            
            # Force at least one dimension from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, v, pop)
            
            # Boundary Constraint (Clamp)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 4. Selection
            # Evaluate trials
            pop_improved = False
            for i in range(pop_size):
                if check_timeout(): return best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # jDE: Update parameters only if offspring successful
                    F[i] = F_new[i]
                    CR[i] = CR_new[i]
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_vec = trial_pop[i].copy()
                        pop_improved = True
            
            # 5. Stagnation Check
            current_min = np.min(fitness)
            if current_min < local_best:
                local_best = current_min
                stall_count = 0
            else:
                stall_count += 1
            
            fit_std = np.std(fitness)
            
            # Restart if:
            # - Population has converged (low variance)
            # - No improvement for many generations
            if fit_std < 1e-6 or stall_count > 40:
                # Before restart, Polish the best individual in this population
                # to squeeze out remaining precision
                best_idx = np.argmin(fitness)
                # Only polish if it's reasonably good to save time
                if fitness[best_idx] < best_val * 1.5 + 1.0: 
                    polish(pop[best_idx], fitness[best_idx])
                
                # Break inner loop to trigger restart
                break

    return best_val
