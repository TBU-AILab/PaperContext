#The following algorithm implements a **Vectorized Self-Adaptive Differential Evolution (jDE)** with **Linear Population Size Reduction (LPSR)** and a **Gaussian Local Search** mechanism.
#
#**Key Improvements:**
#1.  **Vectorized Operations**: Unlike previous iterations that used Python loops for mutation indices, this version uses fully vectorized NumPy operations. This significantly increases execution speed, allowing for many more generations within the `max_time`.
#2.  **Linear Population Size Reduction (LPSR)**: The population size starts large (~20 * dim) to explore the landscape and linearly reduces to 4. This automatically shifts the focus from exploration to exploitation as time runs out.
#3.  **Gaussian Local Search**: At the end of every generation, the algorithm performs a cheap "Gaussian Polish" on the current best solution. The step size of this search decays over time, allowing for fine-tuning of the global optimum that standard DE mutation might miss.
#4.  **Restart with Elitism**: If the population converges (variance drops), the algorithm restarts but keeps the global best solution, ensuring the search never regresses.
#5.  **Robust Boundary Handling**: Uses a reflection (bounce-back) method to handle constraints, keeping the search within bounds without clumping at the edges.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a function using Vectorized jDE with LPSR and Gaussian Local Search.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population Sizing
    # Start with a healthy size for exploration, cap at 300 for efficiency
    pop_size_init = min(300, max(50, 20 * dim))
    pop_size_min = 4  # Minimum required for DE/rand/1
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None
    
    # --- Helper: Initialize Population ---
    def init_population(n):
        return min_b + np.random.rand(n, dim) * diff_b

    # --- Main Loop (Restart Mechanism) ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Initialize for new Restart
        pop_size = pop_size_init
        population = init_population(pop_size)
        fitness = np.full(pop_size, float('inf'))
        
        # Elitism: Inject global best from previous runs
        start_eval_idx = 0
        if best_sol is not None:
            population[0] = best_sol.copy()
            fitness[0] = best_val
            start_eval_idx = 1
        
        # jDE Parameters (Self-Adaptive)
        # F ~ U(0.1, 0.9), CR ~ U(0.0, 1.0)
        F = np.random.uniform(0.1, 0.9, pop_size)
        CR = np.random.uniform(0.0, 1.0, pop_size)
        
        # Evaluate Initial Population
        for i in range(start_eval_idx, pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            val = func(population[i])
            fitness[i] = val
            
            if val < best_val:
                best_val = val
                best_sol = population[i].copy()
                
        # 2. Evolutionary Cycle
        while True:
            # Check Time
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= max_time:
                return best_val
            
            # --- Linear Population Size Reduction (LPSR) ---
            # Linearly reduce population size based on elapsed time
            progress = elapsed / max_time
            target_pop = int(round(pop_size_init - (pop_size_init - pop_size_min) * progress))
            target_pop = max(pop_size_min, target_pop)
            
            if pop_size > target_pop:
                # Sort by fitness and truncate worst
                idxs = np.argsort(fitness)
                keep = idxs[:target_pop]
                population = population[keep]
                fitness = fitness[keep]
                F = F[keep]
                CR = CR[keep]
                pop_size = target_pop
                
            # --- Convergence Check (Trigger Restart) ---
            # If population has collapsed, restart to search elsewhere
            if np.std(fitness) < 1e-8 or (np.max(fitness) - np.min(fitness)) < 1e-8:
                break
            
            # --- Vectorized Mutation (DE/rand/1) ---
            # Generate indices for mutation: r1 != r2 != r3 != i
            # We use a shift method to ensure distinctness without loops
            idxs = np.arange(pop_size)
            
            r1 = np.random.randint(0, pop_size, pop_size)
            # Collision handling: if r1 == i, shift it
            mask = r1 == idxs
            r1[mask] = (r1[mask] + 1) % pop_size
            
            r2 = np.random.randint(0, pop_size, pop_size)
            mask = (r2 == idxs) | (r2 == r1)
            r2[mask] = (r2[mask] + 2) % pop_size
            
            r3 = np.random.randint(0, pop_size, pop_size)
            mask = (r3 == idxs) | (r3 == r1) | (r3 == r2)
            r3[mask] = (r3[mask] + 3) % pop_size
            
            # Parameter Adaptation (jDE Logic)
            # 10% chance to pick new F or CR
            tau1, tau2 = 0.1, 0.1
            
            # Create trial parameters
            # Use random masks to determine where to update
            rand_f = np.random.rand(pop_size)
            rand_cr = np.random.rand(pop_size)
            
            mask_f = rand_f < tau1
            mask_cr = rand_cr < tau2
            
            F_trial = F.copy()
            CR_trial = CR.copy()
            
            # Update triggered values
            if np.any(mask_f):
                F_trial[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
            if np.any(mask_cr):
                CR_trial[mask_cr] = np.random.rand(np.sum(mask_cr))
            
            # Compute Mutant Vector: v = x_r1 + F * (x_r2 - x_r3)
            # Reshape F for broadcasting across dimensions
            F_broad = F_trial[:, None]
            diffs = population[r2] - population[r3]
            mutant = population[r1] + F_broad * diffs
            
            # --- Bounce-Back Boundary Handling ---
            # Reflect lower bound violations
            viol_l = mutant < min_b
            if np.any(viol_l):
                # Reflection: min + (min - v)
                mutant[viol_l] = 2 * min_b[np.where(viol_l)[1]] - mutant[viol_l]
                # Fix double bounce
                viol_l_2 = mutant < min_b
                mutant[viol_l_2] = min_b[np.where(viol_l_2)[1]]
                
            # Reflect upper bound violations
            viol_u = mutant > max_b
            if np.any(viol_u):
                # Reflection: max - (v - max)
                mutant[viol_u] = 2 * max_b[np.where(viol_u)[1]] - mutant[viol_u]
                # Fix double bounce
                viol_u_2 = mutant > max_b
                mutant[viol_u_2] = max_b[np.where(viol_u_2)[1]]
            
            # --- Crossover (Binomial) ---
            rand_j = np.random.rand(pop_size, dim)
            mask_cross = rand_j < CR_trial[:, None]
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask_cross[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask_cross, mutant, population)
            
            # --- Selection ---
            # Evaluate trials
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    # Trial is better or equal
                    population[i] = trial_pop[i]
                    fitness[i] = f_trial
                    F[i] = F_trial[i]
                    CR[i] = CR_trial[i]
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = trial_pop[i].copy()
            
            # --- Gaussian Local Search on Best ---
            # Small refinement step around the global best
            if best_sol is not None:
                # Decaying sigma based on progress
                # Decays from 5% of range down to near 0
                sigma_scale = 1.0 - progress
                sigma = (diff_b * 0.05 * sigma_scale)
                
                # Single perturbation
                perturbation = np.random.normal(0, 1, dim) * sigma
                candidate = best_sol + perturbation
                candidate = np.clip(candidate, min_b, max_b)
                
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                
                f_cand = func(candidate)
                if f_cand < best_val:
                    best_val = f_cand
                    best_sol = candidate
                    # Replace worst in population to maintain size
                    worst_idx = np.argmax(fitness)
                    population[worst_idx] = candidate
                    fitness[worst_idx] = f_cand

    return best_val
