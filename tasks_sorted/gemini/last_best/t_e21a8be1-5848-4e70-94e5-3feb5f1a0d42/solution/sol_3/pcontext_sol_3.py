#The following algorithm implements **L-SHADE (Success-History based Adaptive Differential Evolution with Linear Population Size Reduction) with Restart**.
#
#**Key Improvements:**
#1.  **L-SHADE Architecture**: This is an enhancement over JADE. It maintains a **historical memory** ($M_{CR}, M_F$) of successful control parameters rather than a simple mean. This allows the algorithm to recall and utilize diverse parameter configurations that worked well in different stages of optimization.
#2.  **Linear Population Size Reduction (LPSR)**: The population size linearly decreases from a large initial value ($N_{init}$) to a small minimum ($N_{min}$) based on the remaining time. This fosters exploration at the beginning (high diversity) and exploitation at the end (fast convergence).
#3.  **Time-Based Restart Strategy**: If the population converges (low variance) or shrinks to the minimum size while time remains, the algorithm **restarts** with a fresh population. This prevents getting stuck in local optima and utilizes the full computational budget.
#4.  **Vectorized Operations**: Parameter generation and mutation are vectorized using NumPy for efficiency, minimizing overhead outside the objective function evaluation.
#
import numpy as np
import random
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Restart and Linear Population Reduction.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Helper to check if time is up
    def is_timeout():
        return datetime.now() - start_time >= time_limit

    # Parse bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Global best solution found across all restarts
    best_val = float('inf')
    best_sol = None
    
    # --- Configuration ---
    # Initial population size: High value for exploration. 
    # Capped at 300 to prevent excessive slowness in high dimensions/expensive functions.
    N_init = int(round(max(20, min(300, 18 * dim))))
    N_min = 4
    
    # Memory size for historical parameters
    H = 5
    
    # --- Main Loop (Handles Restarts) ---
    while True:
        if is_timeout(): break
        
        # Determine time budget for this run
        elapsed_now = datetime.now() - start_time
        remaining_seconds = max(0.01, max_time - elapsed_now.total_seconds())
        run_start_time = datetime.now()
        
        # If very little time is left, a restart might not be useful, but we try anyway.
        
        # Initialize Memory
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0  # Memory index
        
        # Initialize Population (Latin Hypercube Sampling)
        pop_size = N_init
        population = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
            samples = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(samples)
            population[:, d] = samples
            
        fitness = np.full(pop_size, float('inf'))
        archive = []
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if is_timeout(): return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_sol = population[i].copy()
        
        # --- Evolutionary Loop ---
        while True:
            if is_timeout(): return best_val
            
            # 1. Linear Population Size Reduction (LPSR)
            # Calculate progress relative to the remaining time for this restart
            run_elapsed = (datetime.now() - run_start_time).total_seconds()
            progress = run_elapsed / remaining_seconds
            if progress > 1.0: progress = 1.0
            
            # Target population size based on progress
            target_size = int(round((N_min - N_init) * progress + N_init))
            target_size = max(N_min, target_size)
            
            # Resize if needed
            if pop_size > target_size:
                # Keep best individuals
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices[:target_size]]
                fitness = fitness[sorted_indices[:target_size]]
                pop_size = target_size
                
                # Resize archive to match new pop size
                if len(archive) > pop_size:
                    archive = random.sample(archive, pop_size)
            
            # 2. Convergence/Restart Check
            # If population is minimal or variance is negligible, restart to explore new basins
            if pop_size <= N_min or np.std(fitness) < 1e-9:
                break
                
            # 3. Parameter Generation (Vectorized)
            # Pick random memory slot for each individual
            r_indices = np.random.randint(0, H, pop_size)
            mu_cr = M_cr[r_indices]
            mu_f = M_f[r_indices]
            
            # Generate CR ~ Normal(mu_cr, 0.1)
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(mu_f, 0.1)
            f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
            f = np.where(f <= 0, 0.1, f) # Clamp <= 0 to 0.1
            f = np.clip(f, 0, 1)         # Clamp > 1 to 1
            
            # 4. Mutation: current-to-pbest/1
            # Sort population to find p-best
            sorted_indices = np.argsort(fitness)
            
            # p-best selection (random from top p%)
            p = max(2.0/pop_size, 0.11)
            top_p_count = int(max(2, pop_size * p))
            
            pbest_ptr = np.random.randint(0, top_p_count, pop_size)
            pbest_indices = sorted_indices[pbest_ptr]
            x_pbest = population[pbest_indices]
            
            # r1 selection (random distinct from i)
            r1_indices = np.random.randint(0, pop_size, pop_size)
            # Fix collisions where r1 == i
            collisions = (r1_indices == np.arange(pop_size))
            while np.any(collisions):
                r1_indices[collisions] = np.random.randint(0, pop_size, np.sum(collisions))
                collisions = (r1_indices == np.arange(pop_size))
            x_r1 = population[r1_indices]
            
            # r2 selection (random distinct from i and r1, from Pop U Archive)
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((population, archive_np))
            else:
                union_pop = population
            
            union_size = len(union_pop)
            r2_indices = np.random.randint(0, union_size, pop_size)
            
            # Fix collisions for r2 (simple iterative fix is fast enough for reduced pop)
            for i in range(pop_size):
                while r2_indices[i] == i or r2_indices[i] == r1_indices[i]:
                    r2_indices[i] = np.random.randint(0, union_size)
            x_r2 = union_pop[r2_indices]
            
            # Compute Mutant Vector V
            F_matrix = f[:, np.newaxis]
            mutant = population + F_matrix * (x_pbest - population) + F_matrix * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 5. Crossover (Binomial)
            mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            # Ensure at least one dimension is mutated
            j_rand = np.random.randint(0, dim, pop_size)
            for i in range(pop_size):
                mask[i, j_rand[i]] = True
                
            trial = np.where(mask, mutant, population)
            
            # 6. Selection
            new_population = []
            new_fitness = []
            success_f = []
            success_cr = []
            diff_fitness = []
            
            for i in range(pop_size):
                if is_timeout(): return best_val
                
                val_trial = func(trial[i])
                
                if val_trial <= fitness[i]:
                    new_population.append(trial[i])
                    new_fitness.append(val_trial)
                    
                    # Add loser (parent) to archive
                    archive.append(population[i].copy())
                    
                    # Record success
                    if val_trial < fitness[i]:
                        success_f.append(f[i])
                        success_cr.append(cr[i])
                        diff_fitness.append(fitness[i] - val_trial)
                    
                    if val_trial < best_val:
                        best_val = val_trial
                        best_sol = trial[i].copy()
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
            
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            # Maintain Archive Size
            if len(archive) > pop_size:
                archive = random.sample(archive, pop_size)
                
            # 7. Update Historical Memory (SHADE logic)
            if len(success_f) > 0:
                s_f = np.array(success_f)
                s_cr = np.array(success_cr)
                w = np.array(diff_fitness)
                
                # Normalize weights
                if np.sum(w) > 0:
                    w = w / np.sum(w)
                    
                    # Weighted Lehmer Mean for F
                    mean_sf = np.sum(w * (s_f ** 2)) / np.sum(w * s_f)
                    
                    # Weighted Arithmetic Mean for CR
                    mean_scr = np.sum(w * s_cr)
                    
                    # Update memory slot
                    M_cr[k_mem] = mean_scr
                    M_f[k_mem] = mean_sf
                    k_mem = (k_mem + 1) % H

    return best_val
