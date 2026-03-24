#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 205.319777248808
#2. output value is: 34.672250836311804
#3. output value is: 19.52283558004548
#4. output value is: 21.084219718900755
#5. output value is: 36.68353309516924
#6. output value is: 35.57453524271733
#7. output value is: 37.541773629154484
#
#The following algorithm implements an improved version of **Algorithm 3 (jDE + LPSR)**. It enhances the previous best approach by integrating **Reflection Boundary Handling** (to prevent solutions from sticking to bounds), **Gaussian Walk Local Search** (to refine the best solution and accelerate convergence), and **Linear Population Size Reduction (LPSR)** for optimal budget management.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using jDE (Self-Adaptive Differential Evolution)
    enhanced with Linear Population Size Reduction (LPSR), 
    Gaussian Walk Local Search (GWLS), and Reflection Boundary Handling.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    # Safety buffer to ensure we return before hard timeout
    end_time = start_time + time_limit - timedelta(seconds=0.05)

    # --- Hyperparameters ---
    # LPSR: Start large for Exploration, end small for Exploitation
    # 20*dim is a robust heuristic for initial population
    init_pop_size = int(np.clip(20 * dim, 50, 200))
    min_pop_size = 5
    
    current_pop_size = init_pop_size
    
    # jDE Parameter Adaptation Rates
    tau_f = 0.1
    tau_cr = 0.1
    
    # Restart triggers
    stall_limit = 30
    tol_std = 1e-8
    
    # Local Search Parameters
    ls_iter = 2 # Number of local search steps per generation on best_sol
    
    # --- Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Helper: Reflection Boundary Handling
    # Prevents particles from sticking to the edges of the search space
    def boundary_handle(x):
        # Bounce off lower bounds
        mask_l = x < min_b
        if np.any(mask_l):
            x[mask_l] = 2.0 * min_b[mask_l] - x[mask_l]
            
        # Bounce off upper bounds
        mask_u = x > max_b
        if np.any(mask_u):
            x[mask_u] = 2.0 * max_b[mask_u] - x[mask_u]
            
        # Final safety clip in case of extreme multiple bounces
        return np.clip(x, min_b, max_b)

    # --- Initialization ---
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    # jDE Parameters (F: Mutation Factor, CR: Crossover Rate)
    # Encoded into the population, self-adapting
    F = np.random.uniform(0.1, 1.0, current_pop_size)
    CR = np.random.uniform(0.0, 1.0, current_pop_size)
    
    # Archive for diversity (stores replaced parents)
    archive = []
    
    best_fitness = float('inf')
    best_sol = None
    
    # Evaluate Initial Population
    for i in range(current_pop_size):
        if datetime.now() >= end_time:
            return best_fitness if best_fitness != float('inf') else func(pop[i])
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    stall_counter = 0
    
    # --- Main Optimization Loop ---
    while True:
        now = datetime.now()
        if now >= end_time:
            return best_fitness
            
        # 1. Update Time Progress
        elapsed = (now - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # 2. Linear Population Size Reduction (LPSR)
        # Reduce population size linearly with time to focus computational budget
        target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        if target_size < min_pop_size: target_size = min_pop_size
        
        if current_pop_size > target_size:
            # Sort by fitness and keep top N
            sorted_indices = np.argsort(fitness)
            keep_indices = sorted_indices[:target_size]
            
            pop = pop[keep_indices]
            fitness = fitness[keep_indices]
            F = F[keep_indices]
            CR = CR[keep_indices]
            current_pop_size = target_size
            
            # Reduce archive size similarly
            if len(archive) > current_pop_size:
                del archive[current_pop_size:]
                
        # 3. Parameter Adaptation (jDE)
        # Randomly reset F and CR for some individuals
        mask_f = np.random.rand(current_pop_size) < tau_f
        mask_cr = np.random.rand(current_pop_size) < tau_cr
        
        if np.any(mask_f):
            F[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f)) # F in [0.1, 1.0]
        if np.any(mask_cr):
            CR[mask_cr] = np.random.rand(np.sum(mask_cr))          # CR in [0.0, 1.0]
            
        # 4. Mutation: current-to-pbest/1
        # p-value linearly reduces from 0.15 (exploration) to 0.02 (exploitation)
        p_val = 0.15 - 0.13 * progress
        p_val = max(p_val, 2.0 / current_pop_size)
        
        # Identify p-best
        sorted_idx = np.argsort(fitness)
        num_pbest = int(max(2, current_pop_size * p_val))
        top_indices = sorted_idx[:num_pbest]
        
        # Select p-best indices
        pbest_indices = np.random.choice(top_indices, current_pop_size)
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
        for i in range(current_pop_size):
            while r1_indices[i] == i:
                r1_indices[i] = np.random.randint(0, current_pop_size)
                
        # Select r2 (distinct from i and r1; from Union of Pop and Archive)
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
        len_union = len(union_pop)
        
        r2_indices = np.random.randint(0, len_union, current_pop_size)
        for i in range(current_pop_size):
            while (r2_indices[i] < current_pop_size and r2_indices[i] == i) or r2_indices[i] == r1_indices[i]:
                r2_indices[i] = np.random.randint(0, len_union)
                
        # Calculate Mutant Vectors
        X_pbest = pop[pbest_indices]
        X_r1 = pop[r1_indices]
        X_r2 = union_pop[r2_indices]
        
        F_col = F[:, None]
        mutant = pop + F_col * (X_pbest - pop) + F_col * (X_r1 - X_r2)
        
        # Apply Reflection Boundary Handling
        mutant = boundary_handle(mutant)
        
        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(current_pop_size, dim)
        cross_mask = rand_vals < CR[:, None]
        j_rand = np.random.randint(0, dim, current_pop_size)
        cross_mask[np.arange(current_pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 6. Selection
        gen_improved = False
        for i in range(current_pop_size):
            if datetime.now() >= end_time: return best_fitness
            
            val = func(trial_pop[i])
            
            if val <= fitness[i]:
                # Update Archive: Add replaced solution
                if len(archive) < current_pop_size:
                    archive.append(pop[i].copy())
                else:
                    if len(archive) > 0:
                        ridx = np.random.randint(0, len(archive))
                        archive[ridx] = pop[i].copy()
                        
                fitness[i] = val
                pop[i] = trial_pop[i]
                
                if val < best_fitness:
                    best_fitness = val
                    best_sol = trial_pop[i].copy()
                    gen_improved = True
        
        # 7. Gaussian Walk Local Search (GWLS) on Global Best
        # Fine-tunes the best solution found so far
        if best_sol is not None:
            # Dynamic Step Size Calculation
            # Combines current population diversity (std) and global scale (diff_b)
            # Decays as time progresses
            pop_std_avg = np.mean(np.std(pop, axis=0))
            scale = (diff_b * 0.01 * (1.0 - progress)) + (pop_std_avg * 0.5)
            
            for _ in range(ls_iter):
                if datetime.now() >= end_time: return best_fitness
                
                # Sample normally around best_sol
                noise = np.random.normal(0, 1, dim) * scale
                cand = boundary_handle(best_sol + noise)
                val_ls = func(cand)
                
                if val_ls < best_fitness:
                    best_fitness = val_ls
                    best_sol = cand.copy()
                    gen_improved = True
                    # Share info: replace worst individual in population
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = cand
                    fitness[worst_idx] = val_ls

        # 8. Restart Mechanism
        if gen_improved:
            stall_counter = 0
        else:
            stall_counter += 1
            
        fit_std = np.std(fitness)
        
        # Trigger restart if stalled or converged
        # Only if we have > 10% time remaining (no point restarting at the very end)
        if (stall_counter >= stall_limit or fit_std < tol_std) and progress < 0.90:
            stall_counter = 0
            
            # Re-initialize population
            pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
            fitness = np.full(current_pop_size, float('inf'))
            
            # Elitism: Inject Global Best into new population
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset Parameters and Archive for fresh search
            F = np.random.uniform(0.1, 1.0, current_pop_size)
            CR = np.random.uniform(0.0, 1.0, current_pop_size)
            archive = []
            
            # Evaluate new population (skip elite at index 0)
            for i in range(1, current_pop_size):
                if datetime.now() >= end_time: return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()

    return best_fitness
