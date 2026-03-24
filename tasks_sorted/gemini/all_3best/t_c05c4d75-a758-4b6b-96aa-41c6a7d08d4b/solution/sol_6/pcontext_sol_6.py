#The following algorithm implements an **Enhanced jDE (Self-Adaptive Differential Evolution)** with **Linear Population Size Reduction (LPSR)**, **Opposition-Based Learning (OBL)**, and a **Reflection-based Boundary Handling** strategy.
#
##### **Key Improvements:**
#1.  **Opposition-Based Learning (OBL)**: Applied during initialization and restarts. For every random solution generated, its opposite solution ($min + max - x$) is also considered. The algorithm greedily selects the better of the two, significantly improving the quality of the initial population and restarts.
#2.  **Reflection Boundary Handling**: Instead of simply clipping values to bounds (which piles solutions at the edges), particles that overshoot the bounds "bounce" back into the search space. This preserves diversity and distribution near the boundaries.
#3.  **LPSR (Linear Population Size Reduction)**: Linearly reduces the population size from a large value (exploration) to a small value (exploitation) as the time limit approaches, optimizing the computational budget.
#4.  **jDE with Time-Aware Restart**: Uses self-adaptive parameters ($F, CR$) and triggers a restart if the population stagnates or converges, but only if sufficient time remains to make the restart meaningful.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Enhanced jDE (Self-Adaptive Differential Evolution)
    with Opposition-Based Learning (OBL), Linear Population Size Reduction (LPSR),
    and Reflection Boundary Handling.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    # End time buffer to ensure we return a result before hard timeout
    end_time = start_time + time_limit - timedelta(seconds=0.05)

    # --- Hyperparameters ---
    # Population Size: LPSR from Init to Min
    # Larger initial population for better exploration
    max_pop_init = 200
    min_pop_init = 50
    # Scale initial population with dimension but cap it
    init_pop_size = int(np.clip(25 * dim, min_pop_init, max_pop_init))
    min_pop_size = 5
    
    current_pop_size = init_pop_size
    
    # jDE Adaptation Rates (Probabilities to update F and CR)
    tau_f = 0.1
    tau_cr = 0.1
    
    # Restart triggers
    stall_limit = 25
    stall_counter = 0
    tol_std = 1e-8
    
    # --- Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best
    best_fitness = float('inf')
    best_sol = None
    
    # --- Helper Functions ---
    def check_time():
        return datetime.now() >= end_time

    def boundary_handle(x):
        """
        Reflection/Bounce-back strategy:
        If a particle overshoots the bound, it is reflected back into the space.
        """
        x_new = np.copy(x)
        
        # Lower bound reflection: x' = min + (min - x) = 2*min - x
        mask_l = x_new < min_b
        x_new[mask_l] = 2 * min_b[mask_l] - x_new[mask_l]
        
        # Upper bound reflection: x' = max - (x - max) = 2*max - x
        mask_u = x_new > max_b
        x_new[mask_u] = 2 * max_b[mask_u] - x_new[mask_u]
        
        # Final clip to ensure numerical safety (in case of extreme double overshoot)
        return np.clip(x_new, min_b, max_b)

    # --- Initialization with OBL (Opposition-Based Learning) ---
    # 1. Generate random population
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    # Evaluate initial population
    for i in range(current_pop_size):
        if check_time(): return best_fitness
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # 2. OBL Step: Check opposite points
    # X_opp = min + max - X
    if not check_time():
        opp_pop = min_b + max_b - pop
        # Clip opposite points to bounds
        opp_pop = np.clip(opp_pop, min_b, max_b)
        
        for i in range(current_pop_size):
            if check_time(): return best_fitness
            
            # Evaluate Opposition
            val_opp = func(opp_pop[i])
            
            # Greedy selection: if Opposite is better, swap
            if val_opp < fitness[i]:
                fitness[i] = val_opp
                pop[i] = opp_pop[i]
                if val_opp < best_fitness:
                    best_fitness = val_opp
                    best_sol = opp_pop[i].copy()

    # Initialize jDE Control Parameters (F in [0.1, 1.0], CR in [0.0, 1.0])
    F = np.random.uniform(0.1, 1.0, current_pop_size)
    CR = np.random.uniform(0.0, 1.0, current_pop_size)
    
    # External Archive (stores replaced parent solutions for diversity)
    archive = []
    
    # --- Main Optimization Loop ---
    while True:
        now = datetime.now()
        if now >= end_time:
            return best_fitness
            
        # 1. Linear Population Size Reduction (LPSR)
        elapsed = (now - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # Calculate target population size
        target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        if target_size < min_pop_size:
            target_size = min_pop_size
            
        # Reduce population if needed
        if current_pop_size > target_size:
            # Sort by fitness (ascending)
            sorted_indices = np.argsort(fitness)
            keep_indices = sorted_indices[:target_size]
            
            # Truncate
            pop = pop[keep_indices]
            fitness = fitness[keep_indices]
            F = F[keep_indices]
            CR = CR[keep_indices]
            current_pop_size = target_size
            
            # Resize Archive
            if len(archive) > current_pop_size:
                del archive[current_pop_size:]
        
        # 2. Adaptation (jDE)
        # Probabilistically update F and CR
        mask_f = np.random.rand(current_pop_size) < tau_f
        mask_cr = np.random.rand(current_pop_size) < tau_cr
        
        if np.any(mask_f):
            F[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f)) # F ~ U(0.1, 1.0)
        if np.any(mask_cr):
            CR[mask_cr] = np.random.rand(np.sum(mask_cr))          # CR ~ U(0.0, 1.0)
            
        # 3. Mutation: current-to-pbest/1
        # p linearly decreases from 0.15 to 0.05
        p_val = 0.15 - 0.10 * progress
        p_val = max(p_val, 2.0 / current_pop_size)
        
        sorted_idx = np.argsort(fitness)
        num_pbest = int(max(2, current_pop_size * p_val))
        top_indices = sorted_idx[:num_pbest]
        
        # Select p-best
        pbest_indices = np.random.choice(top_indices, current_pop_size)
        X_pbest = pop[pbest_indices]
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
        for i in range(current_pop_size):
            while r1_indices[i] == i:
                r1_indices[i] = np.random.randint(0, current_pop_size)
        X_r1 = pop[r1_indices]
        
        # Select r2 (distinct from i and r1; from Union of Pop and Archive)
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
        len_union = len(union_pop)
        
        r2_indices = np.random.randint(0, len_union, current_pop_size)
        for i in range(current_pop_size):
            while (r2_indices[i] < current_pop_size and (r2_indices[i] == i or r2_indices[i] == r1_indices[i])):
                r2_indices[i] = np.random.randint(0, len_union)
        X_r2 = union_pop[r2_indices]
        
        # Calculate Mutant Vector
        F_col = F[:, None]
        mutant = pop + F_col * (X_pbest - pop) + F_col * (X_r1 - X_r2)
        
        # Apply Reflection Boundary Handling
        mutant = boundary_handle(mutant)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(current_pop_size, dim)
        cross_mask = rand_vals < CR[:, None]
        j_rand = np.random.randint(0, dim, current_pop_size)
        cross_mask[np.arange(current_pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 5. Selection and Archive Update
        gen_improved = False
        for i in range(current_pop_size):
            if check_time(): return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Add replaced solution to Archive
                if len(archive) < current_pop_size:
                    archive.append(pop[i].copy())
                elif len(archive) > 0:
                    ridx = np.random.randint(0, len(archive))
                    archive[ridx] = pop[i].copy()
                
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    gen_improved = True
                    
        # 6. Restart Mechanism
        if gen_improved:
            stall_counter = 0
        else:
            stall_counter += 1
            
        fit_std = np.std(fitness)
        
        # Trigger restart if stalled or converged (low variance)
        # Only restart if there is enough time left (>10%) to justify re-exploration
        if (stall_counter >= stall_limit or fit_std < tol_std) and progress < 0.9:
            stall_counter = 0
            
            # --- Restart with OBL ---
            # 1. Elitism: Keep the global best
            new_pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
            new_pop[0] = best_sol.copy()
            new_fitness = np.full(current_pop_size, float('inf'))
            new_fitness[0] = best_fitness
            
            # 2. OBL for the rest of the population
            opp_pop = min_b + max_b - new_pop
            opp_pop = np.clip(opp_pop, min_b, max_b)
            
            # Reset Parameters and Archive
            F = np.random.uniform(0.1, 1.0, current_pop_size)
            CR = np.random.uniform(0.0, 1.0, current_pop_size)
            archive = []
            
            # Evaluate new population (skipping elite index 0)
            for i in range(1, current_pop_size):
                if check_time(): return best_fitness
                
                # Compare Random vs Opposite
                
                # Evaluate Random
                val_rand = func(new_pop[i])
                
                # Check time before second evaluation
                if check_time():
                    if val_rand < best_fitness:
                        best_fitness = val_rand
                    return best_fitness

                # Evaluate Opposite
                val_opp = func(opp_pop[i])
                
                # Pick better
                if val_opp < val_rand:
                    new_pop[i] = opp_pop[i]
                    new_fitness[i] = val_opp
                    val = val_opp
                else:
                    new_fitness[i] = val_rand
                    val = val_rand
                    
                if val < best_fitness:
                    best_fitness = val
                    best_sol = new_pop[i].copy()
            
            pop = new_pop
            fitness = new_fitness

    return best_fitness
