#The following algorithm implements an enhanced **jDE (Self-Adaptive Differential Evolution)** with **LPSR (Linear Population Size Reduction)** and a **Time-Aware Restart Mechanism**.
#
##### **Key Improvements:**
#1.  **LPSR (Linear Population Size Reduction)**: Instead of a fixed population size, the algorithm starts with a large population to maximize diversity and exploration, then linearly reduces it over time. This forces the algorithm to focus its computational budget on refining the best candidates (exploitation) as the time limit approaches.
#2.  **Adaptive Parameters (jDE)**: Utilizes the robust jDE mechanism where mutation factor $F$ and crossover rate $CR$ are encoded in the individuals and evolved, avoiding manual tuning and adapting to the fitness landscape.
#3.  **`current-to-pbest/1` with Dynamic $p$**: Moves solutions towards the top $p\%$ of individuals. The value of $p$ decreases over time (from 15% to 5%), transitioning the search from exploring good regions to converging on the global optimum.
#4.  **Time-Aware Restart**: A mechanism to restart the population if convergence is detected (low variance) or stagnation occurs. Crucially, restarts are disabled near the end of the time limit to prevent wasting the remaining budget on random initialization.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using jDE (Self-Adaptive Differential Evolution)
    enhanced with Linear Population Size Reduction (LPSR) and a Restart Mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    # End time buffer to ensure we return a result before hard timeout
    end_time = start_time + time_limit - timedelta(seconds=0.05)

    # --- Hyperparameters ---
    # Population Size: LPSR from Init to Min
    # Start larger to explore, reduce over time to exploit
    max_pop_init = 150
    min_pop_init = 40
    # Scale initial population with dimension but cap it
    init_pop_size = int(np.clip(20 * dim, min_pop_init, max_pop_init))
    min_pop_size = 5
    
    current_pop_size = init_pop_size
    
    # jDE Adaptation Rates (Probabilities to update F and CR)
    tau_f = 0.1
    tau_cr = 0.1
    
    # Restart triggers
    stall_limit = 30
    stall_counter = 0
    tol_std = 1e-8
    
    # --- Setup ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    # jDE Control Parameters (F in [0.1, 1.0], CR in [0.0, 1.0])
    F = np.random.uniform(0.1, 1.0, current_pop_size)
    CR = np.random.uniform(0.0, 1.0, current_pop_size)
    
    # External Archive for diversity (stores replaced parent solutions)
    archive = []
    
    best_fitness = float('inf')
    best_sol = None
    
    # Initial Evaluation
    for i in range(current_pop_size):
        if datetime.now() >= end_time:
            return best_fitness if best_fitness != float('inf') else func(pop[i])
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        now = datetime.now()
        if now >= end_time:
            return best_fitness
            
        # 1. Linear Population Size Reduction (LPSR) based on Time
        elapsed = (now - start_time).total_seconds()
        progress = elapsed / max_time
        if progress > 1.0: progress = 1.0
        
        # Calculate target population size based on remaining time
        # Formula: N_t = N_init + (N_min - N_init) * progress
        target_size = int(round(init_pop_size + (min_pop_size - init_pop_size) * progress))
        if target_size < min_pop_size:
            target_size = min_pop_size
            
        # Reduce population if needed
        if current_pop_size > target_size:
            # Sort by fitness (ascending) to keep the best
            sorted_indices = np.argsort(fitness)
            keep_indices = sorted_indices[:target_size]
            
            # Truncate Population and Parameters
            pop = pop[keep_indices]
            fitness = fitness[keep_indices]
            F = F[keep_indices]
            CR = CR[keep_indices]
            current_pop_size = target_size
            
            # Truncate Archive to match current population size
            if len(archive) > current_pop_size:
                del archive[current_pop_size:]
                
        # 2. Adaptation (jDE)
        # Probabilistically generate new control parameters
        mask_f = np.random.rand(current_pop_size) < tau_f
        mask_cr = np.random.rand(current_pop_size) < tau_cr
        
        if np.any(mask_f):
            F[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f)) # F in [0.1, 1.0]
        if np.any(mask_cr):
            CR[mask_cr] = np.random.rand(np.sum(mask_cr))          # CR in [0.0, 1.0]
            
        # 3. Mutation: current-to-pbest/1
        # Dynamic p: scales linearly from 0.15 down to 0.05
        p_val = 0.15 - 0.10 * progress
        p_val = max(p_val, 2.0 / current_pop_size) # Ensure p-best group has at least 2 members
        
        sorted_idx = np.argsort(fitness)
        num_pbest = int(max(2, current_pop_size * p_val))
        top_indices = sorted_idx[:num_pbest]
        
        # Select p-best individuals
        pbest_indices = np.random.choice(top_indices, current_pop_size)
        X_pbest = pop[pbest_indices]
        
        # Select r1 (distinct from i)
        r1_indices = np.random.randint(0, current_pop_size, current_pop_size)
        for i in range(current_pop_size):
            while r1_indices[i] == i:
                r1_indices[i] = np.random.randint(0, current_pop_size)
        X_r1 = pop[r1_indices]
        
        # Select r2 (distinct from i and r1; chosen from Population U Archive)
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
        len_union = len(union_pop)
        
        r2_indices = np.random.randint(0, len_union, current_pop_size)
        for i in range(current_pop_size):
            # If r2 is in the current population part of union, check distinctness
            while (r2_indices[i] < current_pop_size and (r2_indices[i] == i or r2_indices[i] == r1_indices[i])):
                r2_indices[i] = np.random.randint(0, len_union)
        X_r2 = union_pop[r2_indices]
        
        # Compute Mutant Vectors: V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        F_col = F[:, None]
        mutant = pop + F_col * (X_pbest - pop) + F_col * (X_r1 - X_r2)
        mutant = np.clip(mutant, min_b, max_b)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(current_pop_size, dim)
        cross_mask = rand_vals < CR[:, None]
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, current_pop_size)
        cross_mask[np.arange(current_pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 5. Selection and Archive Update
        gen_improved = False
        for i in range(current_pop_size):
            if datetime.now() >= end_time:
                return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Add replaced parent to archive
                if len(archive) < current_pop_size:
                    archive.append(pop[i].copy())
                else:
                    # Replace random individual in archive
                    if len(archive) > 0:
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
        # Condition: Only restart if we have substantial time left (> 15%)
        if (stall_counter >= stall_limit or fit_std < tol_std) and progress < 0.85:
            stall_counter = 0
            
            # Re-initialize population (keeping the current reduced size)
            pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
            fitness = np.full(current_pop_size, float('inf'))
            
            # Elitism: Keep the global best found so far
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset Archive and Parameters
            archive = []
            F = np.random.uniform(0.1, 1.0, current_pop_size)
            CR = np.random.uniform(0.0, 1.0, current_pop_size)
            
            # Evaluate new population (skip index 0)
            for i in range(1, current_pop_size):
                if datetime.now() >= end_time:
                    return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()

    return best_fitness
