#The following algorithm implements **JADE (Adaptive Differential Evolution with Optional External Archive)** combined with a **Restart Mechanism**.
#
#### **Key Improvements:**
#1.  **Adaptive Control Parameters ($F$, $CR$)**: The algorithm automatically tunes the mutation factor $F$ and crossover rate $CR$ for each individual (similar to jDE), removing the need for manual hyperparameter tuning.
#2.  **`current-to-pbest/1` Mutation**: Instead of greedily following the absolute best solution (which can trap the search in local optima), this strategy directs the population towards the top $p\%$ of best individuals. This balances exploration (diversity) and exploitation (convergence).
#3.  **External Archive**: Solutions that are replaced by better offspring are stored in an archive. This archive is used during mutation to provide diversity in the difference vectors, significantly preventing premature convergence.
#4.  **Restart Mechanism**: If the population stagnates (no improvement for a set number of generations) or converges (standard deviation of fitness becomes negligible), the algorithm restarts the population while preserving the global best solution found so far.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Adaptive Differential Evolution (JADE-like variant)
    with an Archive and Restart Mechanism.
    """
    start_time = datetime.now()
    # Time management: use a buffer to ensure we return before hard timeout
    time_limit = timedelta(seconds=max_time)
    end_time = start_time + time_limit - timedelta(seconds=0.05)

    # --- Hyperparameters ---
    # Population size: Standard setting (10*dim to 20*dim). 
    # Capped at 100 to ensure iteration speed, min 30 for diversity.
    pop_size = int(np.clip(15 * dim, 30, 100))
    
    # Archive size (usually equal to population size)
    archive_size = pop_size
    
    # Adaptation probabilities (jDE style)
    tau_f = 0.1
    tau_cr = 0.1
    
    # p-best parameter for mutation (Greedy but robust)
    p_best_rate = 0.10  # Top 10%
    
    # Restart triggers
    stall_limit = 30    # Max generations without improvement
    tol_std = 1e-7      # Convergence tolerance

    # --- Setup Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Helper to generate new random population
    def init_pop(n_size):
        return min_b + np.random.rand(n_size, dim) * diff_b

    # 1. Initialization
    pop = init_pop(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    # Self-adaptive parameters initialized randomly
    # F in [0.1, 0.9], CR in [0.0, 1.0]
    F = np.random.uniform(0.1, 0.9, pop_size)
    CR = np.random.uniform(0.0, 1.0, pop_size)
    
    # External Archive
    archive = []
    
    best_fitness = float('inf')
    best_sol = None
    
    # Initial Evaluation
    for i in range(pop_size):
        if datetime.now() >= end_time:
            # Early exit if time is extremely tight
            return best_fitness if best_fitness != float('inf') else func(pop[i])
        
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()
            
    stall_counter = 0
    
    # --- Main Optimization Loop ---
    while True:
        if datetime.now() >= end_time:
            return best_fitness
        
        # --- A. Update Control Parameters (jDE Logic) ---
        # With probability tau, generate new F/CR, otherwise keep old
        mask_f = np.random.rand(pop_size) < tau_f
        mask_cr = np.random.rand(pop_size) < tau_cr
        
        if np.any(mask_f):
            F[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
        if np.any(mask_cr):
            CR[mask_cr] = np.random.rand(np.sum(mask_cr))
            
        # --- B. Mutation: current-to-pbest/1 ---
        # V_i = X_i + F_i * (X_pbest - X_i) + F_i * (X_r1 - X_r2)
        
        # 1. Identify X_pbest
        sorted_idx = np.argsort(fitness)
        # Determine size of p-best group (at least 2 individuals)
        num_pbest = max(2, int(pop_size * p_best_rate))
        top_indices = sorted_idx[:num_pbest]
        
        # Assign a random p-best neighbor to each individual
        pbest_indices = np.random.choice(top_indices, pop_size)
        X_pbest = pop[pbest_indices]
        
        # 2. Identify X_r1 (from Population, distinct from i)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        for i in range(pop_size):
            while r1_indices[i] == i:
                r1_indices[i] = np.random.randint(0, pop_size)
        X_r1 = pop[r1_indices]
        
        # 3. Identify X_r2 (from Union of Population and Archive, distinct from i and r1)
        # Create Union Population
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        len_union = len(union_pop)
        r2_indices = np.random.randint(0, len_union, pop_size)
        
        for i in range(pop_size):
            # If r2 is in the current population range, it must check distinctness from i.
            # It must always check distinctness from r1.
            while (r2_indices[i] < pop_size and r2_indices[i] == i) or (r2_indices[i] == r1_indices[i]):
                r2_indices[i] = np.random.randint(0, len_union)
                
        X_r2 = union_pop[r2_indices]
        
        # 4. Compute Mutant Vectors
        F_col = F[:, None]
        mutant = pop + F_col * (X_pbest - pop) + F_col * (X_r1 - X_r2)
        
        # Bounds Constraint (Clip)
        mutant = np.clip(mutant, min_b, max_b)
        
        # --- C. Crossover ---
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR[:, None]
        
        # Ensure at least one dimension is taken from mutant to avoid duplication
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # --- D. Selection ---
        improved_any = False
        
        for i in range(pop_size):
            # Strict time check inside the loop
            if datetime.now() >= end_time:
                return best_fitness
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # 1. Update Archive (Store the replaced solution)
                if len(archive) < archive_size:
                    archive.append(pop[i].copy())
                else:
                    # Randomly replace an existing archive member
                    arc_idx = np.random.randint(0, archive_size)
                    archive[arc_idx] = pop[i].copy()
                
                # 2. Update Population
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                # 3. Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    improved_any = True
        
        # --- E. Restart Mechanism ---
        # Update stall counter
        if improved_any:
            stall_counter = 0
        else:
            stall_counter += 1
            
        # Check convergence (standard deviation of fitness)
        fit_std = np.std(fitness)
        
        # Restart if stalled too long or converged
        if stall_counter >= stall_limit or fit_std < tol_std:
            # Re-initialize population
            pop = init_pop(pop_size)
            fitness.fill(float('inf'))
            
            # Elitism: Inject the best solution found so far into the new population
            pop[0] = best_sol.copy()
            fitness[0] = best_fitness
            
            # Reset Parameters and Archive for a fresh search context
            F = np.random.uniform(0.1, 0.9, pop_size)
            CR = np.random.uniform(0.0, 1.0, pop_size)
            archive = []
            
            stall_counter = 0
            
            # Evaluate the new population (skipping the elite index 0)
            for i in range(1, pop_size):
                if datetime.now() >= end_time:
                    return best_fitness
                
                val = func(pop[i])
                fitness[i] = val
                
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()

    return best_fitness
