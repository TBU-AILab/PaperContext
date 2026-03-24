#The following code implements the **L-SHADE** (Linear Population Size Reduction - Success-History based Adaptive Differential Evolution) algorithm.
#
#**Key Improvements over the previous algorithm:**
#1.  **L-SHADE Strategy**: This is a state-of-the-art variant of Differential Evolution that wins many optimization competitions. It performs significantly better on multimodal and complex functions than standard DE or jDE.
#2.  **Linear Population Size Reduction (LPSR)**: The population size starts large to maximize exploration and linearly decreases over time to focus on exploitation (fine-tuning) as the deadline approaches.
#3.  **History-Based Parameter Adaptation**: Instead of random adaptation (jDE), it uses a memory of successful control parameters ($F$ and $CR$) from previous generations to guide the search distribution.
#4.  **External Archive**: Maintains a history of recently replaced inferior solutions to preserve diversity and prevent premature convergence.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History based Adaptive Differential Evolution).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    # --- Parameters ---
    # Initial population size: 18 * dim is a standard heuristic for L-SHADE
    # Capped at 500 for very high dimensions to maintain speed
    initial_pop_size = int(max(30, min(500, 18 * dim)))
    min_pop_size = 4
    
    # Size of the memory for adaptive parameters
    history_size = 6 
    
    # Pre-process bounds
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b

    # --- Initialization ---
    pop_size = initial_pop_size
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Memory for Scaling Factor (F) and Crossover Rate (CR)
    # initialized to 0.5
    memory_sf = np.full(history_size, 0.5)
    memory_scr = np.full(history_size, 0.5)
    mem_k = 0  # memory index pointer

    # External Archive
    archive = []

    # Best solution tracking
    best_val = float('inf')
    best_idx = -1

    # --- Evaluate Initial Population ---
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Loop ---
    while True:
        elapsed = datetime.now() - start_time
        if elapsed >= time_limit:
            return best_val

        # 1. Linear Population Size Reduction (LPSR)
        # Calculate progress ratio (0.0 to 1.0) based on time
        progress = min(1.0, elapsed.total_seconds() / max_time)
        
        # Calculate target population size based on linear reduction
        target_pop_size = int(round((min_pop_size - initial_pop_size) * progress + initial_pop_size))
        target_pop_size = max(min_pop_size, target_pop_size)

        # Resize population if needed
        if pop_size > target_pop_size:
            # Sort population by fitness (ascending, best first)
            sort_indices = np.argsort(fitness)
            # Keep only the top 'target_pop_size' individuals
            keep_indices = sort_indices[:target_pop_size]
            
            pop = pop[keep_indices]
            fitness = fitness[keep_indices]
            pop_size = target_pop_size
            
            # Resize archive (keep random elements up to new pop_size)
            if len(archive) > pop_size:
                # Shuffle and slice
                import random
                random.shuffle(archive)
                archive = archive[:pop_size]

            # Re-locate best index (it's at 0 after sort, but let's be robust)
            best_idx = np.argmin(fitness)
            best_val = fitness[best_idx]

        # 2. Parameter Generation (SHADE)
        # Select random memory index for each individual
        r_indices = np.random.randint(0, history_size, pop_size)
        mu_sf = memory_sf[r_indices]
        mu_scr = memory_scr[r_indices]

        # Generate CR ~ Normal(mu_scr, 0.1), clipped to [0, 1]
        cr = np.random.normal(mu_scr, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F ~ Cauchy(mu_sf, 0.1)
        f = np.zeros(pop_size)
        for i in range(pop_size):
            while True:
                val = mu_sf[i] + 0.1 * np.random.standard_cauchy()
                if val > 0:
                    if val > 1: val = 1.0
                    f[i] = val
                    break

        # 3. Mutation: current-to-pbest/1
        # v = x + F * (x_pbest - x) + F * (x_r1 - x_r2)
        
        # Select pbest from top p% individuals (p=0.11 is common)
        p_best_rate = 0.11
        num_pbest = max(2, int(p_best_rate * pop_size))
        sorted_indices = np.argsort(fitness)
        pbest_indices = sorted_indices[:num_pbest]

        mutants = np.zeros_like(pop)
        
        # Prepare Union P U A for r2 selection
        if len(archive) > 0:
            archive_arr = np.array(archive)
            union_pop = np.vstack((pop, archive_arr))
        else:
            union_pop = pop

        for i in range(pop_size):
            # Select pbest
            pbest = pop[np.random.choice(pbest_indices)]
            
            # Select r1 from P, r1 != i
            while True:
                r1 = np.random.randint(0, pop_size)
                if r1 != i: break
            xr1 = pop[r1]
            
            # Select r2 from P U A, r2 != i, r2 != r1
            while True:
                r2 = np.random.randint(0, len(union_pop))
                # Check constraints
                if r2 < pop_size: # r2 is in P
                    if r2 != i and r2 != r1: break
                else: # r2 is in A (distinct memory)
                    break
            xr2 = union_pop[r2]
            
            # Mutation
            mutants[i] = pop[i] + f[i] * (pbest - pop[i]) + f[i] * (xr1 - xr2)

        # 4. Crossover (Binomial)
        # Generate random mask based on CR
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals <= cr[:, None]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trials = np.where(mask, mutants, pop)
        
        # Constrain to bounds
        trials = np.clip(trials, min_b, max_b)

        # 5. Selection
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        success_sf = []
        success_scr = []
        diff_f = []

        for i in range(pop_size):
            # Time check inside evaluation loop
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            trial_val = func(trials[i])
            
            if trial_val <= fitness[i]:
                # Improvement or equal
                if trial_val < fitness[i]:
                    # Add replaced parent to archive
                    archive.append(pop[i].copy())
                    
                    # Record successful parameters
                    success_sf.append(f[i])
                    success_scr.append(cr[i])
                    diff_f.append(fitness[i] - trial_val)
                
                # Update population
                new_pop[i] = trials[i]
                new_fitness[i] = trial_val
                
                # Update global best
                if trial_val < best_val:
                    best_val = trial_val
                    best_idx = i
        
        pop = new_pop
        fitness = new_fitness

        # Maintain Archive Size (<= pop_size)
        while len(archive) > pop_size:
            # Remove random element
            archive.pop(np.random.randint(0, len(archive)))

        # 6. Update Memory (Weighted Lehmer Mean)
        if len(success_sf) > 0:
            success_sf = np.array(success_sf)
            success_scr = np.array(success_scr)
            diff_f = np.array(diff_f)
            
            # Weights based on fitness improvement
            if np.sum(diff_f) > 0:
                weights = diff_f / np.sum(diff_f)
                
                # Weighted Lehmer Mean for F
                mean_sf = np.sum(weights * (success_sf ** 2)) / np.sum(weights * success_sf)
                
                # Weighted Mean for CR
                mean_scr = np.sum(weights * success_scr)
            else:
                mean_sf = np.mean(success_sf)
                mean_scr = np.mean(success_scr)
            
            # Update history slots
            memory_sf[mem_k] = mean_sf
            memory_scr[mem_k] = mean_scr
            mem_k = (mem_k + 1) % history_size

    return best_val
