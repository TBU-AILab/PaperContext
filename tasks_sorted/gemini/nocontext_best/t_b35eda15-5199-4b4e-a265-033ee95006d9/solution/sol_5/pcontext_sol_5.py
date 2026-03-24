#The best generated algorithm code:
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Implements SHADE (Success-History based Adaptive Differential Evolution) with an
    External Archive and Restart mechanism.
    
    Improvements over standard DE:
    1. Adaptive Parameters: Uses historical memory (H) to adapt Mutation (F) and Crossover (CR) 
       parameters based on successful updates, weighted by fitness improvement.
    2. External Archive: Preserves diversity by allowing mutation strategies to pull from 
       previously discarded inferior solutions (current-to-pbest/1 with archive).
    3. Restart Mechanism: Detects population stagnation (low variance) and resets the population
       while keeping the elite solution to escape local optima.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Hyperparameters ---
    # Population size: Compromise between exploration and speed. 
    # SHADE typically uses 18*dim, clamped here for safety.
    pop_size = int(np.clip(18 * dim, 30, 100))
    archive_size = pop_size
    
    # Memory size for adaptive parameters
    H = 6
    
    # --- Initialization ---
    # Memory for F and CR (initialized to 0.5)
    mem_cr = np.full(H, 0.5)
    mem_f = np.full(H, 0.5)
    k_mem = 0  # Index for memory update

    # Process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize Population
    # Latin Hypercube Sampling-like or simple Random Uniform
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Archive (Empty initially)
    archive = np.empty((0, dim))

    # Global Best Tracking
    best_val = float('inf')
    best_vec = None

    # --- Initial Evaluation ---
    for i in range(pop_size):
        if check_time(): return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()

    # --- Main Loop ---
    while not check_time():
        
        # 1. Sort Population
        # Required for 'current-to-pbest' to identify top performers
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Sync best (index 0 is best due to sort)
        if fitness[0] < best_val:
            best_val = fitness[0]
            best_vec = pop[0].copy()

        # 2. Restart Mechanism
        # Check for stagnation (variance of fitness or population spatial spread)
        if np.std(fitness) < 1e-8 * (1 + abs(best_val)):
            # Retain elite, randomize the rest
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_vec
            fitness[:] = float('inf')
            fitness[0] = best_val
            
            # Reset Adaptive Memory
            mem_cr.fill(0.5)
            mem_f.fill(0.5)
            archive = np.empty((0, dim))
            
            # Re-evaluate
            for i in range(1, pop_size):
                if check_time(): return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            continue

        # 3. Parameter Generation
        # Assign a memory slot index to each individual
        r_idx = np.random.randint(0, H, pop_size)
        mu_cr = mem_cr[r_idx]
        mu_f = mem_f[r_idx]

        # Generate CR ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)

        # Generate F ~ Cauchy(mu_f, 0.1)
        # Standard Cauchy is loc=0, scale=1. We scale and shift.
        f = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        f[f > 1] = 1.0  # Clamp upper
        f[f <= 0] = 0.5 # Clamp lower (simple repair)

        # 4. Mutation: current-to-pbest/1
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # Select p-best (top 10% to 20%)
        p_val = 0.11
        top_p = max(2, int(pop_size * p_val))
        pbest_indices = np.random.randint(0, top_p, pop_size)
        x_pbest = pop[pbest_indices]

        # Select r1 (random from P)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = pop[r1_indices]

        # Select r2 (random from P U A)
        if archive.shape[0] > 0:
            # Union of population and archive
            pop_archive = np.vstack((pop, archive))
        else:
            pop_archive = pop
            
        r2_indices = np.random.randint(0, pop_archive.shape[0], pop_size)
        x_r2 = pop_archive[r2_indices]

        # Compute Mutant Vectors (Vectorized)
        f_col = f[:, np.newaxis]
        # pop is sorted, but we apply mutation to the whole population array in order
        mutants = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)

        # 5. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals < cr[:, np.newaxis]
        # Ensure at least one parameter comes from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        mask[np.arange(pop_size), j_rand] = True
        
        trials = np.where(mask, mutants, pop)
        
        # Bound Constraints (Clipping)
        trials = np.clip(trials, min_b, max_b)

        # 6. Selection and Memory Update Prep
        success_f = []
        success_cr = []
        improvement_diffs = []
        new_archive_candidates = []

        for i in range(pop_size):
            if check_time(): return best_val
            
            # Evaluate trial vector
            # Handle potential math errors in func nicely if needed, 
            # but template assumes func handles valid inputs.
            f_trial = func(trials[i])

            # Greedy Selection
            if f_trial <= fitness[i]:
                # If it's an improvement (strict), record for memory and archive
                if f_trial < fitness[i]:
                    new_archive_candidates.append(pop[i].copy())
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    improvement_diffs.append(fitness[i] - f_trial)
                
                # Replace parent
                pop[i] = trials[i]
                fitness[i] = f_trial
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trials[i].copy()

        # 7. Update Archive
        if len(new_archive_candidates) > 0:
            cands = np.array(new_archive_candidates)
            if archive.shape[0] == 0:
                archive = cands
            else:
                archive = np.vstack((archive, cands))
            
            # Resize if archive exceeds size limit
            if archive.shape[0] > archive_size:
                # Randomly remove excess
                keep_indices = np.random.choice(archive.shape[0], archive_size, replace=False)
                archive = archive[keep_indices]

        # 8. Update Historical Memory (SHADE logic)
        if len(improvement_diffs) > 0:
            # Calculate weights based on fitness improvement amount
            diffs = np.array(improvement_diffs)
            weights = diffs / np.sum(diffs)
            
            s_f = np.array(success_f)
            s_cr = np.array(success_cr)
            
            # Weighted Lehmer Mean for F
            mean_f = np.sum(weights * (s_f ** 2)) / np.sum(weights * s_f)
            
            # Weighted Mean for CR
            mean_cr = np.sum(weights * s_cr)
            
            # Update the memory slot
            mem_f[k_mem] = mean_f
            mem_cr[k_mem] = mean_cr
            
            # Increment memory pointer
            k_mem = (k_mem + 1) % H

    return best_val
