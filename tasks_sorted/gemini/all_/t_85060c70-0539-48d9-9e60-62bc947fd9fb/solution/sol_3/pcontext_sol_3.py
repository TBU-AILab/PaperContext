#The following Python code implements the **L-SHADE** (Linear Population Size Reduction Success-History based Adaptive Differential Evolution) algorithm. This is a state-of-the-art improvement over the previously used SHADE algorithm.
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR)**: The population size linearly decreases from a large initial value to a small final value as time progresses. This ensures high exploration at the start and fine-grained exploitation at the end.
#2.  **Time-Aware Scheduling**: The algorithm continuously monitors the elapsed time to adjust the population size and strictly strictly respects the `max_time` constraint.
#3.  **Adaptive Parameters**: Maintains historical memory of successful crossover ($CR$) and mutation ($F$) rates to adapt to the objective function landscape.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE (Linear Population Size Reduction 
    Success-History based Adaptive Differential Evolution).
    """
    
    # --- 1. Time Management ---
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    def check_timeout():
        return datetime.now() >= end_time

    def get_time_ratio():
        # Returns value between 0.0 (start) and 1.0 (end)
        elapsed = (datetime.now() - start_time).total_seconds()
        return min(elapsed / max_time, 1.0)

    # --- 2. Configuration ---
    # Initial Population Size (N_init)
    # Heuristic: 18 * dim. We clamp it between [30, 200] to handle time constraints gracefully.
    N_init = int(np.clip(18 * dim, 30, 200))
    
    # Final Population Size (N_min)
    # Small enough to converge, large enough for mutation strategies.
    N_min = 6 
    
    # Archive Size Parameter
    arc_rate = 1.4 # Archive size = 1.4 * Current Population Size

    # Memory for Adaptive Parameters (H)
    H = 5
    mem_cr = np.full(H, 0.5) # Memory for Crossover Probability
    mem_f = np.full(H, 0.5)  # Memory for Mutation Factor
    k_mem = 0                # Memory pointer

    # --- 3. Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(N_init, dim) * diff_b
    fitness = np.full(N_init, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None
    
    # External Archive
    archive = []

    # Evaluate Initial Population
    for i in range(N_init):
        if check_timeout(): return best_fitness
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()

    # --- 4. Main Loop ---
    while not check_timeout():
        
        # A. Linear Population Size Reduction
        # Calculate target population size based on elapsed time
        t_ratio = get_time_ratio()
        N_target = int(round(((N_min - N_init) * t_ratio) + N_init))
        if N_target < N_min: N_target = N_min
        
        curr_N = len(pop)
        
        # Reduce Population if needed
        if curr_N > N_target:
            # Sort population by fitness (worst at the end)
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Truncate population (remove worst individuals)
            pop = pop[:N_target]
            fitness = fitness[:N_target]
            curr_N = N_target
            
            # Reduce Archive Size accordingly
            max_arc_size = int(curr_N * arc_rate)
            while len(archive) > max_arc_size:
                # Randomly remove elements from archive
                archive.pop(np.random.randint(len(archive)))

        # B. Parameter Generation
        # Assign random memory index to each individual
        r_idx = np.random.randint(0, H, curr_N)
        
        # Generate CR ~ Normal(mem_cr, 0.1)
        cr = np.random.normal(mem_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(mem_f, 0.1)
        # Cauchy = location + scale * tan(pi * (rand - 0.5))
        f = mem_f[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand(curr_N) - 0.5))
        
        # Sanitize F
        f = np.where(f > 1.0, 1.0, f)
        f = np.where(f <= 0.0, 0.5, f) # Fallback to 0.5 if invalid
        
        # C. Evolution Cycle
        # Sort indices to find top p-best individuals
        sorted_indices = np.argsort(fitness)
        
        # p-best parameter (top 11% is a robust standard)
        p_val = 0.11
        top_count = max(2, int(curr_N * p_val))
        
        # Storage for successful updates
        succ_cr = []
        succ_f = []
        diff_fitness = []
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        # Prepare Union of Population + Archive for mutation
        if len(archive) > 0:
            # Stacking is feasible for these population sizes
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
        union_size = len(union_pop)
        
        # Iterate over individuals
        for i in range(curr_N):
            if check_timeout(): return best_fitness
            
            # 1. Mutation: DE/current-to-pbest/1
            # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
            
            # Select p-best (randomly from top p%)
            pbest_idx = sorted_indices[np.random.randint(0, top_count)]
            x_pbest = pop[pbest_idx]
            
            # Select r1 (distinct from i)
            r1 = i
            while r1 == i:
                r1 = np.random.randint(0, curr_N)
            x_r1 = pop[r1]
            
            # Select r2 (distinct from i and r1, from Union)
            # Using simple rejection sampling
            while True:
                r2_idx = np.random.randint(0, union_size)
                if r2_idx == i or r2_idx == r1:
                    continue
                x_r2 = union_pop[r2_idx]
                break
            
            # Compute Mutant
            mutant = pop[i] + f[i] * (x_pbest - pop[i]) + f[i] * (x_r1 - x_r2)
            
            # 2. Crossover (Binomial)
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < cr[i]
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, pop[i])
            
            # 3. Bound Constraint
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            trial_val = func(trial)
            
            if trial_val <= fitness[i]:
                # Update Next Generation
                new_pop[i] = trial
                new_fitness[i] = trial_val
                
                if trial_val < fitness[i]:
                    # Strict improvement: Update Archive and Success Memory
                    archive.append(pop[i].copy())
                    succ_cr.append(cr[i])
                    succ_f.append(f[i])
                    diff_fitness.append(fitness[i] - trial_val)
                
                # Update Global Best
                if trial_val < best_fitness:
                    best_fitness = trial_val
                    best_sol = trial.copy()
        
        # Apply updates
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size (Random Removal)
        max_arc_size = int(curr_N * arc_rate)
        while len(archive) > max_arc_size:
            archive.pop(np.random.randint(len(archive)))
            
        # D. Update Historical Memory
        if len(succ_cr) > 0:
            succ_cr = np.array(succ_cr)
            succ_f = np.array(succ_f)
            diff_fitness = np.array(diff_fitness)
            
            # Calculate weights based on improvement magnitude
            total_diff = np.sum(diff_fitness)
            if total_diff == 0:
                weights = np.ones(len(diff_fitness)) / len(diff_fitness)
            else:
                weights = diff_fitness / total_diff
            
            # Update Memory CR (Weighted Mean)
            mean_cr = np.sum(weights * succ_cr)
            mem_cr[k_mem] = mean_cr
            
            # Update Memory F (Weighted Lehmer Mean)
            denom = np.sum(weights * succ_f)
            if denom == 0:
                mean_f = 0.5
            else:
                mean_f = np.sum(weights * (succ_f ** 2)) / denom
            mem_f[k_mem] = mean_f
            
            # Advance memory pointer
            k_mem = (k_mem + 1) % H

    return best_fitness
