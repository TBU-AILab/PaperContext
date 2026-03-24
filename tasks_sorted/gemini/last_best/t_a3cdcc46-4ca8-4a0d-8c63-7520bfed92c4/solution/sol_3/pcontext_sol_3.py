#The output value of the last generated algorithm is: 34.4160096580172
#The output value of the best generated algorithm is: 21.461227845749107
#
#The best generated algorithm code:
#Here is a fully operational, self-contained Python algorithm.
#
#This algorithm implements **L-SHADE with Linear Population Size Reduction (LPSR) and Restarts**, 
#augmented with a final **Coordinate Descent** polishing phase.
#
#**Why this improves results:**
#1.  **L-SHADE**: It is a top-performing evolutionary algorithm that adapts Mutation ($F$) and Crossover ($CR$) rates based on historical success, avoiding manual parameter tuning.
#2.  **LPSR (Linear Population Size Reduction)**: It dynamically reduces the population size as time progresses. This allows for broad exploration early on and focused exploitation later, matching the "limited time" constraint perfectly.
#3.  **Adaptive Restart**: If the population converges/stagnates too early (while time remains), it triggers a restart with a new population (while preserving the global best) to search different basins of attraction. The restart size scales with remaining time due to LPSR logic.
#4.  **Coordinate Polish**: Uses the final 5% of the time budget to perform a deterministic coordinate descent pattern search, ensuring the final solution is refined to the maximum possible precision.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with LPSR, Adaptive Restarts, and Pattern Search.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Reserve 5% of time for final polishing (Pattern Search)
    polish_start_ratio = 0.95 
    
    def get_time_ratio():
        # Returns 0.0 to 1.0 representing consumption of max_time
        elapsed = (datetime.now() - start_time).total_seconds()
        return elapsed / max_time
        
    def check_timelimit():
        return datetime.now() - start_time >= time_limit

    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Best found so far
    best_fitness = float('inf')
    best_pos = None
    
    # --- L-SHADE Parameters ---
    # Initial population size: 18 * dim (standard heuristic for L-SHADE)
    # Clamped to reasonable limits for performance
    init_pop_size = int(np.clip(18 * dim, 20, 300))
    min_pop_size = 4
    
    # External Archive (stores inferior solutions to maintain diversity)
    archive = []
    
    # Memory for adaptive parameters (History length H=5)
    memory_size = 5
    memory_sf = np.full(memory_size, 0.5)
    memory_scr = np.full(memory_size, 0.5)
    memory_pos = 0
    
    # Initialize Population
    pop_size = init_pop_size
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initial Evaluation
    for i in range(pop_size):
        if check_timelimit():
            return best_fitness if best_pos is not None else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_pos = pop[i].copy()
            
    # --- Main Optimization Loop ---
    # Runs until 95% of time is consumed, leaving 5% for polish
    while get_time_ratio() < polish_start_ratio:
        
        # 1. Generate Adaptive Parameters (F and CR)
        # Select random memory index for each individual
        r_idx = np.random.randint(0, memory_size, pop_size)
        
        # Generate CR: Normal distribution around memory
        cr = np.random.normal(memory_scr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F: Cauchy distribution
        # If F <= 0, regenerate. If F > 1, clamp to 1.
        f = np.random.standard_cauchy(pop_size) * 0.1 + memory_sf[r_idx]
        
        # Robust fix for invalid F values
        while np.any(f <= 0):
            mask = f <= 0
            f[mask] = np.random.standard_cauchy(np.sum(mask)) * 0.1 + memory_sf[r_idx][mask]
        f = np.clip(f, 0, 1)
        
        # 2. Sort population for p-best selection
        sorted_indices = np.argsort(fitness)
        pop_sorted = pop[sorted_indices]
        
        # Standard L-SHADE: p is fixed at top 11% (or similar)
        p_best_rate = 0.11
        num_pbest = max(2, int(p_best_rate * pop_size))
        
        # 3. Mutation: DE/current-to-pbest/1
        # v = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
        
        # Select pbest for each individual randomly from top p%
        pbest_indices = np.random.randint(0, num_pbest, pop_size)
        x_pbest = pop_sorted[pbest_indices]
        
        # Create Union of Population and Archive for r2 selection
        if len(archive) > 0:
            archive_np = np.array(archive)
            pool = np.vstack((pop, archive_np))
        else:
            pool = pop
            
        # Select r1 (from Pop) and r2 (from Pool)
        idxs = np.arange(pop_size)
        
        # r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        conflict_r1 = (r1 == idxs)
        r1[conflict_r1] = (r1[conflict_r1] + 1) % pop_size
        x_r1 = pop[r1]
        
        # r2 != i, r2 != r1
        r2 = np.random.randint(0, len(pool), pop_size)
        x_r2 = pool[r2]
        
        # Compute Mutant Vector V
        # F needs column shape for broadcasting
        f_col = f[:, None]
        v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # Binomial Crossover
        mask_rand = np.random.rand(pop_size, dim)
        cross_mask = mask_rand < cr[:, None]
        # Force at least one dimension
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[idxs, j_rand] = True
        
        u = np.where(cross_mask, v, pop)
        
        # Boundary Handling (Clamping)
        u = np.clip(u, min_b, max_b)
        
        # 4. Selection and Evaluation
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        successful_f = []
        successful_cr = []
        diff_fitness = []
        
        for i in range(pop_size):
            if check_timelimit(): return best_fitness
            
            f_new = func(u[i])
            
            if f_new < fitness[i]:
                new_pop[i] = u[i]
                new_fitness[i] = f_new
                
                # Record success for adaptation
                successful_f.append(f[i])
                successful_cr.append(cr[i])
                diff_fitness.append(fitness[i] - f_new)
                
                # Add old solution to archive
                archive.append(pop[i].copy())
                
                # Update Global Best
                if f_new < best_fitness:
                    best_fitness = f_new
                    best_pos = u[i].copy()
                    
        pop = new_pop
        fitness = new_fitness
        
        # Maintain Archive Size (<= pop_size)
        while len(archive) > pop_size:
            idx_del = np.random.randint(0, len(archive))
            del archive[idx_del]
            
        # 5. Update Historical Memories (Adaptive logic)
        if len(successful_f) > 0:
            scr = np.array(successful_cr)
            sf = np.array(successful_f)
            df = np.array(diff_fitness)
            
            # Weighted by improvement magnitude
            if np.sum(df) > 0:
                w = df / np.sum(df)
            else:
                w = np.ones(len(df)) / len(df)
                
            # Weighted Lehmer Mean for F
            mean_pow2 = np.sum(w * (sf ** 2))
            mean_pow1 = np.sum(w * sf)
            mean_sf = mean_pow2 / mean_pow1 if mean_pow1 > 0 else 0.5
            
            memory_sf[memory_pos] = 0.5 * memory_sf[memory_pos] + 0.5 * mean_sf
            
            # Weighted Mean for CR
            mean_scr = np.sum(w * scr)
            memory_scr[memory_pos] = 0.5 * memory_scr[memory_pos] + 0.5 * mean_scr
            
            memory_pos = (memory_pos + 1) % memory_size
            
        # 6. Linear Population Size Reduction (LPSR)
        # Calculates target population size based on remaining time
        t_ratio = get_time_ratio()
        remaining_ratio = 1.0 - t_ratio
        if remaining_ratio < 0: remaining_ratio = 0
        
        # N_next = N_min + (N_init - N_min) * (RemainingTime / TotalTime)
        target_pop_size = int(round(min_pop_size + (init_pop_size - min_pop_size) * remaining_ratio))
        
        if pop_size > target_pop_size:
            reduce_count = pop_size - target_pop_size
            # Identify worst individuals
            worst_indices = np.argsort(fitness)[-reduce_count:]
            # Sort indices descending to delete correctly without shifting
            worst_indices = -np.sort(-worst_indices) 
            
            pop = np.delete(pop, worst_indices, axis=0)
            fitness = np.delete(fitness, worst_indices, axis=0)
            pop_size = target_pop_size
            
            # Shrink archive to match new pop size
            while len(archive) > pop_size:
                del archive[np.random.randint(0, len(archive))]
                
        # 7. Restart Mechanism
        # If population is minimal and we still have significant time (e.g., < 75% used),
        # or if diversity is lost (std dev approx 0), we restart.
        # Note: LPSR normally prevents small pop until the end, so small pop here implies
        # we are conceptually "done" with the schedule. Restarting allows a new schedule
        # scaled to the remaining short time.
        if (pop_size <= min_pop_size + 2) and (t_ratio < 0.75):
            # Reset Population Size
            pop_size = init_pop_size 
            
            # Initialize new population (Exploration)
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Elitism: Inject the global best into the new population
            if best_pos is not None:
                pop[0] = best_pos.copy()
                
            fitness = np.full(pop_size, float('inf'))
            archive = [] # Clear archive
            
            # Soft reset memories (bias towards neutral but keep some history)
            memory_sf = 0.5 * memory_sf + 0.25
            memory_scr = 0.5 * memory_scr + 0.25
            
            # Evaluate new population (skip index 0 as it is best_pos)
            fitness[0] = best_fitness
            for i in range(1, pop_size):
                if check_timelimit(): return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_pos = pop[i].copy()
                    
    # --- Phase 2: Coordinate Descent Polish ---
    # Use the remaining ~5% time to greedily refine the best solution
    if best_pos is not None:
        # Start step size: small fraction of bounds
        step_sizes = diff_b * 0.005 
        current_x = best_pos.copy()
        current_f = best_fitness
        
        # Coordinate Descent Loop
        while not check_timelimit():
            improved_in_cycle = False
            for d in range(dim):
                if check_timelimit(): break
                
                original_val = current_x[d]
                
                # Test positive step
                current_x[d] = np.clip(original_val + step_sizes[d], min_b[d], max_b[d])
                f_new = func(current_x)
                
                if f_new < current_f:
                    current_f = f_new
                    improved_in_cycle = True
                else:
                    # Test negative step
                    current_x[d] = np.clip(original_val - step_sizes[d], min_b[d], max_b[d])
                    f_new = func(current_x)
                    if f_new < current_f:
                        current_f = f_new
                        improved_in_cycle = True
                    else:
                        # Revert
                        current_x[d] = original_val
            
            if improved_in_cycle:
                best_fitness = current_f
                best_pos = current_x.copy()
            else:
                # Decrease step size if no improvement in full cycle
                step_sizes *= 0.5
                if np.max(step_sizes) < 1e-9: # Limit precision
                    break

    return best_fitness
