#Here is a fully operational, self-contained Python algorithm. 
#
#This approach uses **JADE (Adaptive Differential Evolution with Optional External Archive)** combined with **Linear Population Size Reduction** and a final **Pattern Search (Hooke-Jeeves)** polishing phase.
#
#**Why this is better:**
#1.  **Parameter Adaptation:** Unlike standard DE which uses fixed or random mutation ($F$) and crossover ($CR$) rates, JADE adapts these parameters based on successful evolution steps. This removes the need to guess hyperparameters.
#2.  **Archive:** It maintains an archive of inferior solutions to preserve diversity and prevent premature convergence, allowing the algorithm to learn from "second-best" areas.
#3.  **Population Reduction:** It starts with a larger population to explore the global space and linearly reduces the population size over time to concentrate computational power on the most promising regions.
#4.  **Deterministic Polishing:** In the final phase, it switches to a coordinate-based pattern search to refine the best solution with high precision, which is often more effective than random hill-climbing for the final descent.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using JADE (Adaptive Differential Evolution) 
    with Linear Population Size Reduction and Coordinate Pattern Search.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def get_elapsed_seconds():
        return (datetime.now() - start_time).total_seconds()

    # --- Initialization ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Hyperparameters ---
    # Initial population size (larger to explore)
    initial_pop_size = int(round(18 * dim)) 
    # Minimum population size (smaller to exploit)
    min_pop_size = 4 
    
    current_pop_size = initial_pop_size
    
    # Adaptive Memory (JADE)
    mu_cr = 0.5  # Mean Crossover Rate
    mu_f = 0.5   # Mean Mutation Factor
    c = 0.1      # Learning rate for parameter adaptation
    p_best_rate = 0.05 # Top percentage to select for mutation
    
    # Archive for diversity
    archive = []
    
    # Initialize Population
    pop = min_b + np.random.rand(current_pop_size, dim) * diff_b
    fitness = np.full(current_pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_pos = None
    
    # Initial Evaluation
    for i in range(current_pop_size):
        if get_elapsed_seconds() >= max_time:
            return best_fitness if best_pos is not None else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_pos = pop[i].copy()
            
    # Max generations estimation (dynamic)
    # We estimate based on first generation time, but update dynamically
    gen_count = 0
    max_gens_est = 1000 # Placeholder, will be ignored/updates
    
    while True:
        elapsed = get_elapsed_seconds()
        
        # --- Phase 2: End Game Pattern Search (Last 10% of time) ---
        if elapsed >= max_time * 0.90:
            # Switch to Coordinate Descent (Hooke-Jeeves simplified)
            # This is deterministic and efficient for final polishing
            step_size = diff_b * 0.005 # Start small
            
            while get_elapsed_seconds() < max_time:
                improved = False
                for d in range(dim):
                    if get_elapsed_seconds() >= max_time: break
                    
                    # Try positive step
                    old_val = best_pos[d]
                    best_pos[d] = np.clip(old_val + step_size[d], min_b[d], max_b[d])
                    f_new = func(best_pos)
                    
                    if f_new < best_fitness:
                        best_fitness = f_new
                        improved = True
                    else:
                        # Try negative step
                        best_pos[d] = np.clip(old_val - step_size[d], min_b[d], max_b[d])
                        f_new = func(best_pos)
                        if f_new < best_fitness:
                            best_fitness = f_new
                            improved = True
                        else:
                            # Revert
                            best_pos[d] = old_val
                
                if not improved:
                    step_size *= 0.5 # Shrink steps
                    if np.max(step_size) < 1e-9: # Converged precision
                         break
                
            return best_fitness

        # --- Phase 1: Adaptive Differential Evolution ---
        
        # 1. Generate Parameters (F and CR) for this generation
        # Cauchy distribution for F, Normal for CR
        cr_g = np.random.normal(mu_cr, 0.1, current_pop_size)
        cr_g = np.clip(cr_g, 0, 1)
        
        # F generation with retry for invalid values
        f_g = np.random.standard_cauchy(current_pop_size) * 0.1 + mu_f
        # Logic to regenerate F if <= 0 and cap at 1
        while np.any(f_g <= 0):
            mask = f_g <= 0
            f_g[mask] = np.random.standard_cauchy(np.sum(mask)) * 0.1 + mu_f
        f_g = np.clip(f_g, 0, 1) # Conventionally capped at 1.0 (though JADE allows >1)

        # 2. Mutation: DE/current-to-pbest/1
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        p_num = max(1, int(p_best_rate * current_pop_size))
        p_best_indices = sorted_indices[:p_num]
        
        trials = np.zeros_like(pop)
        
        # Lists to store successful parameters
        succ_scr = []
        succ_sf = []
        
        # Create pool (Population + Archive) for r2 selection
        if len(archive) > 0:
            archive_np = np.array(archive)
            pool = np.vstack((pop, archive_np))
        else:
            pool = pop
            
        for i in range(current_pop_size):
            # Check time strictly inside the loop
            if get_elapsed_seconds() >= max_time:
                return best_fitness
            
            # Select r1 != i
            r1 = np.random.randint(0, current_pop_size)
            while r1 == i:
                r1 = np.random.randint(0, current_pop_size)
                
            # Select r2 != i, != r1 from Pool
            r2 = np.random.randint(0, len(pool))
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, len(pool))
                
            # Select p_best randomly from top p%
            r_pbest = np.random.choice(p_best_indices)
            
            # Mutation Vector
            # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            mutant = pop[i] + f_g[i] * (pop[r_pbest] - pop[i]) + \
                     f_g[i] * (pop[r1] - pool[r2])
            
            # Boundary correction (reflection or clamp)
            # Clamping is safer for strict bounds
            mutant = np.clip(mutant, min_b, max_b)
            
            # Crossover (Binomial)
            cross_points = np.random.rand(dim) < cr_g[i]
            j_rand = np.random.randint(0, dim)
            cross_points[j_rand] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            
            f_trial = func(trial)
            
            # Selection
            if f_trial < fitness[i]:
                # Add old solution to archive
                archive.append(pop[i].copy())
                
                fitness[i] = f_trial
                pop[i] = trial
                succ_scr.append(cr_g[i])
                succ_sf.append(f_g[i])
                
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_pos = trial.copy()
            
        # Manage Archive Size (Keep size roughly equal to pop size)
        while len(archive) > current_pop_size:
            # Randomly remove to keep diversity
            del archive[np.random.randint(0, len(archive))]
            
        # 3. Update Adaptive Parameters (Lehman Mean)
        if len(succ_scr) > 0:
            succ_scr = np.array(succ_scr)
            succ_sf = np.array(succ_sf)
            
            # Mean of CR
            mu_cr = (1 - c) * mu_cr + c * np.mean(succ_scr)
            
            # Lehmer Mean of F
            mean_pow2 = np.mean(succ_sf ** 2)
            mean_pow1 = np.mean(succ_sf)
            if mean_pow1 > 0:
                lehmer_mean = mean_pow2 / mean_pow1
                mu_f = (1 - c) * mu_f + c * lehmer_mean
        
        # 4. Linear Population Size Reduction (LPSR)
        # Calculate N_next based on time
        rem_time_ratio = 1.0 - (elapsed / max_time)
        # Avoid division by zero or negative logic close to end
        if rem_time_ratio < 0: rem_time_ratio = 0
        
        # Target size linearly decreases
        plan_pop_size = int(round(min_pop_size + (initial_pop_size - min_pop_size) * rem_time_ratio))
        
        if current_pop_size > plan_pop_size:
            # Remove worst individuals
            num_to_remove = current_pop_size - plan_pop_size
            worst_indices = np.argsort(fitness)[-num_to_remove:]
            
            # Delete indices (need to be sorted descending to delete correctly)
            pop = np.delete(pop, worst_indices, axis=0)
            fitness = np.delete(fitness, worst_indices, axis=0)
            current_pop_size = plan_pop_size

        gen_count += 1
        
    return best_fitness
