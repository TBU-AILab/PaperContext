#The following algorithm implements **L-SHADE-MTS (Linear Population Reduction SHADE with Multiple Trajectory Local Search)** adapted for time-constrained environments.
#
#**Why this is an improvement:**
#1.  **Time-Driven Linear Population Reduction (LPSR)**: Unlike standard DE which uses fixed populations, or algorithms that guess generation counts, this approach linearly reduces the population size from a large swarm (high exploration) to a minimal set (high exploitation) based on the *actual elapsed wall-clock time*. This maximizes the use of the computational budget.
#2.  **MTS-LS1 Local Search**: Differential Evolution is excellent at locating the global basin of attraction but can be slow to converge to high precision. This algorithm integrates a Coordinate Descent Local Search (MTS-LS1) that periodically refines the best solution found so far, significantly improving the fitness value in the final stages.
#3.  **Archive & Historical Memory**: It utilizes the specific SHADE mechanisms (Archive for diversity, weighted Lehmer mean for parameter adaptation) to learn the best mutation factors ($F$) and crossover rates ($CR$) dynamically.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Time-based Linear Population Reduction
    and MTS-LS1 Local Search.
    """
    start_time = datetime.now()
    # Reserve small buffer for safe return
    end_time = start_time + timedelta(seconds=max_time * 0.99)

    # --- Pre-process Bounds ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Global Best Tracking ---
    best_val = float('inf')
    best_vec = None

    # --- L-SHADE Configuration ---
    # Initial population size: High for exploration (approx 18*dim)
    # Capped at 200 to prevent slowness in high dimensions within limited time
    N_init = min(200, max(30, 18 * dim))
    N_min = 4  # Minimum population size for mutation/crossover
    
    # Historical Memory Size
    H = 6 
    
    # --- Helper: Local Search (MTS-LS1 Coordinate Descent) ---
    def local_search(current_best_vec, current_best_val, search_budget_time):
        """Refines the best solution by probing dimensions."""
        ls_start = datetime.now()
        ls_vec = current_best_vec.copy()
        ls_val = current_best_val
        
        # Search range (step size)
        sr = (max_b - min_b) * 0.4
        
        improved = False
        
        # We iterate through dimensions randomly
        dims = np.arange(dim)
        np.random.shuffle(dims)
        
        for j in dims:
            if datetime.now() - ls_start > timedelta(seconds=search_budget_time):
                break
            if datetime.now() >= end_time:
                break
                
            # Try negative move
            original = ls_vec[j]
            ls_vec[j] = np.clip(original - sr[j], min_b[j], max_b[j])
            val = func(ls_vec)
            
            if val < ls_val:
                ls_val = val
                improved = True
            else:
                # Restore and try positive move
                ls_vec[j] = np.clip(original + 0.5 * sr[j], min_b[j], max_b[j])
                val = func(ls_vec)
                
                if val < ls_val:
                    ls_val = val
                    improved = True
                else:
                    # Restore original
                    ls_vec[j] = original
        
        return ls_vec, ls_val, improved

    # --- Restart Loop ---
    # If the algorithm converges early (pop size reaches minimum or variance 0), 
    # we restart but keep the best solution (Elitism).
    while True:
        # Check total remaining time
        now = datetime.now()
        if now >= end_time:
            return best_val
            
        remaining_seconds = (end_time - now).total_seconds()
        # If very little time left, do a quick local search on best and return
        if remaining_seconds < 0.2:
            return best_val

        # Setup for this run
        run_start = now
        # We assume this "run" tries to use the remaining time
        # The population reduction schedule is mapped to this window
        run_total_time = remaining_seconds 
        
        # Initialization
        pop_size = N_init
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best into new population
        if best_vec is not None:
            population[0] = best_vec.copy()
            # Inject a few mutated clones of best to encourage search around it
            for k in range(1, 4):
                if k < pop_size:
                    population[k] = best_vec + np.random.normal(0, 0.05, dim) * diff_b
                    population[k] = np.clip(population[k], min_b, max_b)

        fitness = np.full(pop_size, float('inf'))
        
        # Memory for SHADE
        mem_f = np.full(H, 0.5)
        mem_cr = np.full(H, 0.5) # Start CR at 0.5
        k_mem = 0
        archive = []
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_val
            val = func(population[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_vec = population[i].copy()
        
        # --- Evolutionary Loop ---
        while True:
            t_now = datetime.now()
            if t_now >= end_time: return best_val
            
            # 1. Linear Population Size Reduction (LPSR) based on Time
            elapsed = (t_now - run_start).total_seconds()
            progress = min(1.0, elapsed / (run_total_time + 1e-9))
            
            # Calculate target population size
            target_pop = int(round(N_init - (N_init - N_min) * progress))
            target_pop = max(N_min, target_pop)
            
            # Reduce population if needed
            if target_pop < pop_size:
                # Sort by fitness (ascending) and keep best
                sorted_idx = np.argsort(fitness)
                population = population[sorted_idx[:target_pop]]
                fitness = fitness[sorted_idx[:target_pop]]
                pop_size = target_pop
                
                # Resize Archive (Maintain Archive size <= Pop size)
                if len(archive) > pop_size:
                    random.shuffle(archive)
                    archive = archive[:pop_size]

            # 2. Check for Stagnation / Restart Trigger
            # If population variance is negligible or we reached min pop size with no progress
            if pop_size == N_min or np.std(fitness) < 1e-9:
                # Perform a Local Search before restarting to squeeze value
                ls_budget = min(0.5, (end_time - datetime.now()).total_seconds())
                if ls_budget > 0.05:
                    b_vec, b_val, _ = local_search(best_vec, best_val, ls_budget)
                    if b_val < best_val:
                        best_val = b_val
                        best_vec = b_vec
                break # Break inner loop -> Restart

            # 3. Parameter Generation
            r_idx = np.random.randint(0, H, pop_size)
            m_f = mem_f[r_idx]
            m_cr = mem_cr[r_idx]
            
            # Generate CR ~ Normal(m_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(m_f, 0.1)
            f = m_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
            
            # Constraints for F
            # If F <= 0, regenerate until > 0
            neg_mask = f <= 0
            while np.any(neg_mask):
                f[neg_mask] = m_f[neg_mask] + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(neg_mask)) - 0.5))
                neg_mask = f <= 0
            f = np.minimum(f, 1.0) # Clip at 1.0
            
            # 4. Mutation: current-to-pbest/1
            # Sort for pbest selection
            sorted_indices = np.argsort(fitness)
            # p varies from 0.2 (exploration) to 0.1 (exploitation) based on progress
            p_val = 0.2 - 0.1 * progress 
            top_cnt = max(2, int(pop_size * p_val))
            pbest_indices = sorted_indices[:top_cnt]
            
            pbest_ind = np.random.choice(pbest_indices, pop_size)
            x_pbest = population[pbest_ind]
            
            # r1: random from population (distinct from i)
            r1_ind = np.random.randint(0, pop_size, pop_size)
            # Fix collisions
            cols = (r1_ind == np.arange(pop_size))
            while np.any(cols):
                r1_ind[cols] = np.random.randint(0, pop_size, np.sum(cols))
                cols = (r1_ind == np.arange(pop_size))
            x_r1 = population[r1_ind]
            
            # r2: random from Union(Population, Archive) (distinct from i and r1)
            if len(archive) > 0:
                archive_np = np.array(archive)
                union_pop = np.vstack((population, archive_np))
            else:
                union_pop = population
            
            union_size = len(union_pop)
            r2_ind = np.random.randint(0, union_size, pop_size)
            
            # Collision handling for r2
            cols = (r2_ind == np.arange(pop_size)) | (r2_ind == r1_ind)
            while np.any(cols):
                r2_ind[cols] = np.random.randint(0, union_size, np.sum(cols))
                cols = (r2_ind == np.arange(pop_size)) | (r2_ind == r1_ind)
            x_r2 = union_pop[r2_ind]
            
            f_col = f[:, np.newaxis]
            mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
            
            # 5. Crossover
            cross_mask = np.random.rand(pop_size, dim) < cr[:, np.newaxis]
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # 6. Selection and Updates
            # We can't vectorize func(trial_pop) directly, loop needed
            success_f = []
            success_cr = []
            diff_fitness = []
            
            # Arrays to bulk update population later
            new_fitness = fitness.copy()
            new_population = population.copy()
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_val
                
                f_trial = func(trial_pop[i])
                
                if f_trial < fitness[i]:
                    # Improved
                    new_population[i] = trial_pop[i]
                    new_fitness[i] = f_trial
                    
                    # Add old to archive
                    archive.append(population[i].copy())
                    
                    # Record success for memory update
                    success_f.append(f[i])
                    success_cr.append(cr[i])
                    diff_fitness.append(fitness[i] - f_trial)
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_vec = trial_pop[i].copy()
            
            population = new_population
            fitness = new_fitness
            
            # Maintain Archive Size
            while len(archive) > pop_size:
                idx_remove = random.randint(0, len(archive)-1)
                archive.pop(idx_remove)
            
            # 7. Update Historical Memory (Weighted Lehmer Mean)
            if len(diff_fitness) > 0:
                weights = np.array(diff_fitness)
                total_w = np.sum(weights)
                if total_w > 0:
                    weights /= total_w
                    
                    sf = np.array(success_f)
                    scr = np.array(success_cr)
                    
                    mean_f = np.sum(weights * (sf**2)) / (np.sum(weights * sf) + 1e-10)
                    mean_cr = np.sum(weights * (scr**2)) / (np.sum(weights * scr) + 1e-10)
                    
                    # Clip means
                    mean_cr = np.clip(mean_cr, 0, 1)
                    mean_f = np.clip(mean_f, 0, 1)
                    
                    mem_f[k_mem] = mean_f
                    mem_cr[k_mem] = mean_cr
                    k_mem = (k_mem + 1) % H

    return best_val
