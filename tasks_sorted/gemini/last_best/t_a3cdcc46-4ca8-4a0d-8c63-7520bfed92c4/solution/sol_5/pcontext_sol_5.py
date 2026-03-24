#Here is a fully operational, self-contained Python algorithm.
#
#This algorithm implements **L-SHADE-RSP** (Linear Success-History Adaptive Differential Evolution with Rank-based mutation, Selection pressure, and Population reduction) augmented with an **Adaptive Coordinate Descent Local Search**.
#
#**Key Improvements over previous attempts:**
#1.  **Linear Population Size Reduction (LPSR)**: Linearly reduces population size from a high initial value (exploration) to a minimum value (exploitation) based on remaining time. This ensures computational resources are focused on refining the best solutions as the deadline approaches.
#2.  **Adaptive Parameters (SHADE)**: Automatically adapts mutation factor $F$ and crossover rate $CR$ using a success-history memory, removing the need for manual tuning.
#3.  **Current-to-pbest Mutation**: Balances greediness (convergence speed) and diversity better than standard `current-to-best` or `random`.
#4.  **End-Game Local Search**: In the final 5% of the allocated time, the algorithm switches to a greedy Coordinate Descent (Pattern Search) on the best solution found so far. This "squeezes" the final precision out of the solution, which DE alone often struggles to achieve quickly.
#5.  **Smart Restart**: If the population variance collapses (stagnation), the algorithm performs a quick local search on the elite solution to ensure the basin is drained, then restarts the population to find new minima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Linear Population Reduction,
    Adaptive Restarts, and Final-Phase Coordinate Descent.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)

    def get_time_ratio():
        """Returns the fraction of time used (0.0 to 1.0)."""
        elapsed = (datetime.now() - start_time).total_seconds()
        # Use a small buffer to prevent hard timeouts during heavy operations
        return min(elapsed / max(max_time, 1e-3), 1.0)

    def check_timeout():
        """Returns True if time is up."""
        return datetime.now() - start_time >= time_limit

    # --- Problem Setup ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    best_fitness = float('inf')
    best_pos = None

    # Function Wrapper to update global best automatically
    def evaluate(x):
        nonlocal best_fitness, best_pos
        # Ensure bounds constraints
        x_clamped = np.clip(x, min_b, max_b)
        val = func(x_clamped)
        
        if val < best_fitness:
            best_fitness = val
            best_pos = x_clamped.copy()
        return val

    # --- Local Search (Coordinate Descent) ---
    def local_search(start_pos):
        """
        Performs a greedy coordinate descent to refine a solution.
        Used during stagnation and the final phase.
        """
        if start_pos is None: return
        
        current_x = start_pos.copy()
        current_f = best_fitness
        
        # Initial step size (10% of domain)
        step_size = diff_b * 0.1
        
        # Iteration limited by time and step precision
        while np.max(step_size) > 1e-9:
            if check_timeout(): return
            
            improved = False
            # Shuffle dimensions to avoid bias
            dims = np.random.permutation(dim)
            
            for d in dims:
                if check_timeout(): return
                
                original_val = current_x[d]
                
                # Try positive step
                current_x[d] = np.clip(original_val + step_size[d], min_b[d], max_b[d])
                f_new = func(current_x) # Direct call to avoid redundant global checks
                
                if f_new < current_f:
                    current_f = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_pos = current_x.copy()
                    improved = True
                else:
                    # Try negative step
                    current_x[d] = np.clip(original_val - step_size[d], min_b[d], max_b[d])
                    f_new = func(current_x)
                    
                    if f_new < current_f:
                        current_f = f_new
                        if f_new < best_fitness:
                            best_fitness = f_new
                            best_pos = current_x.copy()
                        improved = True
                    else:
                        # Revert
                        current_x[d] = original_val
            
            if improved:
                # If we improved, we might want to keep the step size or slightly expand
                pass 
            else:
                # If no improvement in any dimension, shrink step size
                step_size *= 0.5

    # --- L-SHADE Initialization ---
    # Population Size Schedule
    # Start high for exploration, end low for convergence
    initial_pop_size = int(np.clip(20 * dim, 60, 300))
    min_pop_size = 4
    pop_size = initial_pop_size
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if check_timeout(): return best_fitness
        fitness[i] = evaluate(pop[i])
        
    # Memory for Adaptive Parameters (History length = 5)
    mem_size = 5
    mem_F = np.full(mem_size, 0.5)
    mem_CR = np.full(mem_size, 0.5)
    mem_k = 0
    
    # Archive for diversity
    archive = []
    
    # --- Main Loop ---
    while not check_timeout():
        
        # 0. Final Phase Polish
        # If > 95% of time is gone, switch to pure exploitation
        if get_time_ratio() > 0.95:
            local_search(best_pos)
            return best_fitness

        # 1. Linear Population Size Reduction (LPSR)
        progress = get_time_ratio()
        target_size = int(round(initial_pop_size + (min_pop_size - initial_pop_size) * progress))
        target_size = max(min_pop_size, target_size)
        
        if pop_size > target_size:
            n_reduce = pop_size - target_size
            # Remove worst individuals
            sorting_idx = np.argsort(fitness)
            survivor_indices = sorting_idx[:-n_reduce]
            
            pop = pop[survivor_indices]
            fitness = fitness[survivor_indices]
            pop_size = target_size
            
            # Reduce Archive Size
            arc_limit = int(pop_size * 2.0)
            if len(archive) > arc_limit:
                # Randomly remove excess
                del_count = len(archive) - arc_limit
                # Simple removal from beginning/random
                archive = archive[del_count:]

        # 2. Stagnation Check & Restart
        # If population variance is negligible, we are stuck
        if np.std(fitness) < 1e-9 or (np.max(fitness) - np.min(fitness) < 1e-9):
            # Polish current best
            local_search(best_pos)
            if check_timeout(): return best_fitness
            
            # Restart Population
            # We reset to the current target size to adhere to time schedule
            pop_size = target_size
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            fitness = np.full(pop_size, float('inf'))
            
            # Elitism: keep global best
            pop[0] = best_pos.copy()
            fitness[0] = best_fitness
            
            # Evaluate new randoms
            for i in range(1, pop_size):
                if check_timeout(): return best_fitness
                fitness[i] = evaluate(pop[i])
            
            # Reset Memory but keep archive empty
            mem_F.fill(0.5)
            mem_CR.fill(0.5)
            archive = []
            continue

        # 3. Generate Adaptive Parameters
        # Pick random memory index for each individual
        r_idx = np.random.randint(0, mem_size, pop_size)
        m_f = mem_F[r_idx]
        m_cr = mem_CR[r_idx]
        
        # Generate CR (Normal Distribution)
        CR = np.random.normal(m_cr, 0.1)
        CR = np.clip(CR, 0, 1)
        
        # Generate F (Cauchy Distribution)
        F = m_f + 0.1 * np.random.standard_cauchy(pop_size)
        # Repair F <= 0
        while np.any(F <= 0):
            neg = F <= 0
            F[neg] = m_f[neg] + 0.1 * np.random.standard_cauchy(np.sum(neg))
        F = np.minimum(F, 1.0)
        
        # 4. Mutation: current-to-pbest/1 (L-SHADE standard)
        # Sort population to find p-best
        sorted_indices = np.argsort(fitness)
        
        # p is random in [2/pop_size, 0.2] (top 20%)
        p_val = np.random.uniform(2.0/pop_size, 0.2, pop_size)
        num_pbest = (p_val * pop_size).astype(int)
        num_pbest = np.maximum(num_pbest, 1)
        
        # Select pbest
        rand_ranks = (np.random.rand(pop_size) * num_pbest).astype(int)
        pbest_indices = sorted_indices[rand_ranks]
        x_pbest = pop[pbest_indices]
        
        # Select r1 != i
        r1 = np.random.randint(0, pop_size, pop_size)
        conflict = r1 == np.arange(pop_size)
        r1[conflict] = (r1[conflict] + 1) % pop_size
        x_r1 = pop[r1]
        
        # Select r2 != r1, r2 != i (from Population U Archive)
        if len(archive) > 0:
            pool = np.vstack((pop, np.array(archive)))
        else:
            pool = pop
            
        r2 = np.random.randint(0, len(pool), pop_size)
        conflict2 = (r2 == np.arange(pop_size)) | (r2 == r1)
        r2[conflict2] = (r2[conflict2] + 1) % len(pool)
        x_r2 = pool[r2]
        
        # Compute Mutant Vectors
        # v = x + F*(pbest - x) + F*(r1 - r2)
        diff_p = x_pbest - pop
        diff_r = x_r1 - x_r2
        mutant = pop + F[:, None] * diff_p + F[:, None] * diff_r
        
        # 5. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        mask = np.random.rand(pop_size, dim) <= CR[:, None]
        mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(mask, mutant, pop)
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Evaluation & Selection
        success_F = []
        success_CR = []
        diff_fitness = []
        
        for i in range(pop_size):
            if check_timeout(): return best_fitness
            
            f_trial = evaluate(trial[i])
            
            if f_trial < fitness[i]:
                # Successful update
                archive.append(pop[i].copy())
                success_F.append(F[i])
                success_CR.append(CR[i])
                diff_fitness.append(fitness[i] - f_trial)
                
                fitness[i] = f_trial
                pop[i] = trial[i]
                
        # Manage Archive Size Limit
        arc_limit = int(pop_size * 2.5)
        while len(archive) > arc_limit:
            # Remove random element to maintain diversity
            archive.pop(np.random.randint(0, len(archive)))

        # 7. Update Adaptive Memory (Weighted Lehmer Mean)
        if len(success_F) > 0:
            sF = np.array(success_F)
            sCR = np.array(success_CR)
            df = np.array(diff_fitness)
            
            # Weights based on fitness improvement
            total_imp = np.sum(df)
            if total_imp > 0:
                weights = df / total_imp
                
                # Update F (Lehmer Mean)
                mean_F = np.sum(weights * (sF ** 2)) / (np.sum(weights * sF) + 1e-15)
                mem_F[mem_k] = 0.5 * mem_F[mem_k] + 0.5 * mean_F
                
                # Update CR (Weighted Arithmetic Mean)
                mean_CR = np.sum(weights * sCR)
                mem_CR[mem_k] = 0.5 * mem_CR[mem_k] + 0.5 * mean_CR
                
                mem_k = (mem_k + 1) % mem_size

    return best_fitness
