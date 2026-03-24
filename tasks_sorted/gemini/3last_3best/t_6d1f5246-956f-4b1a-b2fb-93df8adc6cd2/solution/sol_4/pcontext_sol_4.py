#The following Python code implements a **Restarting JADE (Adaptive Differential Evolution) with Archive and Local Search**. 
#
#This algorithm improves upon standard Differential Evolution and basic JADE by incorporating:
#1.  **Self-Adaptation**: Automatically tunes crossover ($CR$) and mutation ($F$) parameters using success history.
#2.  **External Archive**: Maintains a history of good, recently replaced solutions to maintain diversity and prevent premature convergence.
#3.  **Bound Handling**: Uses a "bounce-back" midpoint strategy instead of simple clipping, which preserves evolutionary direction near boundaries.
#4.  **Local Polish**: Triggers a lightweight coordinate descent (pattern search) when the population converges, aiming to refine the solution to high precision before restarting.
#5.  **Restart Mechanism**: Automatically resets the population if convergence is detected or stagnation occurs, maximizing the use of the available time budget.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting JADE (Self-Adaptive Differential Evolution) 
    with an external archive and a local search polishing step.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # -------------------------------------------------------------------------
    # Helper: Time Check
    # -------------------------------------------------------------------------
    def check_time():
        return datetime.now() - start_time >= time_limit

    # -------------------------------------------------------------------------
    # Algorithm Configuration
    # -------------------------------------------------------------------------
    # Population size: Adaptive based on dimension.
    # JADE works well with NP approx 10*D to 20*D.
    # We clip to [30, 80] to ensure reasonable generation speed.
    NP = int(np.clip(20 * dim, 30, 80))
    
    # Archive size (typically equal to population size)
    archive_size = NP 
    
    # Pre-process bounds for efficient vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracker
    global_best_val = float('inf')
    global_best_vec = None
    
    # -------------------------------------------------------------------------
    # Local Search Helper (Coordinate Descent / Pattern Search)
    # -------------------------------------------------------------------------
    def local_search(start_vec, start_val):
        """
        Performs a lightweight coordinate descent to polish the best solution.
        Stops if time runs out or no improvement is found.
        """
        current_vec = start_vec.copy()
        current_val = start_val
        
        # Initial step size relative to domain
        step_size = 0.05 * diff_b
        
        # Max iterations for polishing to avoid wasting time
        max_iter = 20
        
        for _ in range(max_iter):
            if check_time(): break
            
            improved = False
            for i in range(dim):
                if check_time(): break
                
                original_x = current_vec[i]
                
                # Try moving in positive direction
                current_vec[i] = np.clip(original_x + step_size[i], min_b[i], max_b[i])
                val = func(current_vec)
                
                if val < current_val:
                    current_val = val
                    improved = True
                    continue
                
                # Try moving in negative direction
                current_vec[i] = np.clip(original_x - step_size[i], min_b[i], max_b[i])
                val = func(current_vec)
                
                if val < current_val:
                    current_val = val
                    improved = True
                    continue
                
                # No improvement, revert
                current_vec[i] = original_x
            
            # Reduce step size if no improvement in this pass
            if not improved:
                step_size *= 0.5
                # Terminate if step size is negligible
                if np.max(step_size) < 1e-8:
                    break
            
        return current_vec, current_val

    # -------------------------------------------------------------------------
    # Main Restart Loop
    # -------------------------------------------------------------------------
    while not check_time():
        
        # 1. Initialization
        pop = min_b + np.random.rand(NP, dim) * diff_b
        fitness = np.full(NP, float('inf'))
        
        # Evaluate Initial Population
        for i in range(NP):
            if check_time(): return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_vec = pop[i].copy()
                
        # Initialize JADE Adaptive Parameters
        mu_cr = 0.5
        mu_f = 0.5
        c = 0.1     # Adaptation rate
        p = 0.05    # Top percentage for p-best
        
        # Initialize Archive
        archive = np.zeros((archive_size, dim))
        arc_count = 0
        
        # 2. Evolutionary Loop
        while not check_time():
            
            # Sort population by fitness
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # Convergence Detection
            # If population variance is extremely low, polish and restart
            if np.std(fitness) < 1e-8 or (fitness[-1] - fitness[0]) < 1e-8:
                # Polish the best found in this run
                best_loc, best_val = local_search(pop[0], fitness[0])
                if best_val < global_best_val:
                    global_best_val = best_val
                break # Break inner loop to restart
            
            # -----------------------------------------------------------
            # Parameter Generation
            # -----------------------------------------------------------
            # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1, NP)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1)
            # Generated as: mu_f + 0.1 * tan(pi * (rand - 0.5))
            rand_u = np.random.rand(NP)
            f = mu_f + 0.1 * np.tan(np.pi * (rand_u - 0.5))
            
            # Handle F bounds
            f = np.where(f > 1.0, 1.0, f)
            f = np.where(f <= 0.0, 0.1, f) # If too small/negative, reset to conservative small value
            
            # -----------------------------------------------------------
            # Mutation: DE/current-to-pbest/1 with Archive
            # V = X + F*(X_pbest - X) + F*(X_r1 - X_r2)
            # -----------------------------------------------------------
            
            # Select X_pbest (top p%)
            top_p_cnt = max(1, int(NP * p))
            pbest_idxs = np.random.randint(0, top_p_cnt, NP)
            x_pbest = pop[pbest_idxs]
            
            # Select r1 (random from pop)
            r1_idxs = np.random.randint(0, NP, NP)
            x_r1 = pop[r1_idxs]
            
            # Select r2 (random from Population U Archive)
            # Create a virtual union by indexing
            current_archive_size = arc_count
            total_size = NP + current_archive_size
            r2_idxs = np.random.randint(0, total_size, NP)
            
            # Construct x_r2 based on indices
            # Since numpy doesn't support conditional indexing easily across arrays without concat:
            if current_archive_size > 0:
                # Stack for vectorization (efficient for typical pop sizes)
                pop_archive_concat = np.vstack((pop, archive[:current_archive_size]))
                x_r2 = pop_archive_concat[r2_idxs]
            else:
                x_r2 = pop[r2_idxs % NP]
            
            # Compute Mutant Vectors
            f_col = f[:, None]
            mutant = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
            
            # -----------------------------------------------------------
            # Crossover (Binomial)
            # -----------------------------------------------------------
            rand_vals = np.random.rand(NP, dim)
            mask = rand_vals < cr[:, None]
            
            # Ensure at least one parameter is taken from mutant
            j_rand = np.random.randint(0, dim, NP)
            mask[np.arange(NP), j_rand] = True
            
            trial_pop = np.where(mask, mutant, pop)
            
            # -----------------------------------------------------------
            # Bound Handling (Midpoint Target)
            # -----------------------------------------------------------
            # Instead of clipping, if a variable violates a bound, 
            # set it to the midpoint between the bound and the parent.
            min_broad = np.tile(min_b, (NP, 1))
            max_broad = np.tile(max_b, (NP, 1))
            
            lower_viol = trial_pop < min_broad
            if np.any(lower_viol):
                trial_pop[lower_viol] = (min_broad[lower_viol] + pop[lower_viol]) * 0.5
                
            upper_viol = trial_pop > max_broad
            if np.any(upper_viol):
                trial_pop[upper_viol] = (max_broad[upper_viol] + pop[upper_viol]) * 0.5
                
            # -----------------------------------------------------------
            # Selection and Updates
            # -----------------------------------------------------------
            succ_f = []
            succ_cr = []
            
            for i in range(NP):
                if check_time(): return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    # Solution improved (or equal):
                    # 1. Archive the old parent
                    if arc_count < archive_size:
                        archive[arc_count] = pop[i].copy()
                        arc_count += 1
                    else:
                        # Randomly replace an individual in full archive
                        rm_idx = np.random.randint(0, archive_size)
                        archive[rm_idx] = pop[i].copy()
                    
                    # 2. Update Population
                    pop[i] = trial_pop[i]
                    fitness[i] = f_trial
                    
                    # 3. Record success parameters
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    
                    # 4. Update Global Best
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_vec = trial_pop[i].copy()
            
            # Update Adaptive Means (Lehmer Mean for F, Arithmetic for CR)
            if len(succ_f) > 0:
                sf = np.array(succ_f)
                scr = np.array(succ_cr)
                
                mean_lehmer = np.sum(sf**2) / (np.sum(sf) + 1e-10)
                mu_f = (1 - c) * mu_f + c * mean_lehmer
                
                mu_cr = (1 - c) * mu_cr + c * np.mean(scr)
                
    return global_best_val
