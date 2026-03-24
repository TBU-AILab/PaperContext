#Here is the optimized algorithm.
#
#### Algorithm Analysis
#The previous result (~27.2) indicates the algorithm is finding a good basin of attraction but failing to converge to the absolute global minimum (likely 0.0) or getting stuck in a local optimum near the global one. This behavior is typical for multimodal functions like Rastrigin or shifted Rosenbrock when the local search is not aggressive enough or the convergence pressure of the population is insufficient.
#
#To address this, the proposed solution implements a **Self-Adaptive Differential Evolution (JADE-like)** coupled with a robust **MTS-LS1 (Multiple Trajectory Search - Local Search 1)** mechanism.
#
#**Key Improvements:**
#1.  **JADE Mutation (`current-to-pbest/1`)**: This strategy directs the search towards the top $p\%$ best individuals, offering significantly faster convergence than standard `rand/1` mutation.
#2.  **Adaptive Parameters**: $F$ (Mutation factor) and $CR$ (Crossover rate) are self-adapted for each individual based on successful history (Lehmer mean), removing the need for manual tuning.
#3.  **MTS-LS1 Local Search**: This is a powerful coordinate descent method specifically designed to refine the best solution. It maintains a "search range" per dimension, expanding or contracting based on success, allowing it to "zoom in" on the global minimum with high precision.
#4.  **Stagnation Handling**: A restart mechanism is triggered if the population stagnates, resetting the search while preserving the best-found solution to escape local optima.
#
#### Python Code
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Adaptive based on dimension, but bounded for Python performance
    # Range [20, 60] strikes a balance between diversity and generation speed
    pop_size = int(np.clip(10 * dim, 20, 60)) 
    
    # JADE / Adaptive DE parameters
    mu_cr = 0.5        # Initial mean Crossover Rate
    mu_f = 0.5         # Initial mean Mutation Factor
    c_adapt = 0.1      # Adaptation learning rate
    p_greedy = 0.05    # Top 5% for current-to-pbest mutation
    
    # Bounds processing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- State Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.zeros(pop_size)
    
    # Archive for storing inferior solutions (maintains diversity in mutation)
    archive = []
    
    # Global Best tracking
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # MTS-LS1 Local Search State: Search range per dimension
    search_range = diff_b * 0.4 
    
    # Helper: Safe Evaluation with boundary enforcement
    def eval_solution(x):
        return func(np.clip(x, min_b, max_b))

    # Initial Population Evaluation
    for i in range(pop_size):
        if time.time() - start_time > max_time * 0.99: break
        val = eval_solution(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()

    # --- Main Optimization Loop ---
    gens_no_improve = 0
    
    while True:
        # Time Check
        elapsed = time.time() - start_time
        if elapsed > max_time * 0.98:
            return best_val
            
        # 1. Parameter Adaptation (JADE style)
        # Generate CR ~ Normal(mu_cr, 0.1)
        cr_g = np.random.normal(mu_cr, 0.1, pop_size)
        cr_g = np.clip(cr_g, 0, 1)
        
        # Generate F ~ Cauchy(mu_f, 0.1)
        # Use standard_cauchy (loc=0, scale=1) -> transform to loc=mu_f, scale=0.1
        f_g = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        
        # Repair F: if <= 0 regenerate, if > 1 clip to 1
        bad_f = f_g <= 0
        while np.any(bad_f):
            count = np.sum(bad_f)
            f_g[bad_f] = np.random.standard_cauchy(count) * 0.1 + mu_f
            bad_f = f_g <= 0
        f_g = np.clip(f_g, 0, 1)
        
        # 2. Mutation: DE/current-to-pbest/1
        # V = X_i + F * (X_pbest - X_i) + F * (X_r1 - X_r2)
        
        # Identify p-best individuals (top p%)
        sorted_idx = np.argsort(fitness)
        num_pbest = max(1, int(pop_size * p_greedy))
        pbest_indices = sorted_idx[:num_pbest]
        
        # Select pbest for each individual
        pbest_selection = np.random.choice(pbest_indices, pop_size)
        x_pbest = pop[pbest_selection]
        
        # Select r1 (random from population)
        r1 = np.random.randint(0, pop_size, pop_size)
        x_r1 = pop[r1]
        
        # Select r2 (random from Population U Archive)
        if len(archive) > 0:
            archive_np = np.array(archive)
            union_pop = np.vstack((pop, archive_np))
        else:
            union_pop = pop
            
        r2 = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2]
        
        # Compute Mutant Vector (Vectorized)
        F_col = f_g[:, np.newaxis]
        mutant = pop + F_col * (x_pbest - pop) + F_col * (x_r1 - x_r2)
        mutant = np.clip(mutant, min_b, max_b)
        
        # 3. Crossover (Binomial)
        rand_matrix = np.random.rand(pop_size, dim)
        cross_mask = rand_matrix < cr_g[:, np.newaxis]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 4. Selection and Update
        succ_f = []
        succ_cr = []
        gen_improved = False
        
        for i in range(pop_size):
            # Frequent time check for expensive functions
            if (i % 10 == 0) and (time.time() - start_time > max_time * 0.99):
                return best_val

            f_trial = eval_solution(trial_pop[i])
            
            if f_trial < fitness[i]:
                # Successful update
                archive.append(pop[i].copy()) # Save old to archive
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                
                # Record successful parameters
                succ_f.append(f_g[i])
                succ_cr.append(cr_g[i])
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()
                    gen_improved = True
                    gens_no_improve = 0
            else:
                pass

        # Maintain Archive Size (Limit to pop_size)
        while len(archive) > pop_size:
            idx_rm = np.random.randint(0, len(archive))
            del archive[idx_rm]
            
        # Update Adaptive Parameters (Lehmer Mean)
        if len(succ_f) > 0:
            sf = np.array(succ_f)
            scr = np.array(succ_cr)
            mean_lehmer = np.sum(sf**2) / (np.sum(sf) + 1e-15)
            mu_f = (1 - c_adapt) * mu_f + c_adapt * mean_lehmer
            mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(scr)
            
        if not gen_improved:
            gens_no_improve += 1
            
        # 5. Local Search (MTS-LS1) on Global Best
        # Triggered on improvement or periodically during stagnation
        do_ls = False
        if gen_improved: 
            do_ls = True
        elif gens_no_improve > 10 and gens_no_improve % 5 == 0:
            do_ls = True
            
        if do_ls:
            # Optimize random subset of dimensions if D is huge, else all
            dims_to_check = np.arange(dim)
            if dim > 50:
                dims_to_check = np.random.choice(dim, 50, replace=False)
            
            ls_improved = False
            curr_ls_vec = best_vec.copy()
            curr_ls_val = best_val
            
            for d in dims_to_check:
                if time.time() - start_time > max_time * 0.99: return best_val
                
                # Reset search range if it vanished (precision limit)
                if search_range[d] < 1e-12:
                     search_range[d] = diff_b[d] * 0.4
                     
                # Try Negative Step
                original_val = curr_ls_vec[d]
                curr_ls_vec[d] = np.clip(original_val - search_range[d], min_b[d], max_b[d])
                val_minus = func(curr_ls_vec)
                
                if val_minus < curr_ls_val:
                    curr_ls_val = val_minus
                    best_val = val_minus
                    best_vec = curr_ls_vec.copy()
                    ls_improved = True
                else:
                    # Revert
                    curr_ls_vec[d] = original_val
                    
                    # Try Positive Step (0.5 size - MTS asymmetry improves convergence)
                    curr_ls_vec[d] = np.clip(original_val + 0.5 * search_range[d], min_b[d], max_b[d])
                    val_plus = func(curr_ls_vec)
                    
                    if val_plus < curr_ls_val:
                        curr_ls_val = val_plus
                        best_val = val_plus
                        best_vec = curr_ls_vec.copy()
                        ls_improved = True
                    else:
                        # Revert and Contract Search Range
                        curr_ls_vec[d] = original_val
                        search_range[d] *= 0.5

            if ls_improved:
                # Inject improvement back into population (replace worst)
                worst_idx = np.argmax(fitness)
                pop[worst_idx] = best_vec
                fitness[worst_idx] = best_val
                gens_no_improve = 0

        # 6. Stagnation Restart
        # If no improvement for many generations, restart population but keep best
        if gens_no_improve > 35:
            # Ensure sufficient time remains for a meaningful restart
            if (time.time() - start_time) < max_time * 0.8:
                saved_best = best_vec.copy()
                saved_val = best_val
                
                # Re-initialize population
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                pop[0] = saved_best # Elitism
                
                # Reset Archive and JADE Params
                archive = []
                mu_f = 0.5
                mu_cr = 0.5
                search_range = diff_b * 0.4 # Reset LS precision
                
                # Evaluate new population
                fitness[0] = saved_val
                for i in range(1, pop_size):
                    if time.time() - start_time > max_time * 0.99: return best_val
                    fitness[i] = eval_solution(pop[i])
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_vec = pop[i].copy()
                
                gens_no_improve = 0
                
    return best_val
