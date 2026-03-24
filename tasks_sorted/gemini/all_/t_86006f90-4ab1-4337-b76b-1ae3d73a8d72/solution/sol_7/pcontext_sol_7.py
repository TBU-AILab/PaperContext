#An improved algorithm for function minimization, **SHADE-PRS (Success-History Adaptive Differential Evolution with Population Reduction, Reflection, and Stagnation-based Restart)**, is provided below.
#
#### Improvements Explanation:
#1.  **Adaptive Population Reduction**: Inspired by the state-of-the-art L-SHADE algorithm, this method dynamically reduces the population size when stagnation is detected (rather than based on a fixed evaluation schedule). This concentrates the search effort on the best individuals ("exploitation") as the population converges, accelerating the finding of the minimum.
#2.  **Robust Restart Strategy with Polishing**: If the population converges (low variance) or reduces below a minimum size, the algorithm triggers a **Local Search (MTS-LS1)** on the best solution to refine it to high precision. Afterward, it performs a **Soft Restart** (keeping the global best, randomizing the rest) to escape local optima and explore new basins.
#3.  **Reflection Boundary Handling**: Instead of simple clipping (which biases the search to the edges), this algorithm uses **Reflection**, bouncing individuals back into the valid range. This preserves the population distribution and improves performance near boundaries.
#4.  **Vectorized Parameter Generation**: The SHADE parameter adaptation (Cauchy/Normal generation) is fully vectorized for performance, allowing more generations within the `max_time` limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using SHADE with Adaptive Population Reduction, 
    Reflection Boundary Handling, and Local Search Restarts (SHADE-PRS).
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Helper: Time Check ---
    def check_time():
        return (datetime.now() - start_time) >= time_limit

    # --- Pre-processing Bounds ---
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Max population: 18*dim is standard for SHADE, but capped for speed
    max_pop_size = int(max(20, 18 * dim))
    if max_pop_size > 120: max_pop_size = 120
    min_pop_size = 5
    
    # SHADE Memory Parameters
    H = 6
    
    # Global Best Tracking
    best_val = float('inf')
    best_sol = None

    # --- Boundary Handling: Reflection ---
    def reflect_bounds(x):
        """Reflects coordinates that go out of bounds back into the domain."""
        # Lower bound reflection
        mask_l = x < min_b
        if np.any(mask_l):
            # 2*min - x reflects x across min
            x[mask_l] = 2 * min_b[mask_l] - x[mask_l]
            # If still out (very far), clip
            x[mask_l] = np.maximum(x[mask_l], min_b[mask_l])
            
        # Upper bound reflection
        mask_u = x > max_b
        if np.any(mask_u):
            x[mask_u] = 2 * max_b[mask_u] - x[mask_u]
            x[mask_u] = np.minimum(x[mask_u], max_b[mask_u])
            
        return np.clip(x, min_b, max_b)

    # --- Local Search: MTS-LS1 (Polishing) ---
    def local_search(current_best, current_val):
        """
        Performs a simplified Multiple Trajectory Search (MTS) local search
        to refine the solution before a restart.
        """
        x = current_best.copy()
        val = current_val
        
        # Initial search range
        sr = diff_b * 0.4
        
        # Budget: Don't spend too long here
        ls_max_evals = 50 * dim
        ls_evals = 0
        
        while ls_evals < ls_max_evals and not check_time():
            improved = False
            # Search dimensions in random order
            dims = np.random.permutation(dim)
            
            for d in dims:
                if check_time(): break
                
                original_x = x[d]
                
                # 1. Try moving in negative direction
                x[d] = original_x - sr[d]
                x = reflect_bounds(x)
                new_val = func(x)
                ls_evals += 1
                
                if new_val < val:
                    val = new_val
                    improved = True
                else:
                    # 2. Try moving in positive direction (0.5 step for asymmetry)
                    x[d] = original_x + 0.5 * sr[d]
                    x = reflect_bounds(x)
                    new_val = func(x)
                    ls_evals += 1
                    
                    if new_val < val:
                        val = new_val
                        improved = True
                    else:
                        # Revert
                        x[d] = original_x
            
            # Adaptation
            if not improved:
                sr *= 0.5
                # Terminate if precision is high enough
                if np.max(sr) < 1e-8:
                    break
            
        return x, val

    # --- Main Restart Loop ---
    while not check_time():
        # 1. Initialization for new run
        pop_size = max_pop_size
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitism: Inject global best
        if best_sol is not None:
            pop[0] = best_sol.copy()
            
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return best_val
            
            if best_sol is not None and i == 0:
                fitness[i] = best_val
            else:
                f = func(pop[i])
                fitness[i] = f
                if f < best_val:
                    best_val = f
                    best_sol = pop[i].copy()
        
        # Sort by fitness
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

        # SHADE Memory
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        archive = []
        
        stagnation_count = 0
        last_best_fit = fitness[0]
        
        # --- Evolutionary Generation Loop ---
        while not check_time():
            # 2. Population Reduction & Stagnation Logic
            current_best_fit = fitness[0]
            
            # Check for improvement
            if np.abs(current_best_fit - last_best_fit) < 1e-8:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best_fit = current_best_fit
                
            # Trigger: Low Variance -> Immediate Polish & Restart
            if np.std(fitness) < 1e-9:
                polished_sol, polished_val = local_search(pop[0], fitness[0])
                if polished_val < best_val:
                    best_val = polished_val
                    best_sol = polished_sol.copy()
                break # Break inner loop to restart
            
            # Trigger: Stagnation -> Reduce Population
            if stagnation_count > 20:
                # Reduce population by 25%
                new_size = int(pop_size * 0.75)
                
                # If population gets too small, restart
                if new_size < min_pop_size:
                    polished_sol, polished_val = local_search(pop[0], fitness[0])
                    if polished_val < best_val:
                        best_val = polished_val
                        best_sol = polished_sol.copy()
                    break # Break inner loop to restart
                
                # Apply Reduction (Keep best individuals, as pop is sorted)
                pop_size = new_size
                pop = pop[:pop_size]
                fitness = fitness[:pop_size]
                
                # Adjust archive size
                target_arc = int(pop_size * 2.0)
                if len(archive) > target_arc:
                    del archive[target_arc:]
                
                # Reset stagnation counter to allow adaptation to new size
                stagnation_count = 0

            # 3. SHADE Parameter Generation
            # Select memory slots
            r_idx = np.random.randint(0, H, pop_size)
            m_cr = M_cr[r_idx]
            m_f = M_f[r_idx]
            
            # Generate CR ~ Normal(M_cr, 0.1)
            cr = np.random.normal(m_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # Generate F ~ Cauchy(M_f, 0.1)
            # Vectorized generation with rejection sampling
            f_params = m_f + 0.1 * np.random.standard_cauchy(pop_size)
            retry_mask = f_params <= 0
            while np.any(retry_mask):
                f_params[retry_mask] = m_f[retry_mask] + 0.1 * np.random.standard_cauchy(np.sum(retry_mask))
                retry_mask = f_params <= 0
            f_params = np.minimum(f_params, 1.0) # Clip > 1 to 1.0
            
            # 4. Evolution Step
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(pop_size)
            
            success_cr = []
            success_f = []
            success_df = []
            
            # p-best selection param (random in [2/N, 0.2])
            p_val = np.random.uniform(2.0/pop_size, 0.2)
            num_p = int(max(2, pop_size * p_val))
            
            # Prepare indices for vectorization logic (semi-vectorized loop)
            for i in range(pop_size):
                if check_time(): return best_val
                
                # Mutation: current-to-pbest/1
                # V = x + F*(x_pbest - x) + F*(x_r1 - x_r2)
                
                p_idx = np.random.randint(0, num_p)
                x_pbest = pop[p_idx]
                
                r1 = np.random.randint(0, pop_size)
                while r1 == i: r1 = np.random.randint(0, pop_size)
                x_r1 = pop[r1]
                
                # r2 from Union(Population, Archive)
                len_arc = len(archive)
                r2 = np.random.randint(0, pop_size + len_arc)
                while r2 == i or r2 == r1: r2 = np.random.randint(0, pop_size + len_arc)
                
                if r2 < pop_size:
                    x_r2 = pop[r2]
                else:
                    x_r2 = archive[r2 - pop_size]
                
                mutant = pop[i] + f_params[i] * (x_pbest - pop[i]) + f_params[i] * (x_r1 - x_r2)
                
                # Crossover: Binomial
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < cr[i]
                mask[j_rand] = True
                
                trial = np.where(mask, mutant, pop[i])
                
                # Boundary Handling: Reflection
                trial = reflect_bounds(trial)
                
                # Selection
                f_trial = func(trial)
                
                if f_trial < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    
                    success_cr.append(cr[i])
                    success_f.append(f_params[i])
                    success_df.append(fitness[i] - f_trial)
                    
                    archive.append(pop[i].copy())
                    
                    if f_trial < best_val:
                        best_val = f_trial
                        best_sol = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # Update Population
            pop = new_pop
            fitness = new_fitness
            
            # Sort for next generation (p-best logic requires sorted pop)
            sorted_indices = np.argsort(fitness)
            pop = pop[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # 5. Archive Maintenance
            target_arc = int(pop_size * 2.0)
            while len(archive) > target_arc:
                # Randomly remove elements
                archive.pop(np.random.randint(0, len(archive)))
            
            # 6. Memory Update (Weighted Lehmer Mean)
            if len(success_df) > 0:
                s_df = np.array(success_df)
                s_cr = np.array(success_cr)
                s_f = np.array(success_f)
                
                total_df = np.sum(s_df)
                if total_df > 0:
                    weights = s_df / total_df
                    
                    mean_cr = np.sum(weights * s_cr)
                    mean_f = np.sum(weights * s_f**2) / np.sum(weights * s_f)
                    
                    M_cr[k_mem] = mean_cr
                    M_f[k_mem] = mean_f
                    k_mem = (k_mem + 1) % H

    return best_val
