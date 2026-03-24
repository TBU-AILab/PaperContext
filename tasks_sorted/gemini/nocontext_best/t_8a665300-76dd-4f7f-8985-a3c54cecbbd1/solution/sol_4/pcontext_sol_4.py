#To improve upon the previous algorithm, we will implement a robust variant of **JADE (Adaptive Differential Evolution with Optional External Archive)** coupled with a **Nelder-Mead Simplex** local search. 
#
#**Key Improvements:**
#1.  **Current-to-pbest Mutation:** instead of "current-to-best", we target the top $p\%$ of individuals. This maintains selection pressure without collapsing diversity as fast as the previous algorithm.
#2.  **Archive of Inferior Solutions:** We store recently discarded parent vectors. The mutation operator can pull difference vectors from this archive. This significantly extends the diversity of directions available for exploration.
#3.  **Adaptive Parameter Control (Success History):** We update the crossover ($CR$) and mutation ($F$) factors based on a history of successful updates (using Lehmer mean), rather than just random jittering.
#4.  **Nelder-Mead Polishing:** The previous coordinate descent failed to account for variable interactions (diagonal valleys). Nelder-Mead forms a simplex shape that adapts to the local topography, handling correlations between variables much better.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using JADE (Adaptive Differential Evolution) with an Archive,
    coupled with Nelder-Mead Simplex local search for final polishing.
    """
    
    # --- Configuration ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Reserve a small slice of time for the final polish
    polish_time_ratio = 0.15 
    
    # Differential Evolution Parameters
    # Population size: Standard JADE suggestion is around 10*dim, 
    # but we cap it for speed in restricted time scenarios.
    pop_size = min(max(20, 5 * dim), 100)
    
    # Adaptive Parameter Memory
    mu_cr = 0.5
    mu_f = 0.5
    c_adaptive = 0.1  # Learning rate for adaptation
    p_best_rate = 0.05 # Top 5% used for mutation target
    
    # Archive Configuration
    archive = []
    max_archive_size = pop_size

    # Bounds preprocessing
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b

    # Global best tracking
    best_fitness = float('inf')
    best_solution = None

    def check_timeout(buffer_seconds=0.0):
        """Returns True if time is up (minus a safety buffer)."""
        return (datetime.now() - start_time) >= (time_limit - timedelta(seconds=buffer_seconds))

    # --- Helper: Nelder-Mead Local Search ---
    def nelder_mead(x0, budget_time_seconds):
        """
        Runs a Nelder-Mead Simplex optimization starting at x0.
        Restricted by time and bounds.
        """
        nm_start = datetime.now()
        nm_limit = timedelta(seconds=budget_time_seconds)
        
        # Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Initialize Simplex
        # x0 plus D points shifted along axes
        simplex = [x0.copy()]
        simplex_vals = [func(x0)]
        
        step_size = 0.05 # Initial step size relative to bounds
        
        for i in range(dim):
            x = x0.copy()
            # Perturb dimension i
            step = step_size * diff_b[i]
            x[i] += step
            
            # Bound check
            if x[i] > max_b[i]:
                x[i] = max_b[i] - step_size * diff_b[i] # Reflect back
            elif x[i] < min_b[i]:
                x[i] = min_b[i] + step_size * diff_b[i]
            
            x = np.clip(x, min_b, max_b)
            val = func(x)
            simplex.append(x)
            simplex_vals.append(val)
            
            if (datetime.now() - nm_start) >= nm_limit:
                break

        simplex = np.array(simplex)
        simplex_vals = np.array(simplex_vals)
        
        # Main Nelder-Mead Loop
        while (datetime.now() - nm_start) < nm_limit:
            # Sort
            order = np.argsort(simplex_vals)
            simplex = simplex[order]
            simplex_vals = simplex_vals[order]
            
            best_local_val = simplex_vals[0]
            worst_local_val = simplex_vals[-1]
            
            # Check for convergence (flat simplex)
            if np.max(np.abs(simplex_vals - best_local_val)) < 1e-9:
                break

            # Centroid of best D points (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, min_b, max_b)
            val_r = func(xr)
            
            if simplex_vals[0] <= val_r < simplex_vals[-2]:
                simplex[-1] = xr
                simplex_vals[-1] = val_r
            
            # Expansion
            elif val_r < simplex_vals[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, min_b, max_b)
                val_e = func(xe)
                if val_e < val_r:
                    simplex[-1] = xe
                    simplex_vals[-1] = val_e
                else:
                    simplex[-1] = xr
                    simplex_vals[-1] = val_r
                    
            # Contraction
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, min_b, max_b)
                val_c = func(xc)
                if val_c < simplex_vals[-1]:
                    simplex[-1] = xc
                    simplex_vals[-1] = val_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], min_b, max_b)
                        simplex_vals[i] = func(simplex[i])

        # Return best from simplex
        idx = np.argmin(simplex_vals)
        return simplex[idx], simplex_vals[idx]

    # --- Initialization ---
    # Latin Hypercube Sampling (LHS) approximation for better initial coverage
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        # Divide dimension d into pop_size intervals
        edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
        # Pick a random point in each interval
        points = np.random.uniform(edges[:-1], edges[1:])
        # Shuffle to uncorrelate dimensions
        np.random.shuffle(points)
        population[:, d] = points

    pop_fitness = np.array([func(ind) for ind in population])
    
    # Update global best
    best_idx = np.argmin(pop_fitness)
    if pop_fitness[best_idx] < best_fitness:
        best_fitness = pop_fitness[best_idx]
        best_solution = population[best_idx].copy()

    # --- Main Loop ---
    while not check_timeout():
        
        # Sort population to easily find p-best
        sorted_indices = np.argsort(pop_fitness)
        population = population[sorted_indices]
        pop_fitness = pop_fitness[sorted_indices]
        
        # Check if we should stop purely evolutionary phase to save time for polish
        # If we have little time left, break to run Nelder-Mead
        remaining = (time_limit - (datetime.now() - start_time)).total_seconds()
        if remaining < max_time * polish_time_ratio:
            break

        # Generate Adaptive Parameters for this generation
        # CR ~ Normal(mu_cr, 0.1)
        # F ~ Cauchy(mu_f, 0.1)
        cr_g = np.random.normal(mu_cr, 0.1, pop_size)
        cr_g = np.clip(cr_g, 0, 1)
        
        f_g = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        f_g = np.clip(f_g, 0.1, 1.0) # Truncate F
        
        # Successful parameters memory
        successful_f = []
        successful_cr = []
        
        # Create mutation pool: Population + Archive
        if len(archive) > 0:
            archive_arr = np.array(archive)
            pool = np.vstack((population, archive_arr))
        else:
            pool = population
            
        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(pop_fitness)
        
        # Iterate through population
        for i in range(pop_size):
            if check_timeout():
                return best_fitness

            x_i = population[i]
            
            # 1. Mutation: Current-to-pbest/1
            # Select p-best (randomly from top p%)
            p_limit = max(1, int(pop_size * p_best_rate))
            p_best_idx = np.random.randint(0, p_limit)
            x_pbest = population[p_best_idx]
            
            # Select r1 from population (distinct from i)
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # Select r2 from Union(Pop, Archive) (distinct from i and r1)
            r2 = np.random.randint(0, len(pool))
            while r2 == i or (r2 < pop_size and r2 == r1):
                r2 = np.random.randint(0, len(pool))
            x_r2 = pool[r2]
            
            # Mutate
            v_i = x_i + f_g[i] * (x_pbest - x_i) + f_g[i] * (x_r1 - x_r2)
            
            # 2. Crossover (Binomial)
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < cr_g[i]
            mask[j_rand] = True # Ensure at least one parameter changes
            
            u_i = np.where(mask, v_i, x_i)
            
            # Bound Constraint Handling (Reflection/Bounce logic is often better than clipping)
            # If violated, set value between old and bound
            lower_violation = u_i < min_b
            upper_violation = u_i > max_b
            
            # Simple bounce back
            u_i[lower_violation] = (x_i[lower_violation] + min_b[lower_violation]) / 2.0
            u_i[upper_violation] = (x_i[upper_violation] + max_b[upper_violation]) / 2.0
            
            # 3. Selection
            fit_u = func(u_i)
            
            if fit_u < pop_fitness[i]:
                new_population[i] = u_i
                new_fitness[i] = fit_u
                
                # Store success for parameter adaptation
                successful_f.append(f_g[i])
                successful_cr.append(cr_g[i])
                
                # Add replaced parent to archive
                archive.append(x_i.copy())
                
                # Update global best immediately
                if fit_u < best_fitness:
                    best_fitness = fit_u
                    best_solution = u_i.copy()
            else:
                new_population[i] = x_i
                new_fitness[i] = pop_fitness[i]
        
        # Update Population
        population = new_population
        pop_fitness = new_fitness
        
        # Manage Archive Size
        if len(archive) > max_archive_size:
            # Remove random elements to maintain size
            remove_count = len(archive) - max_archive_size
            # Simply delete from beginning (oldest) or random
            del archive[:remove_count]
            
        # Update Adaptive Parameters (Lehmer Mean)
        if len(successful_f) > 0:
            mean_pow2 = np.mean(np.array(successful_f)**2)
            mean_pow1 = np.mean(np.array(successful_f))
            mu_f = (1 - c_adaptive) * mu_f + c_adaptive * (mean_pow2 / (mean_pow1 + 1e-10))
            
            mu_cr = (1 - c_adaptive) * mu_cr + c_adaptive * np.mean(successful_cr)

    # --- Final Polish (Nelder-Mead) ---
    # Use remaining time to refine the best solution found
    elapsed = (datetime.now() - start_time).total_seconds()
    remaining = max_time - elapsed
    
    if remaining > 0.1 and best_solution is not None:
        nm_pos, nm_val = nelder_mead(best_solution, remaining)
        if nm_val < best_fitness:
            best_fitness = nm_val
            best_solution = nm_pos

    return best_fitness
