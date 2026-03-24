#To improve upon the previous algorithm, I propose the **L-SHADE-Time (Linear Success-History Adaptive Differential Evolution with Time-Based Reduction)** algorithm. 
#
#**Key Improvements:**
#1.  **Linear Population Size Reduction (LPSR):** Instead of a fixed population, we start with a large population to explore the search space and linearly reduce it over time to a minimum size. This forces the algorithm to shift from exploration to exploitation naturally within the `max_time` constraint.
#2.  **Weighted Lehmer Mean Memory:** The algorithm uses a memory of successful control parameters ($F$ and $CR$) updated using a weighted Lehmer mean based on fitness improvement. This adapts the search strategy to the landscape more effectively than simple averages.
#3.  **Vectorized Operations:** The mutation and crossover steps are fully vectorized using NumPy. This significantly reduces the Python loop overhead, allowing for more generations and function evaluations within the limited time.
#4.  **Adaptive Nelder-Mead Polish:** A local search is applied at the very end using the remaining time budget to refine the best solution found by the evolutionary phase.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using L-SHADE (Linear Success-History Adaptive Differential Evolution)
    adapted for a Time-Based stopping criterion, followed by a Nelder-Mead polish.
    """
    
    # --- Helper: Nelder-Mead Local Search ---
    def nelder_mead(x0, budget_seconds):
        """Standard Nelder-Mead optimization restricted by time."""
        nm_start = datetime.now()
        nm_limit = timedelta(seconds=budget_seconds)
        
        # NM Parameters
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        # Precompute bounds arrays
        b_min = np.array([b[0] for b in bounds])
        b_max = np.array([b[1] for b in bounds])
        b_diff = b_max - b_min
        
        # Initialize Simplex
        simplex = [x0.copy()]
        try:
            simplex_vals = [func(x0)]
        except:
            return x0, float('inf')

        # Create initial simplex points
        step_coeff = 0.05
        for i in range(dim):
            x = x0.copy()
            step = step_coeff * b_diff[i]
            x[i] += step
            
            # Bound handling
            if x[i] > b_max[i]: x[i] = b_max[i] - step
            elif x[i] < b_min[i]: x[i] = b_min[i] + step
            
            val = func(x)
            simplex.append(x)
            simplex_vals.append(val)
            
            if (datetime.now() - nm_start) >= nm_limit:
                break
                
        simplex = np.array(simplex)
        simplex_vals = np.array(simplex_vals)
        
        # Main Loop
        while (datetime.now() - nm_start) < nm_limit:
            # Sort
            order = np.argsort(simplex_vals)
            simplex = simplex[order]
            simplex_vals = simplex_vals[order]
            
            # Check convergence
            if np.std(simplex_vals) < 1e-9:
                break

            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, b_min, b_max)
            val_r = func(xr)
            
            if simplex_vals[0] <= val_r < simplex_vals[-2]:
                simplex[-1] = xr
                simplex_vals[-1] = val_r
            elif val_r < simplex_vals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, b_min, b_max)
                val_e = func(xe)
                if val_e < val_r:
                    simplex[-1] = xe
                    simplex_vals[-1] = val_e
                else:
                    simplex[-1] = xr
                    simplex_vals[-1] = val_r
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, b_min, b_max)
                val_c = func(xc)
                if val_c < simplex_vals[-1]:
                    simplex[-1] = xc
                    simplex_vals[-1] = val_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], b_min, b_max)
                        simplex_vals[i] = func(simplex[i])
        
        best_idx = np.argmin(simplex_vals)
        return simplex[best_idx], simplex_vals[best_idx]

    # --- Main Algorithm Configuration ---
    start_time = datetime.now()
    
    # Reserve a small portion of time for final local search (5-10%)
    polish_ratio = 0.1
    polish_time = max(0.1, max_time * polish_ratio)
    evo_time = max_time - polish_time
    evo_limit = timedelta(seconds=evo_time)
    
    # L-SHADE Parameters
    # Start with a large population for exploration
    # Cap size to prevent slowdowns in Python loop if dim is very high
    pop_size_init = min(int(18 * dim), 200) 
    pop_size_init = max(pop_size_init, 20)
    pop_size_min = 4
    
    # Memory for Adaptive Parameters (H=6 is standard)
    H = 6
    M_cr = np.full(H, 0.5)
    M_f = np.full(H, 0.5)
    k_mem = 0  # Memory index pointer
    
    # Bounds preprocessing
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    
    # Initialization (LHS-like)
    pop = np.zeros((pop_size_init, dim))
    for d in range(dim):
        edges = np.linspace(min_b[d], max_b[d], pop_size_init + 1)
        points = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(points)
        pop[:, d] = points
        
    # Evaluate Initial Population
    fitness = np.array([func(ind) for ind in pop])
    
    # Best Tracking
    best_idx = np.argmin(fitness)
    best_fit = fitness[best_idx]
    best_sol = pop[best_idx].copy()
    
    archive = []
    pop_size = pop_size_init
    
    # --- Evolutionary Loop ---
    while (datetime.now() - start_time) < evo_limit:
        
        # 1. Linear Population Size Reduction (LPSR) based on Time
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = elapsed / evo_time
        if progress > 1.0: progress = 1.0
        
        # Calculate target size
        target_size = int(round((pop_size_min - pop_size_init) * progress + pop_size_init))
        target_size = max(pop_size_min, target_size)
        
        if target_size < pop_size:
            # Sort and truncate population
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices[:target_size]]
            fitness = fitness[sort_indices[:target_size]]
            pop_size = target_size
            
            # Maintain Archive size proportional to initial pop or current pop?
            # Keeping it bounded by init size maintains diversity.
            if len(archive) > pop_size_init:
                del archive[:(len(archive) - pop_size_init)]
                
        # 2. Adaptive Parameter Generation
        # Random index for each individual
        r_idx = np.random.randint(0, H, pop_size)
        
        # Generate CR ~ Normal(M_cr, 0.1)
        cr = np.random.normal(M_cr[r_idx], 0.1)
        cr = np.clip(cr, 0, 1)
        
        # Generate F ~ Cauchy(M_f, 0.1)
        f = M_f[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        f = np.where(f <= 0, 0.05, f) # Resample lower (clamp for speed)
        f = np.clip(f, 0, 1)          # Truncate upper
        
        # 3. Mutation: current-to-pbest/1
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Sort to find p-best
        sorted_indices = np.argsort(fitness)
        
        # Select p-best (top p% individuals, p=0.11 is common)
        p_limit = max(2, int(0.11 * pop_size))
        pbest_indices = np.random.randint(0, p_limit, pop_size)
        pbest_indices = sorted_indices[pbest_indices]
        x_pbest = pop[pbest_indices]
        
        # Select r1 (random from pop)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = pop[r1_indices]
        
        # Select r2 (random from pop U archive)
        if len(archive) > 0:
            union_pop = np.vstack((pop, np.array(archive)))
        else:
            union_pop = pop
        r2_indices = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Calculate Mutation Vectors (Vectorized)
        f_col = f[:, np.newaxis]
        v = pop + f_col * (x_pbest - pop) + f_col * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        j_rand = np.random.randint(0, dim, pop_size)
        rand_vals = np.random.rand(pop_size, dim)
        mask = rand_vals < cr[:, np.newaxis]
        mask[np.arange(pop_size), j_rand] = True # Ensure at least one dim changes
        
        u = np.where(mask, v, pop)
        
        # Bound Constraint Handling (Midpoint Correction)
        lower_vio = u < min_b
        upper_vio = u > max_b
        
        u[lower_vio] = (pop[lower_vio] + min_b[np.where(lower_vio)[1]]) / 2.0
        u[upper_vio] = (pop[upper_vio] + max_b[np.where(upper_vio)[1]]) / 2.0
        
        # 5. Selection and Memory Update
        new_pop_list = []
        new_fit_list = []
        
        # Lists for memory update
        succ_f = []
        succ_cr = []
        succ_diff = []
        
        # Evaluate loop
        for i in range(pop_size):
            if (datetime.now() - start_time) >= evo_limit:
                break
                
            val = func(u[i])
            
            if val <= fitness[i]:
                new_pop_list.append(u[i])
                new_fit_list.append(val)
                
                # If strictly better, store for memory update
                if val < fitness[i]:
                    succ_f.append(f[i])
                    succ_cr.append(cr[i])
                    succ_diff.append(fitness[i] - val)
                    archive.append(pop[i].copy())
                
                # Global best update
                if val < best_fit:
                    best_fit = val
                    best_sol = u[i].copy()
            else:
                new_pop_list.append(pop[i])
                new_fit_list.append(fitness[i])
        
        # Check if loop completed or timed out
        if len(new_pop_list) < pop_size:
            break
            
        pop = np.array(new_pop_list)
        fitness = np.array(new_fit_list)
        
        # 6. Update Memory (Weighted Lehmer Mean)
        if len(succ_f) > 0:
            s_f = np.array(succ_f)
            s_cr = np.array(succ_cr)
            s_diff = np.array(succ_diff)
            
            # Weights based on fitness improvement
            weights = s_diff / np.sum(s_diff)
            
            # Lehmer Mean for F
            mean_f = np.sum(weights * (s_f ** 2)) / (np.sum(weights * s_f) + 1e-10)
            
            # Weighted Arithmetic Mean for CR
            mean_cr = np.sum(weights * s_cr)
            
            M_f[k_mem] = mean_f
            M_cr[k_mem] = mean_cr
            k_mem = (k_mem + 1) % H

    # --- Final Polish ---
    # Use remaining time for Nelder-Mead to refine the best solution
    elapsed = (datetime.now() - start_time).total_seconds()
    remaining = max_time - elapsed
    
    if remaining > 0.05:
        nm_sol, nm_val = nelder_mead(best_sol, remaining)
        if nm_val < best_fit:
            best_fit = nm_val
            
    return best_fit
