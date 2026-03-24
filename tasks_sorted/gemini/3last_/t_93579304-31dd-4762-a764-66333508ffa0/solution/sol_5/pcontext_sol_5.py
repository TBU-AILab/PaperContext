#The following algorithm implements **L-SHADE with Terminal Coordinate Descent**.
#
#**Improvements and Rationale:**
#1.  **L-SHADE Core**: We retain the highly effective **L-SHADE** architecture (Linear Population Size Reduction, History-based Parameter Adaptation) which achieved the best result so far (1.42). It excels at global exploration and finding the basin of attraction.
#2.  **Terminal Coordinate Descent (Polishing)**: Differential Evolution is stochastic and can be slow to converge to high precision once inside the optimal valley. To address the gap between the previous best result (1.42) and the optimum (0.0), this algorithm switches to a deterministic **Coordinate Descent** (local search) when the population converges or time is nearly up. This "polishes" the solution to machine precision.
#3.  **Adaptive `p`-value**: The greediness of the mutation strategy (`current-to-pbest`) adapts linearly from exploration ($p=0.2$) to exploitation ($p=0.05$) over time.
#4.  **Vectorized Operations**: Loop overhead is minimized by processing the population in parallel using NumPy operations.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using L-SHADE with Terminal Coordinate Descent.
    
    1. L-SHADE: Global search with linear population reduction and adaptive parameters.
    2. Coordinate Descent: Local polishing of the best solution in the final stage.
    """
    
    # --- Time Management ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # L-SHADE Parameters
    # Initial Population: 20*dim is a robust heuristic (capped for performance)
    pop_size_init = int(max(30, 20 * dim))
    pop_size_init = min(pop_size_init, 300) 
    min_pop_size = 4
    
    # Memory for adaptive parameters (History size H=5)
    H = 5
    mem_f = np.full(H, 0.5)
    mem_cr = np.full(H, 0.5)
    k_mem = 0
    
    # Archive to maintain diversity
    archive = []
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population (Uniform)
    population = min_b + np.random.rand(pop_size_init, dim) * diff_b
    fitness = np.full(pop_size_init, float('inf'))
    
    best_val = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate Initial Population
    # Check time strictly to handle very short durations
    for i in range(pop_size_init):
        if (datetime.now() - start_time) >= limit:
            return best_val
            
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
            
    # Sort population by fitness (Best at index 0)
    sorted_idxs = np.argsort(fitness)
    population = population[sorted_idxs]
    fitness = fitness[sorted_idxs]
    
    # --- Phase 1: Global Search (L-SHADE) ---
    while True:
        now = datetime.now()
        if now - start_time >= limit:
            break
            
        # Calculate Progress (0.0 to 1.0)
        elapsed = (now - start_time).total_seconds()
        progress = elapsed / max_time
        
        # 1. Linear Population Size Reduction (LPSR)
        # Calculate target size based on time remaining
        n_target = int(round(pop_size_init * (1.0 - progress) + min_pop_size * progress))
        n_target = max(min_pop_size, n_target)
        
        curr_size = len(population)
        
        # Reduce population if needed
        if curr_size > n_target:
            # Since population is sorted, the worst are at the end. Truncate.
            population = population[:n_target]
            fitness = fitness[:n_target]
            curr_size = n_target
            
            # Reduce archive size to match current population
            while len(archive) > curr_size:
                archive.pop(np.random.randint(len(archive)))
                
        # 2. Adaptive Parameter Generation
        # p-value linearly decreases from 0.2 (exploration) to 0.05 (exploitation)
        p_val = 0.2 - 0.15 * progress
        p_val = max(0.05, p_val)
        
        # Pick memory indices randomly
        r_idxs = np.random.randint(0, H, curr_size)
        m_cr_vals = mem_cr[r_idxs]
        m_f_vals = mem_f[r_idxs]
        
        # Generate CR (Normal Distribution)
        cr = np.random.normal(m_cr_vals, 0.1)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F (Cauchy Distribution)
        f = m_f_vals + 0.1 * np.random.standard_cauchy(curr_size)
        f[f > 1.0] = 1.0 # Cap at 1.0
        
        # Repair non-positive F values
        # If F <= 0, regenerate until positive
        while np.any(f <= 0):
            mask = f <= 0
            f[mask] = m_f_vals[mask] + 0.1 * np.random.standard_cauchy(np.sum(mask))
            f[f > 1.0] = 1.0
            
        # 3. Mutation: current-to-pbest/1
        # Select p-best individuals (top p_val %)
        p_limit = max(2, int(p_val * curr_size))
        pbest_idxs = np.random.randint(0, p_limit, curr_size) 
        x_pbest = population[pbest_idxs]
        
        # Select r1 (distinct from i)
        r1_idxs = np.random.randint(0, curr_size, curr_size)
        for i in range(curr_size):
            while r1_idxs[i] == i:
                r1_idxs[i] = np.random.randint(0, curr_size)
        x_r1 = population[r1_idxs]
        
        # Select r2 (distinct from i and r1, chosen from Union of Pop and Archive)
        if len(archive) > 0:
            union_pop = np.vstack((population, np.array(archive)))
        else:
            union_pop = population
            
        r2_idxs = np.random.randint(0, len(union_pop), curr_size)
        for i in range(curr_size):
            while r2_idxs[i] == i or r2_idxs[i] == r1_idxs[i]:
                r2_idxs[i] = np.random.randint(0, len(union_pop))
        x_r2 = union_pop[r2_idxs]
        
        # Compute Mutant Vectors
        f_vec = f[:, np.newaxis]
        mutant = population + f_vec * (x_pbest - population) + f_vec * (x_r1 - x_r2)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(curr_size, dim)
        j_rand = np.random.randint(0, dim, curr_size)
        mask = rand_vals < cr[:, np.newaxis]
        mask[np.arange(curr_size), j_rand] = True
        
        trial = np.where(mask, mutant, population)
        
        # 5. Bound Constraints (Clipping)
        # Clipping is used here as it pushes values to the edges if the optimum is bounded
        trial = np.clip(trial, min_b, max_b)
        
        # 6. Evaluation and Selection
        success_mem = []
        diff_mem = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(curr_size):
            # Check time periodically inside the loop
            if (datetime.now() - start_time) >= limit:
                return best_val
            
            y = func(trial[i])
            
            # Greedy Selection
            if y <= fitness[i]:
                new_pop[i] = trial[i]
                new_fit[i] = y
                
                # If strictly better, update archive and memory stats
                if y < fitness[i]:
                    archive.append(population[i].copy())
                    diff_mem.append(fitness[i] - y)
                    success_mem.append((f[i], cr[i]))
                    
                if y < best_val:
                    best_val = y
                    best_sol = trial[i].copy()
                    
        population = new_pop
        fitness = new_fit
        
        # 7. Memory Update (Weighted Lehmer Mean)
        if len(diff_mem) > 0:
            diff_w = np.array(diff_mem)
            diff_w /= np.sum(diff_w) # Normalize weights
            
            s_params = np.array(success_mem)
            s_f = s_params[:, 0]
            s_cr = s_params[:, 1]
            
            # Lehmer Mean for F
            mean_f = np.sum(diff_w * (s_f**2)) / (np.sum(diff_w * s_f) + 1e-15)
            # Arithmetic Mean for CR
            mean_cr = np.sum(diff_w * s_cr)
            
            mem_f[k_mem] = mean_f
            mem_cr[k_mem] = mean_cr
            k_mem = (k_mem + 1) % H
            
        # Resort population for next generation (required for p-best)
        idxs = np.argsort(fitness)
        population = population[idxs]
        fitness = fitness[idxs]
        
        # Convergence Trigger
        # If population is small and converged, break early to do local search
        if curr_size <= 2 * min_pop_size:
            fit_range = fitness[-1] - fitness[0]
            if fit_range < 1e-9:
                break
                
    # --- Phase 2: Local Search (Coordinate Descent) ---
    # Use any remaining time to polish the single best solution found
    time_left = (start_time + limit) - datetime.now()
    if time_left.total_seconds() > 0:
        
        # Determine initial step size based on search space or current precision
        # Start reasonably small but not tiny
        step = np.mean(diff_b) * 0.005
        
        # Use a temporary working solution
        current_sol = best_sol.copy()
        current_val = best_val
        
        while datetime.now() < (start_time + limit):
            improved = False
            
            # Iterate through each dimension
            for d in range(dim):
                if datetime.now() >= (start_time + limit):
                    return best_val
                
                original_x = current_sol[d]
                
                # Try positive step
                current_sol[d] = np.clip(original_x + step, min_b[d], max_b[d])
                y = func(current_sol)
                if y < current_val:
                    current_val = y
                    if y < best_val: best_val = y
                    improved = True
                    continue # Move to next dimension
                
                # Try negative step
                current_sol[d] = np.clip(original_x - step, min_b[d], max_b[d])
                y = func(current_sol)
                if y < current_val:
                    current_val = y
                    if y < best_val: best_val = y
                    improved = True
                    continue
                
                # Revert if no improvement
                current_sol[d] = original_x
                
            # If we swept all dimensions with no improvement, reduce step size
            if not improved:
                step *= 0.5
                # Break if step size is below machine precision effectiveness
                if step < 1e-15:
                    break
                    
    return best_val
