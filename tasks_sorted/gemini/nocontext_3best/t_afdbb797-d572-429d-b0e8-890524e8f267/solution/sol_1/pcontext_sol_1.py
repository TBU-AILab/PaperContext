#The following Python code implements a **Self-Adaptive Differential Evolution (DE) algorithm with Restarts and Local Search**.
#
#**Key Improvements over standard DE:**
#1.  **Strategy `DE/current-to-best/1/bin`**: This mutation strategy converges faster than random search or standard DE by guiding individuals towards the best solution found so far while maintaining exploration through difference vectors.
#2.  **Restart Mechanism**: If the population converges (standard deviation drops), the algorithm triggers a restart. This prevents getting stuck in local optima and allows checking multiple basins of attraction within the time limit.
#3.  **Local Polish (Memetic Algorithm)**: Before restarting, a lightweight Coordinate Descent (local search) is performed on the best candidate to "squeeze" the maximum precision out of the solution.
#4.  **Adaptive Population**: The population size is dynamically scaled based on the dimension of the problem to balance throughput and diversity.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function 'func' within 'bounds' and 'max_time' using 
    Differential Evolution with Restarts and Local Search polishing.
    """
    
    # --- Timing Setup ---
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Parameters ---
    # Population size: Adaptive to dimension (10*dim), clamped between 20 and 50
    # This balances exploration capability with generation speed.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # Crossover Probability
    CR = 0.9 
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # Initialize random population within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_idx = -1
    
    # Evaluate initial population
    for i in range(pop_size):
        if (datetime.now() - start_time) >= limit:
            return best_fitness
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # --- Main Optimization Loop ---
    while True:
        # Check time at the start of generation
        if (datetime.now() - start_time) >= limit:
            return best_fitness
            
        # 1. Mutation: DE/current-to-best/1
        # Formula: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
        # F is "dithered" (randomized) between 0.5 and 1.0 to improve diversity
        F = 0.5 + 0.5 * np.random.rand()
        
        # Select random indices r1, r2 distinct from each other and current index
        idxs = np.arange(pop_size)
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Fix collisions (simple retry logic)
        mask = (r1 == idxs)
        while np.any(mask):
            r1[mask] = np.random.randint(0, pop_size, np.sum(mask))
            mask = (r1 == idxs)
            
        mask = (r2 == idxs) | (r2 == r1)
        while np.any(mask):
            r2[mask] = np.random.randint(0, pop_size, np.sum(mask))
            mask = (r2 == idxs) | (r2 == r1)
            
        # Calculate Mutant Vectors (Vectorized)
        x_best = pop[best_idx]
        mutant = pop + F * (x_best - pop) + F * (pop[r1] - pop[r2])
        
        # 2. Crossover: Binomial
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR
        
        # Ensure at least one parameter is taken from mutant (DE standard requirement)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[idxs, j_rand] = True
        
        # Create Trial Population
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 3. Bound Handling: Clip to bounds
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 4. Selection & Evaluation
        for i in range(pop_size):
            # Strict time check inside evaluation loop
            if (datetime.now() - start_time) >= limit:
                return best_fitness
            
            trial_val = func(trial_pop[i])
            
            # Greedy selection: if trial is better, overwrite parent
            if trial_val < fitness[i]:
                fitness[i] = trial_val
                pop[i] = trial_pop[i]
                
                if trial_val < best_fitness:
                    best_fitness = trial_val
                    best_idx = i
        
        # 5. Convergence Check & Restart Strategy
        # If population diversity (std dev) is too low, we are likely in a local optimum.
        # Perform local search on the best point, then restart the population.
        std_dev = np.std(fitness)
        
        if std_dev < 1e-7:
            # --- Local Search (Coordinate Descent) ---
            # Attempt to refine the best solution found so far
            current_vec = pop[best_idx].copy()
            current_val = best_fitness
            # Initial step size relative to domain
            step_size = 0.05 
            
            ls_improved = True
            # Run local search until no improvement or step size becomes negligible
            while ls_improved and (datetime.now() - start_time) < limit:
                ls_improved = False
                for d in range(dim):
                    if (datetime.now() - start_time) >= limit: break
                    
                    old_val = current_vec[d]
                    step = step_size * diff_b[d]
                    
                    # Try increasing parameter
                    current_vec[d] = np.clip(old_val + step, min_b[d], max_b[d])
                    val = func(current_vec)
                    if val < current_val:
                        current_val = val
                        ls_improved = True
                        continue 
                    
                    # Try decreasing parameter
                    current_vec[d] = np.clip(old_val - step, min_b[d], max_b[d])
                    val = func(current_vec)
                    if val < current_val:
                        current_val = val
                        ls_improved = True
                        continue
                        
                    # Revert if no improvement
                    current_vec[d] = old_val
                
                # If a full pass didn't improve, shrink step size to look closer
                if not ls_improved and step_size > 1e-6:
                    step_size *= 0.2
                    ls_improved = True 
            
            # Update global best if local search helped
            if current_val < best_fitness:
                best_fitness = current_val
                
            # --- Restart ---
            # Re-initialize population to find new basins of attraction
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Inject the best found solution into the new population (Elitism)
            pop[0] = current_vec
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = current_val
            best_idx = 0
            
            # Evaluate new random individuals
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= limit:
                    return best_fitness
                v = func(pop[i])
                fitness[i] = v
                if v < best_fitness:
                    best_fitness = v
                    best_idx = i

    return best_fitness
