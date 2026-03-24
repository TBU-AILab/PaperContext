#Here is a fully operational, self-contained implementation using a **Hybrid Restarting Differential Evolution with Local Search Polishing**.
#
#**Why this is an improvement:**
#1.  **Hybridization**: It combines the global search capabilities of Differential Evolution (DE) with a coordinate-descent-based **Local Search**. DE finds the promising "basin," and Local Search refines the solution to high precision.
#2.  **Restart Mechanism**: Standard DE can stagnate (converge to a local minimum). This algorithm detects stagnation (low population variance) and triggers a **Restart**, preserving the best solution while scattering the rest of the population to find potentially better global minima.
#3.  **Strategy**: It uses the **"current-to-best/1/bin"** mutation strategy, which offers a better balance between convergence speed and robustness compared to the standard `rand/1/bin` used in the previous example.
#4.  **Dithering**: The mutation factor `F` is randomized (dithered) per generation, which helps escape local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Differential Evolution with 
    Coordinate Descent Local Search polishing.
    """
    t0 = time.time()
    
    # --- Configuration ---
    # Population size: Good balance for exploration vs speed
    pop_size = max(15, int(10 * dim))
    # Mutation strategy parameters (Self-Adaptive logic simplified via Dithering)
    CR = 0.9  # Crossover Probability
    
    # Bounds preparation
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # State variables
    global_best_val = float('inf')
    global_best_x = None
    
    # Helper to check time budget
    def time_remaining():
        return max_time - (time.time() - t0)

    # Helper: Local Search (Coordinate Descent)
    # Refines a solution x_curr to find local minimum
    def local_search(x_start, current_val):
        x_curr = x_start.copy()
        best_local_val = current_val
        
        # Initial step size relative to domain size
        step_size = 0.05 * diff_b
        min_step = 1e-8
        
        # While we have time and steps are significant
        while np.max(step_size) > min_step:
            if time_remaining() < 0.05: # Reserve tiny buffer
                break
                
            improved = False
            # Iterate through dimensions (Coordinate Descent)
            for d in range(dim):
                original_val = x_curr[d]
                
                # Try positive step
                x_curr[d] = np.clip(original_val + step_size[d], min_b[d], max_b[d])
                val = func(x_curr)
                
                if val < best_local_val:
                    best_local_val = val
                    improved = True
                    continue # Keep change and move to next dim
                
                # Try negative step
                x_curr[d] = np.clip(original_val - step_size[d], min_b[d], max_b[d])
                val = func(x_curr)
                
                if val < best_local_val:
                    best_local_val = val
                    improved = True
                    continue
                
                # Revert if neither improved
                x_curr[d] = original_val
                
            # Decrease step size if no improvement found in this sweep
            if not improved:
                step_size *= 0.5
                
        return x_curr, best_local_val

    # --- Initialization ---
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initial Evaluation
    for i in range(pop_size):
        if time_remaining() <= 0: return global_best_val
        val = func(population[i])
        fitness[i] = val
        if val < global_best_val:
            global_best_val = val
            global_best_x = population[i].copy()

    # --- Main Loop ---
    while time_remaining() > 0.1: # Leave small buffer for final return
        
        # Dithering: Randomize F slightly to prevent stagnation [0.5, 1.0]
        F = 0.5 + 0.5 * np.random.rand() 
        
        # 1. Mutation: current-to-best/1
        # V = X + F * (Best - X) + F * (r1 - r2)
        # This strategy pulls population towards the best found so far
        
        # Select random indices
        idxs = np.arange(pop_size)
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Ensure distinctness (approximate for speed) using shift if r1==i or r2==i
        # In high-speed Python DE, strict distinctness checks inside loops are costly.
        # The vector math robustness usually compensates for occasional self-selection.
        
        # Compute difference vectors
        diff_best = global_best_x - population
        diff_rand = population[r1] - population[r2]
        
        mutants = population + F * diff_best + F * diff_rand
        mutants = np.clip(mutants, min_b, max_b)
        
        # 2. Crossover (Binomial)
        cross_mask = np.random.rand(pop_size, dim) < CR
        trials = np.where(cross_mask, mutants, population)
        
        # 3. Selection & Evaluation
        improved_any = False
        for i in range(pop_size):
            if time_remaining() <= 0: return global_best_val
            
            f_trial = func(trials[i])
            
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                population[i] = trials[i]
                
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_x = trials[i].copy()
                    improved_any = True
        
        # 4. Convergence Check & Restart Logic
        # If population diversity is very low, we are stuck in a basin.
        # Solution: Perform local search on the best, then restart the population.
        std_dev = np.std(fitness)
        range_fit = np.max(fitness) - np.min(fitness)
        
        is_converged = std_dev < 1e-6 or range_fit < 1e-6
        
        if is_converged:
            # Polishing: Squeeze the last bit of precision from the best solution
            polished_x, polished_val = local_search(global_best_x, global_best_val)
            if polished_val < global_best_val:
                global_best_val = polished_val
                global_best_x = polished_x
            
            # Restart Population (Scattered Restart)
            # Re-initialize population uniformly
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            fitness = np.full(pop_size, float('inf'))
            
            # Keep the best found so far in the new population (Elitism)
            population[0] = global_best_x
            fitness[0] = global_best_val
            
            # Re-evaluate the rest (or lazy eval in next loop, 
            # but better to have valid fitness for mutation logic immediately)
            for i in range(1, pop_size):
                if time_remaining() <= 0: return global_best_val
                val = func(population[i])
                fitness[i] = val
                if val < global_best_val: # Should rarely happen immediately
                    global_best_val = val
                    global_best_x = population[i].copy()

    # Final Polish if time permits and we haven't just restarted
    if time_remaining() > 0:
        polished_x, polished_val = local_search(global_best_x, global_best_val)
        if polished_val < global_best_val:
            global_best_val = polished_val

    return global_best_val
