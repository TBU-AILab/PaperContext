#Here is a fully operational, self-contained implementation using a **Self-Adaptive Differential Evolution (jDE) with Restart Mechanism**.
#
#**Improvements over previous version:**
#1.  **Restarts**: The previous result (~42) suggests entrapment in a local optimum. This algorithm detects stagnation (low variance or no improvement) and restarts the population (preserving the elite solution) to explore new areas.
#2.  **Greedy Strategy**: Uses `DE/current-to-best/1/bin`. This strategy converges faster than `rand/1`, which is crucial for getting good results within a `max_time` limit.
#3.  **Self-Adaptation (jDE)**: Automatically tunes the mutation factor (`F`) and crossover rate (`CR`) for each individual during the search, removing the need for manual hyperparameter guessing.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE)
    with a Restart mechanism to escape local optima.
    
    Strategy: DE/current-to-best/1/bin
    - Exploits the gradient towards the best solution found so far.
    - Adapts F and CR parameters dynamically.
    - Restarts population if convergence is detected.
    """
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population size: Adaptive to dimension
    # We clip between 20 and 100 to balance exploration with iteration speed
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # Allocate arrays
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # jDE Adaptive Parameters (F: Mutation, CR: Crossover)
    # Initialize with standard values
    f_vals = np.full(pop_size, 0.5)
    cr_vals = np.full(pop_size, 0.9)
    
    # Global best tracking
    best_val = float('inf')
    best_idx = -1
    best_sol = None
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if datetime.now() - start_time >= limit:
            return best_val if best_idx != -1 else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i
            best_sol = pop[i].copy()
            
    # Restart triggers
    generations_without_improv = 0
    restart_threshold = 25  # Restart if no improvement for N gens
    min_variance = 1e-8     # Restart if population converged (std dev of fitness)
    
    # --- Main Loop ---
    while True:
        # Check time strictness at start of generation
        if datetime.now() - start_time >= limit:
            return best_val
            
        # --- Restart Logic ---
        # Calculate spread of fitness to detect convergence
        current_std = np.std(fitness)
        
        should_restart = (current_std < min_variance) or \
                         (generations_without_improv >= restart_threshold)
        
        if should_restart:
            # Perform soft restart: Keep elite, randomize others
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol # Preserve elite solution at index 0
            
            # Reset Adaptive params to defaults
            f_vals = np.full(pop_size, 0.5)
            cr_vals = np.full(pop_size, 0.9)
            
            # Re-evaluate new population (skip index 0 which is elite)
            fitness[0] = best_val
            best_idx = 0
            
            for i in range(1, pop_size):
                if datetime.now() - start_time >= limit:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_idx = i
                    best_sol = pop[i].copy()
            
            generations_without_improv = 0
            continue # Skip to next loop iteration immediately
            
        # --- jDE Parameter Adaptation ---
        # Create trial parameters based on probabilities
        mask_f = np.random.rand(pop_size) < 0.1
        mask_cr = np.random.rand(pop_size) < 0.1
        
        trial_f = f_vals.copy()
        trial_cr = cr_vals.copy()
        
        # Update masked values with new random parameters
        trial_f[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
        trial_cr[mask_cr] = np.random.rand(np.sum(mask_cr))
        
        # --- Mutation: DE/current-to-best/1 ---
        # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
        
        # Select random indices r1, r2
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        x_best = pop[best_idx]
        
        # Compute difference vectors
        diff_to_best = x_best - pop
        diff_rand = pop[r1] - pop[r2]
        
        # Compute mutant vectors (vectorized)
        # F is broadcasted: (pop_size, 1) * (pop_size, dim)
        mutant = pop + trial_f[:, None] * diff_to_best + trial_f[:, None] * diff_rand
        
        # --- Crossover: Binomial ---
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < trial_cr[:, None]
        
        # Ensure at least one parameter is taken from mutant to avoid exact copies
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # --- Boundary Handling ---
        # Clip values to stay within bounds
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # --- Selection ---
        improved = False
        for i in range(pop_size):
            # Strict time check inside evaluation loop
            if datetime.now() - start_time >= limit:
                return best_val
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                # Accept trial solution
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                # Adopt successful parameters for next generation
                f_vals[i] = trial_f[i]
                cr_vals[i] = trial_cr[i]
                
                # Update Global Best
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i
                    best_sol = pop[i].copy()
                    improved = True
        
        if improved:
            generations_without_improv = 0
        else:
            generations_without_improv += 1
            
    return best_val
