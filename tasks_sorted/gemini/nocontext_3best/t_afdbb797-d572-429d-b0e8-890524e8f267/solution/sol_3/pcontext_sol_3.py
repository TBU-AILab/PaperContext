#The output values and codes for the best generated algorithms are as follows:
#1. output value is: 27.263006486680595
#...
#2. output value is: 36.98206262443294
#...
#3. output value is: 45.279237607066776
#...
#
#The following Python code implements a **Self-Adaptive Differential Evolution (jDE) with Restart Mechanism**.
#
#**Key Improvements:**
#1.  **Self-Adaptation (jDE)**: Unlike standard DE (fixed parameters) or complex SHADE (history-based), this uses a lightweight mechanism where each individual adapts its own mutation (`F`) and crossover (`CR`) rates. This allows the algorithm to dynamically fit the optimization landscape with minimal computational overhead.
#2.  **Robust Strategy (`DE/rand/1/bin`)**: While greedy strategies like `current-to-best` can converge faster, they often trap in local optima (evidenced by the higher scores in algorithms 2 & 3). We use `rand/1` for better population diversity, relying on parameter adaptation for speed.
#3.  **Reflective Bound Handling**: Instead of simple clipping (which stacks solutions on the edge of the search space), we use reflection. If a particle hits a bound, it "bounces" back, preserving the statistical distribution of the population.
#4.  **Restart Mechanism**: If the population converges (low variance in fitness), the algorithm triggers a "soft restart". It keeps the best solution found so far (Elitism) and re-initializes the rest of the population to explore new basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population: Adaptive based on dimension, clamped to ensuring speed
    # A size of 20-50 is empirically robust for short-time execution
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # jDE Control Parameters (Self-Adaptive)
    # Initialize F ~ U(0.1, 1.0) and CR ~ U(0.0, 1.0)
    F = np.random.uniform(0.1, 1.0, pop_size)
    CR = np.random.uniform(0.0, 1.0, pop_size)
    
    # Pre-process bounds for vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best = float('inf')
    best_sol = np.zeros(dim)
    
    # Evaluate initial population
    for i in range(pop_size):
        if (datetime.now() - start_time) >= limit:
            return best
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best:
            best = val
            best_sol = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Strict Time Check
        if (datetime.now() - start_time) >= limit:
            return best
            
        # 1. Parameter Adaptation (jDE Logic)
        # Probabilities to reset F and CR
        tau1 = 0.1
        tau2 = 0.1
        
        # Create masks for updates
        mask_f = np.random.rand(pop_size) < tau1
        mask_cr = np.random.rand(pop_size) < tau2
        
        # Prepare new parameters
        F_new = F.copy()
        CR_new = CR.copy()
        
        # Update F: rand(0.1, 1.0)
        if np.any(mask_f):
            F_new[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
        
        # Update CR: rand(0.0, 1.0)
        if np.any(mask_cr):
            CR_new[mask_cr] = np.random.rand(np.sum(mask_cr))
            
        # 2. Mutation Strategy: DE/rand/1
        # Select r1, r2, r3 distinct from i and each other
        idxs = np.arange(pop_size)
        
        # Efficient distinct index selection via shifting
        r1 = np.random.randint(0, pop_size, pop_size)
        mask = (r1 == idxs)
        while np.any(mask):
            r1[mask] = np.random.randint(0, pop_size, np.sum(mask))
            mask = (r1 == idxs)
            
        r2 = np.random.randint(0, pop_size, pop_size)
        mask = (r2 == idxs) | (r2 == r1)
        while np.any(mask):
            r2[mask] = np.random.randint(0, pop_size, np.sum(mask))
            mask = (r2 == idxs) | (r2 == r1)
            
        r3 = np.random.randint(0, pop_size, pop_size)
        mask = (r3 == idxs) | (r3 == r1) | (r3 == r2)
        while np.any(mask):
            r3[mask] = np.random.randint(0, pop_size, np.sum(mask))
            mask = (r3 == idxs) | (r3 == r1) | (r3 == r2)
            
        # Calculate Mutant Vector: V = X_r1 + F * (X_r2 - X_r3)
        diffs = pop[r2] - pop[r3]
        mutant = pop[r1] + F_new[:, np.newaxis] * diffs
        
        # 3. Crossover: Binomial
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR_new[:, np.newaxis]
        
        # Force at least one dimension from mutant (Standard DE rule)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[idxs, j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 4. Bound Handling: Reflection
        # Reflects particles back into the domain rather than clipping to edges
        for k in range(dim):
            # Lower bound
            mask_l = trial_pop[:, k] < min_b[k]
            if np.any(mask_l):
                trial_pop[mask_l, k] = 2 * min_b[k] - trial_pop[mask_l, k]
                # Fail-safe clip if reflection is still out
                mask_l2 = trial_pop[:, k] < min_b[k]
                trial_pop[mask_l2, k] = min_b[k]
            
            # Upper bound
            mask_u = trial_pop[:, k] > max_b[k]
            if np.any(mask_u):
                trial_pop[mask_u, k] = 2 * max_b[k] - trial_pop[mask_u, k]
                mask_u2 = trial_pop[:, k] > max_b[k]
                trial_pop[mask_u2, k] = max_b[k]
                
        # 5. Selection
        for i in range(pop_size):
            if (datetime.now() - start_time) >= limit:
                return best
            
            f_trial = func(trial_pop[i])
            
            # Greedy selection
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                # If trial was successful, adopt the new parameters
                F[i] = F_new[i]
                CR[i] = CR_new[i]
                
                if f_trial < best:
                    best = f_trial
                    best_sol = pop[i].copy()
            # Else: Discard trial, keep old parameters (implicit in not updating)
            
        # 6. Restart Mechanism
        # If population diversity is lost, we are likely stuck.
        # Check standard deviation of fitness values.
        if np.std(fitness) < 1e-6:
            # Re-initialize population to find new basins
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Elitism: Keep the best solution found so far
            pop[0] = best_sol
            
            # Reset Parameters
            F = np.random.uniform(0.1, 1.0, pop_size)
            CR = np.random.uniform(0.0, 1.0, pop_size)
            
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best
            
            # Re-evaluate (skipping the elite at index 0)
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= limit:
                    return best
                val = func(pop[i])
                fitness[i] = val
                if val < best:
                    best = val
                    best_sol = pop[i].copy()
    
    return best
