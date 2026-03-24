#To improve the performance and robustness of the optimization, especially for multimodal functions where basic Differential Evolution (DE) might stagnate (as suggested by the previous result of ~49.5), I propose a **Self-Adaptive Differential Evolution (jDE)** algorithm with a **Restart Strategy**.
#
#### Key Improvements:
#1.  **Self-Adaptation (jDE):** Instead of fixed parameters $F$ and $CR$, each individual carries its own parameters. These are adapted during the search: successful parameters that lead to better offspring are propagated, while others are re-sampled. This removes the need to manually tune hyperparameters.
#2.  **Current-to-Best Mutation:** The algorithm uses the `current-to-best/1/bin` strategy. This guides individuals towards the best solution found so far while maintaining exploration, significantly speeding up convergence compared to random search strategies.
#3.  **Restart Mechanism:** If the population converges (low diversity) or stagnates before the time limit is reached, the algorithm saves the best solution and restarts the rest of the population. This helps escape local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE) 
    with a 'current-to-best' mutation strategy and restart mechanism.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: Adaptive based on dimension, bounded to reasonable limits
    # Small enough for fast generations, large enough for diversity
    pop_size = int(max(20, min(100, 20 * dim)))
    
    # Restart tolerance: threshold for population standard deviation
    restart_tol = 1e-6
    
    # jDE Adaptation probabilities
    tau_F = 0.1
    tau_CR = 0.1
    
    # Pre-process bounds
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # --- Helper: Initialization ---
    def init_pop(size):
        return min_b + np.random.rand(size, dim) * diff_b

    # --- Initial State ---
    pop = init_pop(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize control parameters F (scale) and CR (crossover rate) for each individual
    # F ~ U(0.1, 1.0), CR ~ U(0.0, 1.0) usually start conservatively
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    best_idx = -1
    best_val = float('inf')
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i

    # --- Main Optimization Loop ---
    while True:
        # Check time budget
        if datetime.now() - start_time >= time_limit:
            return best_val

        # --- 1. Restart Check ---
        # Calculate diversity (mean std dev across dimensions)
        pop_std = np.mean(np.std(pop, axis=0))
        if pop_std < restart_tol:
            # Save best solution
            best_sol_vector = pop[best_idx].copy()
            
            # Re-initialize population
            pop = init_pop(pop_size)
            pop[0] = best_sol_vector  # Elitism: keep the best
            
            # Reset adaptation parameters
            F = np.full(pop_size, 0.5)
            CR = np.full(pop_size, 0.9)
            
            # Re-evaluate (skip index 0 as it is known)
            fitness[0] = best_val
            best_idx = 0
            
            for i in range(1, pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_idx = i
            continue

        # --- 2. Parameter Adaptation (jDE) ---
        # Generate trial F and CR values
        mask_F = np.random.rand(pop_size) < tau_F
        mask_CR = np.random.rand(pop_size) < tau_CR
        
        F_trial = F.copy()
        CR_trial = CR.copy()
        
        # F takes value in [0.1, 1.0], CR in [0.0, 1.0]
        if np.any(mask_F):
            F_trial[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
        if np.any(mask_CR):
            CR_trial[mask_CR] = np.random.rand(np.sum(mask_CR))
            
        # --- 3. Mutation: Current-to-Best/1 ---
        # v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
        
        # Select r1, r2 indices distinct from i
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Simple collision repair
        for i in range(pop_size):
            while r1[i] == i:
                r1[i] = np.random.randint(0, pop_size)
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, pop_size)
                
        x_best = pop[best_idx]
        x_r1 = pop[r1]
        x_r2 = pop[r2]
        
        # Reshape F for broadcasting
        F_col = F_trial[:, None]
        
        # Compute mutant vectors (vectorized)
        mutant = pop + F_col * (x_best - pop) + F_col * (x_r1 - x_r2)
        
        # --- 4. Crossover ---
        # Binomial crossover
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR_trial[:, None]
        
        # Guaranteed parameter inheritance (at least one dimension from mutant)
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # Clip to bounds
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # --- 5. Selection ---
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
            
            trial_val = func(trial_pop[i])
            
            if trial_val <= fitness[i]:
                # Trial is better: replace parent
                pop[i] = trial_pop[i]
                fitness[i] = trial_val
                
                # Adopt the successful parameters
                F[i] = F_trial[i]
                CR[i] = CR_trial[i]
                
                # Update global best
                if trial_val < best_val:
                    best_val = trial_val
                    best_idx = i
    
    return best_val
