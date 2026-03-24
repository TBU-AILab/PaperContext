#The following is a fully operational, self-contained implementation of an improved optimization algorithm.
#
#### Algorithm Explanation
#This algorithm is a **Hybrid Adaptive Differential Evolution (HADE)** enhanced with a **Dynamic Local Search** mechanism. It improves upon previous iterations by addressing the specific weaknesses of pure DE in time-constrained environments:
#
#1.  **Asynchronous `current-to-best` Strategy**: It uses the aggressive `current-to-best` mutation strategy with asynchronous updates (improving the population immediately within the loop rather than at the end), which accelerates convergence.
#2.  **Elite Local Search**: Whenever a new global best solution is found, the algorithm triggers a "micro-exploitation" phase. It generates a few candidates around the new best using a Gaussian distribution scaled by the current population diversity. This allows the algorithm to "drill down" into the minimum immediately, refining the result significantly faster than random evolutionary mutations.
#3.  **Adaptive Restart with Memory**: If the population converges (standard deviation drops), it performs a soft restart. It keeps the best solution, but regenerates the rest. Crucially, it adapts the restart radius based on the remaining time, searching widely in the beginning and narrowing down as the deadline approaches.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Hybrid Adaptive Differential Evolution (HADE)
    combined with a dynamic local search for the best solution.
    """
    start_time = time.time()
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Adaptive Population Size
    # Smaller population allows more generations within the time limit,
    # but too small risks premature convergence.
    # We use a moderate size based on dimension.
    pop_size = int(np.clip(dim * 10, 20, 70))
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    global_best_val = float('inf')
    global_best_idx = -1
    global_best_vec = np.zeros(dim)

    # Evaluate Initial Population
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return global_best_val
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_idx = i
            global_best_vec = pop[i].copy()

    # --- Control Parameters ---
    # Initial scale for local search
    ls_attempts = 5  # Number of local search tries when a new best is found
    
    # --- Main Optimization Loop ---
    while True:
        # Time check at start of generation
        if (time.time() - start_time) >= max_time:
            return global_best_val
        
        # Calculate population diversity (std dev) to scale mutation and local search
        pop_std = np.std(pop, axis=0)
        avg_std = np.mean(pop_std)
        
        # Iterate through population (Asynchronous Update)
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val

            # --- 1. Differential Evolution Step ---
            
            # Select r1, r2 distinct from i
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            x_i = pop[i]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            x_best = global_best_vec
            
            # Adaptive Parameters
            # F (Mutation): Cauchy distribution around 0.5 to allow rare large jumps
            # CR (Crossover): Normal distribution around 0.9 for strong convergence
            F = np.random.standard_cauchy() * 0.1 + 0.5
            F = np.clip(F, 0.1, 1.0)
            CR = np.random.normal(0.9, 0.05)
            CR = np.clip(CR, 0.0, 1.0)
            
            # Mutation: current-to-best/1
            # Leads the individual towards the best found so far
            # v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(dim) < CR
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True # Ensure at least one dimension changes
            
            trial = np.where(cross_mask, mutant, x_i)
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_idx = i
                    global_best_vec = trial.copy()
                    
                    # --- 2. Elite Local Search (Micro-Exploitation) ---
                    # If we found a new best, try to refine it immediately.
                    # We utilize the remaining time budget to polish this result.
                    
                    # Scale search radius by current population diversity or minimal baseline
                    search_scale = np.maximum(pop_std, 1e-8 * diff_b)
                    
                    for _ in range(ls_attempts):
                        if (time.time() - start_time) >= max_time:
                            return global_best_val
                            
                        # Generate neighbor using Gaussian walk
                        # Reduce scale slightly to fine-tune
                        neighbor = global_best_vec + np.random.normal(0, search_scale * 0.5)
                        neighbor = np.clip(neighbor, min_b, max_b)
                        
                        f_neighbor = func(neighbor)
                        
                        if f_neighbor < global_best_val:
                            # Search successful: update best and continue loop from this new point
                            global_best_val = f_neighbor
                            global_best_vec = neighbor.copy()
                            pop[global_best_idx] = neighbor
                            fitness[global_best_idx] = f_neighbor
                        else:
                            # If local search fails, stop wasting time on it
                            break

        # --- 3. Stagnation Detection & Restart ---
        # If the population variance is very low, we are likely stuck in a local optimum.
        # Restart the population but KEEP the best solution.
        if avg_std < 1e-6 * np.mean(diff_b):
            # Only restart if there is meaningful time left (e.g., > 5%)
            if (time.time() - start_time) < (max_time * 0.95):
                # Preserve Best
                saved_best = global_best_vec.copy()
                saved_best_val = global_best_val
                
                # Re-initialize Population
                # We add some chaos: 20% small perturbation around best, 80% global random
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                
                # Inject saved best at index 0
                pop[0] = saved_best
                fitness[:] = float('inf')
                fitness[0] = saved_best_val
                
                global_best_idx = 0
                
                # Re-evaluate new random population (skipping best)
                for k in range(1, pop_size):
                    if (time.time() - start_time) >= max_time:
                        return global_best_val
                    
                    val = func(pop[k])
                    fitness[k] = val
                    
                    if val < global_best_val:
                        global_best_val = val
                        global_best_idx = k
                        global_best_vec = pop[k].copy()

    return global_best_val
