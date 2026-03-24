#The following is a fully operational and self-contained Python algorithm. It implements a **Self-Adaptive Differential Evolution (jDE)** strategy with a **Restart Mechanism**.
#
#### Key Improvements:
#1.  **Self-Adaptation (jDE)**: Instead of fixed mutation ($F$) and crossover ($CR$) parameters, each individual adapts its own parameters during the search. This allows the algorithm to dynamically tune itself to the function landscape.
#2.  **Restart Mechanism**: If the population converges (standard deviation of fitness becomes negligible) before the time limit expires, the algorithm restarts the population while preserving the best solution found so far. This prevents getting stuck in local optima and utilizes the full available time for exploration.
#3.  **Strict Time Management**: Time is checked frequently within the inner loops to ensuring the result is returned within `max_time`.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes the output of 'func' using Self-Adaptive Differential Evolution (jDE)
    with a Restart Mechanism within 'max_time'.
    """
    start_time = datetime.now()
    # Subtract a small buffer to ensure we return before hard timeout
    end_time = start_time + timedelta(seconds=max_time - 0.05)

    # --- Hyperparameters ---
    # Population size: Scale with dimension, clamped to reasonable limits.
    # A size of 10*dim is standard, bounded to ensure generation throughput.
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # jDE Control Parameters (Initialized)
    # F: Mutation factor, CR: Crossover probability
    # These are arrays as they adapt per individual
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_idx = -1
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= end_time:
            # If time runs out during initialization, return best found so far
            return best_fitness if best_fitness != float('inf') else float('inf')
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_idx = i

    # --- Main Optimization Loop ---
    while datetime.now() < end_time:
        
        # 1. Restart Mechanism
        # If population diversity is lost (converged), restart to search other basins.
        if np.std(fitness) < 1e-6 and pop_size > 5:
            # Save the Elite (Global Best)
            elite_pos = pop[best_idx].copy()
            elite_fit = best_fitness
            
            # Re-initialize population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Inject Elite at index 0
            pop[0] = elite_pos
            
            # Reset Adaptive Parameters
            F = np.full(pop_size, 0.5)
            CR = np.full(pop_size, 0.9)
            
            # Reset Fitness array and re-evaluate (skipping elite)
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = elite_fit
            best_idx = 0
            
            for i in range(1, pop_size):
                if datetime.now() >= end_time: return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_idx = i
            
            # Start new generation immediately after restart
            continue

        # 2. jDE Parameter Adaptation Step
        # Generate new candidate F and CR values for this generation
        mask_f = np.random.rand(pop_size) < 0.1
        mask_cr = np.random.rand(pop_size) < 0.1
        
        F_new = F.copy()
        CR_new = CR.copy()
        
        # F evolves in [0.1, 1.0]
        F_new[mask_f] = 0.1 + np.random.rand(np.sum(mask_f)) * 0.9
        # CR evolves in [0.0, 1.0]
        CR_new[mask_cr] = np.random.rand(np.sum(mask_cr))
        
        # 3. Evolution Loop
        for i in range(pop_size):
            if datetime.now() >= end_time:
                return best_fitness
            
            # Mutation (DE/rand/1)
            # Pick 3 distinct random individuals excluding i
            # Using rejection sampling for indices (efficient for pop_size >= 20)
            idxs = np.random.randint(0, pop_size, 3)
            while i in idxs or idxs[0]==idxs[1] or idxs[0]==idxs[2] or idxs[1]==idxs[2]:
                idxs = np.random.randint(0, pop_size, 3)
            
            a, b, c = idxs
            
            mutant = pop[a] + F_new[i] * (pop[b] - pop[c])
            
            # Crossover (Binomial)
            cross_mask = np.random.rand(dim) < CR_new[i]
            # Ensure at least one parameter changes
            if not np.any(cross_mask):
                cross_mask[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_mask, mutant, pop[i])
            
            # Bound Constraint Handling (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Trial is better or equal: accept it
                fitness[i] = f_trial
                pop[i] = trial
                # Update individual's control parameters
                F[i] = F_new[i]
                CR[i] = CR_new[i]
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_idx = i
            # If trial is worse, discard trial and keep old F/CR
            
    return best_fitness
