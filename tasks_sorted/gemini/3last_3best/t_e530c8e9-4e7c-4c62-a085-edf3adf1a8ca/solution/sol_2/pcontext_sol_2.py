#Here is a fully operational, self-contained implementation of an improved optimization algorithm.
#
#**Algorithm Explanation:**
#This implementation employs a **Self-Adaptive Differential Evolution (jDE)** algorithm enhanced with a **"current-to-pbest"** mutation strategy.
#1.  **Current-to-pbest Mutation**: Instead of moving towards the single best solution (which risks premature convergence), individuals move towards a random solution selected from the top $p\%$ (p-best) of the population. This strikes a superior balance between exploitation and exploration.
#2.  **Self-Adaptation (jDE)**: The control parameters $F$ (mutation factor) and $CR$ (crossover rate) are not fixed. Each individual carries its own $F$ and $CR$ values, which evolve along with the solution. Successful parameter values are propagated to the next generation, automatically tuning the algorithm to the specific function landscape.
#3.  **Stagnation Detection & Restart**: The algorithm monitors population diversity. If the population converges (fitness variance drops near zero), it triggers a restart mechanism that preserves the global best solution while re-initializing the rest of the population to search new areas, ensuring the entire time budget is utilized effectively.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE)
    with 'current-to-pbest' mutation strategy and automatic restarts.
    """
    start_time = time.time()
    
    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population Size
    # Adaptive size based on dimension, clamped to [30, 100] to balance
    # exploration capability with computational speed within the time limit.
    pop_size = int(np.clip(dim * 10, 30, 100))
    
    # Initialize Population
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize jDE Control Parameters (one per individual)
    # F (Mutation): Initialized around 0.5
    # CR (Crossover): Initialized around 0.9
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Strategy Parameter: Top percentage for p-best selection (e.g., 15%)
    p_best_rate = 0.15 
    
    global_best_val = float('inf')
    
    # Evaluate initial population
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return global_best_val
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            
    # Sort population by fitness (Indices 0..N are Best..Worst)
    # Sorting is required for efficient p-best selection
    sort_idx = np.argsort(fitness)
    pop = pop[sort_idx]
    fitness = fitness[sort_idx]
    F = F[sort_idx]
    CR = CR[sort_idx]
    
    # --- Main Optimization Loop ---
    while True:
        # Strict time check
        if (time.time() - start_time) >= max_time:
            return global_best_val
            
        # Prepare arrays for the next generation
        next_pop = np.copy(pop)
        next_fitness = np.copy(fitness)
        next_F = np.copy(F)
        next_CR = np.copy(CR)
        
        # Calculate size of the 'p-best' pool
        p_num = max(1, int(pop_size * p_best_rate))
        
        # Iterate over each individual
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return global_best_val
            
            # 1. jDE Parameter Adaptation
            # Update F with probability 0.1
            if np.random.rand() < 0.1:
                # New F in range [0.1, 1.0]
                new_F_val = 0.1 + 0.9 * np.random.rand()
            else:
                new_F_val = F[i]
            
            # Update CR with probability 0.1
            if np.random.rand() < 0.1:
                # New CR in range [0.0, 1.0]
                new_CR_val = np.random.rand()
            else:
                new_CR_val = CR[i]
            
            # 2. Mutation: current-to-pbest/1
            # Vector = Current + F * (P_best - Current) + F * (r1 - r2)
            
            # Select x_pbest randomly from the top p% individuals
            p_idx = np.random.randint(0, p_num)
            x_pbest = pop[p_idx]
            
            # Select r1, r2 distinct from current index i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
                
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
                
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            x_i = pop[i]
            
            # Calculate mutant vector
            # Vectorized operations for speed
            diff_pbest = x_pbest - x_i
            diff_r = x_r1 - x_r2
            mutant = x_i + new_F_val * diff_pbest + new_F_val * diff_r
            
            # 3. Crossover (Binomial)
            cross_mask = np.random.rand(dim) < new_CR_val
            # Guarantee at least one dimension is changed
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # 4. Bound Constraint Handling (Clipping)
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection (Greedy)
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                # Improvement: Accept new vector and new parameters
                next_pop[i] = trial
                next_fitness[i] = f_trial
                next_F[i] = new_F_val
                next_CR[i] = new_CR_val
                
                if f_trial < global_best_val:
                    global_best_val = f_trial
            else:
                # No improvement: Keep old vector and old parameters (implicit in next_* init)
                pass
        
        # Advance generation
        pop = next_pop
        fitness = next_fitness
        F = next_F
        CR = next_CR
        
        # Sort population for next iteration's p-best selection
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]
        F = F[sort_idx]
        CR = CR[sort_idx]
        
        # --- Restart Mechanism ---
        # If population diversity is extremely low, we are likely stuck in a local optimum.
        # We restart the population but keep the best solution found so far.
        fitness_range = fitness[-1] - fitness[0]
        
        if fitness_range < 1e-7:
            # Only restart if we have at least 10% of time remaining
            if (time.time() - start_time) < (max_time * 0.9):
                # Keep the best individual (at index 0 due to sort)
                # Re-initialize indices 1 to end
                n_reset = pop_size - 1
                pop[1:] = min_b + np.random.rand(n_reset, dim) * diff_b
                
                # Reset Control Parameters for new individuals
                F[1:] = 0.5
                CR[1:] = 0.9
                
                # Evaluate new individuals
                for k in range(1, pop_size):
                    if (time.time() - start_time) >= max_time:
                        return global_best_val
                    
                    val = func(pop[k])
                    fitness[k] = val
                    
                    if val < global_best_val:
                        global_best_val = val
                
                # Re-sort to maintain invariant for next loop
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx]
                fitness = fitness[sort_idx]
                F = F[sort_idx]
                CR = CR[sort_idx]

    return global_best_val
