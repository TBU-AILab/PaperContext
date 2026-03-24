#The following algorithm implements a **Self-Adaptive Differential Evolution (jDE)** strategy enhanced with a **Restart Mechanism**. 
#
#Key improvements over the standard Differential Evolution:
#1.  **Self-Adaptation (jDE)**: The mutation factor (`F`) and crossover probability (`CR`) are encoded into each individual and evolve alongside the solution. This eliminates the need for manual parameter tuning and allows the algorithm to adapt to the fitness landscape during different phases of optimization.
#2.  **Greedy Mutation Strategy**: It uses the `DE/current-to-best/1` mutation strategy. This guides the search more aggressively towards the best solution found so far, improving convergence speed compared to the standard `DE/rand/1`.
#3.  **Restart Mechanism**: If the population converges (low variance in fitness) or stagnates (no improvement for a set number of generations), the algorithm triggers a restart. It preserves the global best solution but re-initializes the rest of the population to explore new areas of the search space, helping to escape local optima.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE) 
    with a Restart Mechanism to escape local optima.
    """
    start_time = datetime.now()
    # Subtract a small buffer to ensure safe return before timeout
    end_time = start_time + timedelta(seconds=max_time) - timedelta(milliseconds=50)

    # 1. Hyperparameters
    # Population size: Adapted to dimension but capped to ensure speed
    # A size of 10*dim is standard, but we cap at 60 to ensure enough generations run
    pop_size = int(np.clip(10 * dim, 20, 60))
    
    # Restart triggers
    stall_limit = 20  # Max generations without improvement before restart
    
    # 2. Setup Bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # 3. Initialization
    # Population matrix: (pop_size, dim)
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # jDE Self-Adaptive Parameters initialized randomly
    # F (Mutation Factor) in [0.1, 0.9], CR (Crossover Rate) in [0.0, 1.0]
    F = np.random.uniform(0.1, 0.9, pop_size)
    CR = np.random.uniform(0.0, 1.0, pop_size)

    # Global best tracking
    best_fitness = float('inf')
    best_sol = None

    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() >= end_time:
            # Fallback if time expires during initialization
            return best_fitness if best_fitness != float('inf') else func(pop[i])

        val = func(pop[i])
        fitness[i] = val

        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()

    stall_counter = 0

    # 4. Main Optimization Loop
    while True:
        # Check time at the start of each generation
        if datetime.now() >= end_time:
            return best_fitness

        # --- Self-Adaptation (jDE) ---
        # Update F and CR with probability tau=0.1
        mask_f = np.random.rand(pop_size) < 0.1
        mask_cr = np.random.rand(pop_size) < 0.1
        
        # F takes new value in [0.1, 1.0]
        F[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
        # CR takes new value in [0.0, 1.0]
        CR[mask_cr] = np.random.rand(np.sum(mask_cr))

        # --- Mutation: DE/current-to-best/1 ---
        # V_i = X_i + F_i * (X_best - X_i) + F_i * (X_r1 - X_r2)
        
        # Generate random indices r1, r2
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Ensure distinctness (r1 != r2 != i) via simple correction
        for i in range(pop_size):
            while r1[i] == i:
                r1[i] = np.random.randint(0, pop_size)
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, pop_size)
        
        # Prepare broadcasting for vectorization
        current_best = best_sol 
        F_col = F[:, None]
        
        # Vectorized mutation calculation
        mutant = pop + F_col * (current_best - pop) + F_col * (pop[r1] - pop[r2])
        
        # Boundary constraints (clipping)
        mutant = np.clip(mutant, min_b, max_b)

        # --- Crossover ---
        # Binomial crossover
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < CR[:, None]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)

        # --- Selection ---
        gen_improved = False
        
        for i in range(pop_size):
            # Check time frequently (inside loop) in case func is slow
            if datetime.now() >= end_time:
                return best_fitness

            f_trial = func(trial_pop[i])

            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial_pop[i]

                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
                    gen_improved = True

        # --- Restart Mechanism ---
        if gen_improved:
            stall_counter = 0
        else:
            stall_counter += 1
            
        # Calculate population diversity
        fit_std = np.std(fitness)
        
        # Restart if stalled or converged (variance is too low)
        if stall_counter >= stall_limit or fit_std < 1e-6:
            # Re-initialize population to escape local optimum
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            
            # Elitism: Keep the single best solution found so far at index 0
            pop[0] = best_sol.copy()
            
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fitness
            
            # Reset adaptive parameters for exploration
            F = np.random.uniform(0.1, 0.9, pop_size)
            CR = np.random.uniform(0.0, 1.0, pop_size)
            
            # Re-evaluate the new population (skip index 0 as it's the best)
            for i in range(1, pop_size):
                if datetime.now() >= end_time:
                    return best_fitness
                
                val = func(pop[i])
                fitness[i] = val
                
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            
            stall_counter = 0
            
    return best_fitness
