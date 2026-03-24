#The following algorithm implements a simplified **L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution)** strategy.
#
#### Algorithm Description
#This approach significantly improves upon standard Differential Evolution (DE) by integrating three advanced mechanisms to balance exploration and exploitation dynamically over the limited time:
#
#1.  **Linear Population Size Reduction (LPSR)**: The algorithm starts with a large population to explore the search space broadly. As time progresses, the population size is linearly reduced (discarding worst-performing individuals). This focuses the computational budget on refining the best solutions (exploitation) as the deadline approaches.
#2.  **Current-to-pbest Mutation**: Instead of exploring blindly (random) or too greedily (best), the algorithm steers the population towards a random individual selected from the top $p\%$ of best solutions. The $p$ value also decreases over time, sharpening the convergence.
#3.  **Adaptive Parameters**: The mutation factor ($F$) and crossover rate ($CR$) are randomized for each individual (dithering) around values proven to be robust ($F \approx 0.5, CR \approx 0.9$). This prevents the population from stagnating due to fixed search step sizes.
#
#### Python Implementation
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using an L-SHADE inspired Differential Evolution 
    with Linear Population Size Reduction and Current-to-pbest mutation.
    """
    # --- Timing Setup ---
    start_time = time.time()
    # Set a safety buffer (0.05s) to ensure we return before hard timeout
    end_time = start_time + max_time - 0.05

    # --- Configuration ---
    # 1. Population Sizing (LPSR)
    # Start large (approx 18 * dim) for exploration, capped at 400 for speed
    pop_size_init = min(400, max(40, 18 * dim))
    # End small for fast convergence
    pop_size_min = 4
    
    # Pre-process bounds for vectorization
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # --- Initialization ---
    current_pop_size = pop_size_init
    pop = np.random.uniform(lb, ub, (current_pop_size, dim))
    fitness = np.full(current_pop_size, float('inf'))
    
    best_val = float('inf')

    # Evaluate Initial Population
    # We check time strictly inside the loop
    for i in range(current_pop_size):
        if time.time() >= end_time:
            return best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val

    # Sort population by fitness (best first) to enable p-best selection
    sort_idx = np.argsort(fitness)
    pop = pop[sort_idx]
    fitness = fitness[sort_idx]

    # --- Main Optimization Loop ---
    while True:
        # Check time at start of generation
        t_now = time.time()
        if t_now >= end_time:
            return best_val

        # 1. Linear Population Size Reduction
        # Calculate progress ratio (0.0 to 1.0)
        elapsed = t_now - start_time
        # Target 95% of max_time for full reduction to leave final iterations for convergence
        progress = min(1.0, elapsed / (max_time * 0.95))
        
        # Calculate target population size
        target_size = int(pop_size_init + (pop_size_min - pop_size_init) * progress)
        target_size = max(pop_size_min, target_size)

        # Shrink population if needed (discard worst individuals at the end of sorted array)
        if current_pop_size > target_size:
            current_pop_size = target_size
            pop = pop[:current_pop_size]
            fitness = fitness[:current_pop_size]

        # 2. Adaptive Parameters (Dithering)
        # Generate F (Mutation) centered at 0.5
        F = np.random.normal(0.5, 0.3, (current_pop_size, 1))
        F = np.clip(F, 0.1, 1.0)
        
        # Generate CR (Crossover) centered at 0.9
        CR = np.random.normal(0.9, 0.1, (current_pop_size, 1))
        CR = np.clip(CR, 0.0, 1.0)

        # 3. Mutation: DE/current-to-pbest/1
        # Dynamic p: starts at 0.2 (explore), ends at 0.05 (exploit)
        p = 0.2 - 0.15 * progress
        p = max(0.05, p)
        
        # Select target vectors from top p% (p-best)
        top_p_count = max(2, int(current_pop_size * p))
        pbest_indices = np.random.randint(0, top_p_count, current_pop_size)
        x_pbest = pop[pbest_indices]

        # Select random vectors r1, r2
        r1 = np.random.randint(0, current_pop_size, current_pop_size)
        r2 = np.random.randint(0, current_pop_size, current_pop_size)
        
        # Mutation equation: V = X + F(X_pbest - X) + F(X_r1 - X_r2)
        # (Vectorized operation for entire population)
        diff_pbest = x_pbest - pop
        diff_rand = pop[r1] - pop[r2]
        mutant = pop + F * diff_pbest + F * diff_rand

        # 4. Crossover (Binomial)
        mask = np.random.rand(current_pop_size, dim) < CR
        # Ensure at least one parameter is changed for every individual
        j_rand = np.random.randint(0, dim, current_pop_size)
        mask[np.arange(current_pop_size), j_rand] = True
        
        trial_pop = np.where(mask, mutant, pop)

        # 5. Bound Constraints (Clipping)
        trial_pop = np.clip(trial_pop, lb, ub)

        # 6. Selection
        # Evaluate trial vectors
        for i in range(current_pop_size):
            if time.time() >= end_time:
                return best_val
            
            f_trial = func(trial_pop[i])

            # Greedy selection
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                if f_trial < best_val:
                    best_val = f_trial
        
        # Re-sort population for the next generation
        # This keeps the "p-best" at the top of the array and the "worst" to be removed at the bottom
        sort_idx = np.argsort(fitness)
        pop = pop[sort_idx]
        fitness = fitness[sort_idx]

    return best_val
