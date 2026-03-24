#The following algorithm implements a **Vectorized Restarting Differential Evolution (DE)** with **Gaussian Parameter Dithering** and **Elitist Restarts**.
#
#**Key Improvements:**
#1.  **Gaussian Parameter Dithering**: Instead of using fixed values or uniform randomization, the mutation factor ($F$) and crossover rate ($CR$) are sampled from Gaussian distributions for *each individual* in every generation. This creates a diverse population where some individuals explore (high $F$, high $CR$) while others exploit (low $F$), adapting naturally to the function landscape.
#2.  **Elitist Restarts**: When the population stagnates or converges, the algorithm restarts. Crucially, it **injects the global best solution** found so far into the new population. This prevents the loss of progress and allows the algorithm to refine the best known solution with a fresh set of random vectors.
#3.  **Vectorized `rand/1` Strategy**: Utilizes the robust `DE/rand/1/bin` strategy implemented with efficient NumPy vectorization to maximize the number of evaluations within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Restarting Differential Evolution
    with Gaussian Parameter Dithering and Elitist Restarts.
    """
    # Initialize timing
    start_time = datetime.now()
    # Use a small safety buffer to ensure we return a result within the strict limit
    end_time = start_time + timedelta(seconds=max_time - 0.05)

    # --- Configuration ---
    # Population size: 
    # A size of 15 * dim gives a good balance between diversity and iteration speed.
    # Clamped to [30, 80] to handle various time constraints/dimensions.
    pop_size = int(np.clip(dim * 15, 30, 80))
    
    # Restart triggers
    restart_tol = 1e-6          # Restart if std dev of fitness is below this (convergence)
    max_stagnation = 40         # Restart if no improvement in local best for N gens (stagnation)

    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Global best tracking
    global_best_val = float('inf')
    global_best_pos = None

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Time check
        if datetime.now() >= end_time:
            return global_best_val

        # 1. Initialize Population
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        
        # Elitist Restart: 
        # Inject the best solution found so far into the new population (index 0).
        # This ensures monotonically non-increasing best fitness and helps 
        # local search around the best found basin.
        if global_best_pos is not None:
            pop[0] = global_best_pos.copy()

        fitness = np.full(pop_size, float('inf'))

        # Evaluate Initial Population
        # Skip index 0 if we injected the global best (we already know its fitness)
        start_idx = 0
        if global_best_pos is not None:
            fitness[0] = global_best_val
            start_idx = 1
        
        for i in range(start_idx, pop_size):
            if datetime.now() >= end_time: return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val
                global_best_pos = pop[i].copy()

        # Track local best for stagnation detection
        current_restart_best = np.min(fitness)
        stagnation_count = 0

        # 2. Evolutionary Cycle
        while True:
            if datetime.now() >= end_time: return global_best_val
            
            # --- Restart Checks ---
            # If population lost diversity or isn't improving, break to restart
            if np.std(fitness) < restart_tol or stagnation_count > max_stagnation:
                break 

            # --- Parameter Dithering (Per Individual) ---
            # Sample F and CR from Gaussian distributions for each individual.
            # This allows a mix of exploration (high F/CR) and exploitation (low F/CR)
            # within the same generation.
            # F ~ N(0.5, 0.3), clipped to [0.1, 1.0]
            # CR ~ N(0.6, 0.2), clipped to [0.0, 1.0] (slightly higher CR preference)
            F = np.random.normal(0.5, 0.3, size=(pop_size, 1))
            F = np.clip(F, 0.1, 1.0)
            
            CR = np.random.normal(0.6, 0.2, size=(pop_size, 1))
            CR = np.clip(CR, 0.0, 1.0)

            # --- Vectorized Mutation (DE/rand/1) ---
            # Efficient index generation using permutations to pick random r1, r2, r3
            idxs = np.arange(pop_size)
            r1 = np.random.permutation(idxs)
            r2 = np.random.permutation(idxs)
            r3 = np.random.permutation(idxs)
            
            # Mutant = r1 + F * (r2 - r3)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Enforce bounds (Clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # --- Vectorized Crossover (Binomial) ---
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < CR
            
            # Forced Gene: Ensure at least one dimension comes from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[idxs, j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, pop)

            # --- Selection (Sequential Evaluation) --
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                        global_best_pos = trial_pop[i].copy()
            
            # Check stagnation (Local improvement within this restart)
            new_restart_best = np.min(fitness)
            if new_restart_best < current_restart_best:
                current_restart_best = new_restart_best
                stagnation_count = 0
            else:
                stagnation_count += 1

    return global_best_val
