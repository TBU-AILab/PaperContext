#The following algorithm implements a **Vectorized Restarting Differential Evolution (DE)** using the **current-to-best/1/bin** strategy and **Latin Hypercube Sampling (LHS)** initialization.
#
#**Key Improvements:**
#1.  **Strategy Change (current-to-best/1)**: Previous attempts used `rand/1`, which explores well but converges slowly. The `current-to-best/1` strategy ($\vec{v} = \vec{x} + F(\vec{x}_{best} - \vec{x}) + F(\vec{x}_{r1} - \vec{x}_{r2})$) actively pulls the population towards the local minimum while maintaining diversity via the difference vector. This significantly speeds up convergence, which is critical for the "limited time" constraint.
#2.  **Latin Hypercube Sampling (LHS)**: Instead of purely random initialization, LHS stratifies the samples to ensure the initial population covers the bounds more evenly. This increases the probability of finding a good basin of attraction early in each restart.
#3.  **Independent Restarts**: Unlike previous attempts that injected the global best into the new population (which often caused premature convergence to the same local optimum), this algorithm uses fresh, independent restarts. This prevents getting trapped in one basin and allows the fast-converging DE to explore multiple areas of the landscape effectively.
#4.  **Optimized Parameters**: Uses dithered parameters ($F \in [0.5, 0.9]$, $CR \in [0.8, 1.0]$) optimized for the greedy nature of the `current-to-best` strategy.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Restarting Differential Evolution
    with 'current-to-best/1' strategy and Latin Hypercube Sampling.
    """
    # Initialize timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)

    # --- Configuration ---
    # Population size: 
    # Scaled with dimension but capped at 50 to ensure high generation count within time limit.
    # The 'current-to-best' strategy works well with smaller populations.
    pop_size = int(np.clip(dim * 10, 20, 50))
    
    # Restart triggers
    restart_tol = 1e-6          # Restart if population standard deviation is too low (convergence)
    max_stagnation = 30         # Restart if local best doesn't improve for N generations

    # Pre-process bounds for efficient vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    # No need for diff_b here, used inside loop

    # Global best tracking
    global_best_val = float('inf')

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Strict time check before starting a new heavyweight restart
        if datetime.now() >= end_time:
            return global_best_val

        # 1. Initialization (Latin Hypercube Sampling)
        # LHS ensures a more uniform coverage of the search space than random sampling
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            # Create stratified bins for dimension d
            edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
            lower_edges = edges[:-1]
            bin_width = edges[1] - edges[0]
            # Sample uniformly within each bin and shuffle assignment
            samples = lower_edges + np.random.rand(pop_size) * bin_width
            pop[:, d] = np.random.permutation(samples)

        # Evaluate Initial Population
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if datetime.now() >= end_time: return global_best_val
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < global_best_val:
                global_best_val = val

        # Local tracking for stagnation detection within this restart
        current_restart_best = np.min(fitness)
        stagnation_count = 0

        # 2. Evolutionary Cycle
        while True:
            # Check time
            if datetime.now() >= end_time: return global_best_val

            # --- Convergence & Stagnation Checks ---
            # If population has converged or isn't improving, break to trigger a restart
            if np.std(fitness) < restart_tol or stagnation_count > max_stagnation:
                break

            # --- Mutation Strategy: DE/current-to-best/1/bin ---
            # V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This strategy balances greediness (move to best) and exploration.
            
            # Dynamic Parameters (Dithering)
            # F ~ U(0.5, 0.9): High enough to jump out of small bumps
            # CR ~ U(0.8, 1.0): High crossover preserves the 'current-to-best' trajectory
            F = 0.5 + 0.4 * np.random.rand()
            CR = 0.8 + 0.2 * np.random.rand()

            # Identify best in current population for the strategy
            best_idx = np.argmin(fitness)
            X_best = pop[best_idx] # Broadcasts to (pop_size, dim) implicitly

            # Vectorized Indices for r1, r2
            idxs = np.arange(pop_size)
            r1 = np.random.permutation(idxs)
            r2 = np.random.permutation(idxs)

            # Calculate Mutant Vector
            # pop + F*(X_best - pop) pulls individuals towards the best found so far
            # + F*(pop[r1] - pop[r2]) adds random variation based on population spread
            mutant = pop + F * (X_best - pop) + F * (pop[r1] - pop[r2])

            # Bound Constraints (Clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # --- Crossover (Binomial) ---
            cross_mask = np.random.rand(pop_size, dim) < CR
            # Forced Gene: Ensure at least one dimension is taken from the mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[idxs, j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, pop)

            # --- Selection (Sequential Evaluation) ---
            improved_in_generation = False
            for i in range(pop_size):
                if datetime.now() >= end_time: return global_best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy Selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    if f_trial < global_best_val:
                        global_best_val = f_trial
                    
                    if f_trial < current_restart_best:
                        current_restart_best = f_trial
                        improved_in_generation = True
            
            # Update stagnation counter
            if improved_in_generation:
                stagnation_count = 0
            else:
                stagnation_count += 1
                
    return global_best_val
