#The following algorithm implements **Self-Adaptive Differential Evolution (jDE) with 'Current-to-pBest' Mutation and Local Search Polishing**.
#
#### Rationale for Improvement
#The previous best algorithm (Restarting DE) achieved ~54.38. It used static parameter ranges and a greedy `current-to-best` strategy. The improvements in this version address the likely causes of stagnation:
#1.  **Self-Adaptation (jDE)**: Instead of fixed or purely random hyperparameters ($F$ and $CR$), each individual carries its own parameters. These adapt based on survival, allowing the algorithm to learn whether the landscape requires exploration (high $F$) or exploitation (low $F$).
#2.  **Current-to-pBest Mutation**: Instead of pulling towards the single best (which causes premature convergence) or a random agent (slow convergence), this strategy pulls towards a random agent selected from the top $p\%$ (e.g., top 10%) of the population. This balances greediness with diversity.
#3.  **Local Search Polishing**: Before abandoning a converged population (Restart), the algorithm executes a focused "hill-climbing" phase (Gaussian sampling with shrinking variance) around the global best. This drains the local basin of attraction to its absolute minimum, capturing decimals that DE might miss due to population granularity.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE)
    with 'current-to-pbest' mutation and Local Search Polishing.
    """
    # --- Configuration ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Population Sizing
    # Cap population to ensure reasonable generation count within time limits
    # High dim needs more pop, but too much slows down convergence.
    pop_size = int(max(20, min(10 * dim, 60))) 
    
    # jDE Constants
    tau_F = 0.1  # Probability to update F
    tau_CR = 0.1 # Probability to update CR
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Global Best Tracking
    global_best_fitness = float('inf')
    global_best_pos = None

    # --- Helper: Local Search Polish ---
    def polish_best(center, current_fit, time_budget_seconds=0.5):
        """
        Performs a localized Gaussian search around the best solution 
        to drain the local basin before restarting.
        """
        p_start = datetime.now()
        p_best_val = current_fit
        p_best_loc = center.copy()
        
        # Start with a small sigma relative to bounds, shrink over time
        sigma = diff_b * 0.05
        decay = 0.98
        
        step_count = 0
        while (datetime.now() - p_start).total_seconds() < time_budget_seconds:
            if (datetime.now() - start_time) >= time_limit:
                break
            
            # Generate candidate: Center + Gaussian Noise
            candidate = p_best_loc + np.random.normal(0, 1, dim) * sigma
            candidate = np.clip(candidate, min_b, max_b)
            
            val = func(candidate)
            
            if val < p_best_val:
                p_best_val = val
                p_best_loc = candidate
                # If we improved, reset sigma slightly to keep momentum, else decay
                sigma *= 1.05 
            else:
                sigma *= decay
            
            # Avoid sigma becoming numerically zero
            sigma = np.maximum(sigma, 1e-15)
            step_count += 1
            
        return p_best_loc, p_best_val

    # --- Main Optimization Loop (Restarts) ---
    while True:
        # Time Check
        if (datetime.now() - start_time) >= time_limit:
            return global_best_fitness

        # 1. Initialization
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Initialize jDE adaptive parameters for each individual
        # F in [0.1, 1.0], CR in [0.0, 1.0]
        F_arr = 0.1 + 0.9 * np.random.rand(pop_size)
        CR_arr = np.random.rand(pop_size)

        # Evaluate initial population
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return global_best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val
                global_best_pos = population[i].copy()

        # 2. Evolutionary Loop
        stall_counter = 0
        prev_min_fit = np.min(fitness)

        while True:
            # Check Time
            if (datetime.now() - start_time) >= time_limit:
                return global_best_fitness

            # Sort population for current-to-pbest selection
            sorted_indices = np.argsort(fitness)
            
            # Determine 'p' for p-best (top 5% to 20%, adaptive or fixed)
            # Using top 10% (minimum 2 individuals)
            p_limit = max(2, int(pop_size * 0.10))
            top_p_indices = sorted_indices[:p_limit]

            # Create arrays for new generation
            new_pop = np.zeros_like(population)
            new_fit = np.copy(fitness) # Default to keeping parent
            
            # Arrays to store successful parameters
            # In standard jDE, we update trial params in place. 
            # Here we generate trial params first.
            
            # Generate Updated Parameters (Vectorized)
            rand_F = np.random.rand(pop_size)
            rand_CR = np.random.rand(pop_size)
            
            # Logic: mask where update happens
            mask_F = rand_F < tau_F
            mask_CR = rand_CR < tau_CR
            
            trial_F = np.where(mask_F, 0.1 + 0.9 * np.random.rand(pop_size), F_arr)
            trial_CR = np.where(mask_CR, np.random.rand(pop_size), CR_arr)

            # Generate Mutations: current-to-pbest/1
            # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
            
            # Select r1 != r2 != i
            # Since vectorizing unique exclusion per row is hard in pure numpy without heavy overhead,
            # we accept a small collision risk or use random rolls which is fast.
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Select pbest indices for each individual
            pbest_indices = np.random.choice(top_p_indices, pop_size)
            
            x_pbest = population[pbest_indices]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Add dimensions for broadcasting F to (pop, dim)
            F_matrix = trial_F[:, np.newaxis]
            
            mutant = population + F_matrix * (x_pbest - population) + F_matrix * (x_r1 - x_r2)
            mutant = np.clip(mutant, min_b, max_b)
            
            # Crossover (Binomial)
            # Mask: True if crossover happens (rand < CR) OR is logical index j_rand
            rand_matrix = np.random.rand(pop_size, dim)
            CR_matrix = trial_CR[:, np.newaxis]
            cross_mask = rand_matrix < CR_matrix
            
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, population)
            
            # Selection & Evaluation
            # We must iterate for func evaluation
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return global_best_fitness
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitness[i]:
                    # Trial is better or equal: Replace parent
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update self-adaptive parameters
                    F_arr[i] = trial_F[i]
                    CR_arr[i] = trial_CR[i]
                    
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial
                        global_best_pos = trial_pop[i].copy()
                        stall_counter = 0 # Reset stall on global improvement

            # Convergence Check for Restart
            current_min = np.min(fitness)
            std_dev = np.std(fitness)
            
            if current_min < prev_min_fit:
                prev_min_fit = current_min
                stall_counter = 0
            else:
                stall_counter += 1
            
            # Trigger Restart conditions:
            # 1. Low variance (collapsed population)
            # 2. Significant stagnation (no local improvement for N gens)
            if std_dev < 1e-6 or stall_counter > 40:
                # Before restart, polish the global best
                # Allocate a small slice of remaining time for polishing (e.g., 5% or 1 sec)
                remaining = (time_limit - (datetime.now() - start_time)).total_seconds()
                polish_time = min(1.0, remaining * 0.1)
                
                if polish_time > 0.05 and global_best_pos is not None:
                    p_loc, p_val = polish_best(global_best_pos, global_best_fitness, polish_time)
                    if p_val < global_best_fitness:
                        global_best_fitness = p_val
                        global_best_pos = p_loc
                
                break # Break inner loop -> triggers outer loop restart

    return global_best_fitness
