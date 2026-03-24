#The following algorithm implements a **Restarting Differential Evolution (DE) with a 'Current-to-Best' mutation strategy**.
#
#### Rational for Improvement
#The previous algorithm used a standard `DE/rand/1` strategy with no restarts. The result (103.99) suggests it likely converged prematurely to a local optimum or stagnated. To improve this:
#1.  **Strategy Shift (`current-to-best`)**: Instead of searching randomly, the mutation vectors are biased towards the best solution found so far in the current population. This accelerates convergence significantly.
#2.  **Restarts**: The algorithm monitors the population's diversity (standard deviation) and improvement rate. If the search stagnates or converges to a specific value, it automatically triggers a "hard restart" (re-initializing the population) while preserving the global best. This allows the algorithm to escape local optima (like the 103.99 basin) and explore other regions of the search space within the remaining time.
#
#### Algorithm Code
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restarting Differential Evolution with 
    'current-to-best' mutation strategy for accelerated convergence.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Hyperparameters ---
    # Compact population size to allow for fast iterations and multiple restarts
    pop_size = max(10, int(dim * 1.5))
    
    # Global best tracking
    global_best_fitness = float('inf')
    
    # --- Outer Loop: Global Restarts ---
    while True:
        # Check time before starting a new restart
        if (datetime.now() - start_time) >= time_limit:
            return global_best_fitness

        # 1. Initialization Phase
        # Random initialization within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return global_best_fitness
            
            val = func(population[i])
            fitness[i] = val
            
            if val < global_best_fitness:
                global_best_fitness = val

        # --- Inner Loop: Evolution Phase ---
        # We run this until convergence (low variance) or stagnation
        stall_counter = 0
        local_best_fitness = np.min(fitness)
        
        while True:
            # Strict time check
            if (datetime.now() - start_time) >= time_limit:
                return global_best_fitness
            
            # 2. Dynamic Parameters (Dither)
            # F (Scaling Factor): Randomized between 0.5 and 1.0 to help maintain diversity
            F = 0.5 + 0.5 * np.random.rand() 
            # CR (Crossover Rate): High probability to inherit structure from mutant
            CR = 0.9
            
            # 3. Mutation: DE/current-to-best/1
            # Formula: V = X_current + F*(X_best - X_current) + F*(X_r1 - X_r2)
            # This balances exploration (difference vector) with exploitation (pull towards best)
            
            # Identify best in current population
            idx_best_local = np.argmin(fitness)
            x_best_local = population[idx_best_local]
            
            # Select random indices r1, r2 for the whole population
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Vectorized mutation calculation
            # Note: X_current is 'population' itself in this vectorized operation
            mutant = population + F * (x_best_local - population) + F * (population[r1] - population[r2])
            
            # Bound Constraint Handling (Clip)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < CR
            
            # Guaranteed crossover: ensure at least one dimension comes from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            # Create trial population
            trial_pop = np.where(cross_mask, mutant, population)
            
            # 5. Selection (Greedy)
            # We must iterate to evaluate func, as func takes 1D array
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return global_best_fitness
                
                f_trial = func(trial_pop[i])
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    if f_trial < global_best_fitness:
                        global_best_fitness = f_trial
            
            # 6. Convergence / Restart Detection
            current_local_best = np.min(fitness)
            
            # Check for improvement in this batch
            if current_local_best < local_best_fitness:
                local_best_fitness = current_local_best
                stall_counter = 0
            else:
                stall_counter += 1
            
            # Calculate population diversity (Standard Deviation)
            pop_std = np.std(fitness)
            
            # Restart Trigger:
            # A. Population has converged to a single point (low std dev)
            # B. No improvement for significant number of generations (stagnation)
            if pop_std < 1e-6 or stall_counter > 30:
                break # Break inner loop to trigger restart in outer loop

    return global_best_fitness
