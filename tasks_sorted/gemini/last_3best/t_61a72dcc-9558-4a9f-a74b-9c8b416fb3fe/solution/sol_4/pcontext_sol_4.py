#The previous results indicate that Algorithm 1 (Standard Differential Evolution with `current-to-best/1/bin` and restarts) performed significantly better (12.61) than Algorithm 2 (25.13) and Algorithm 3 (36.51). Algorithm 3 likely failed because the polishing step (Coordinate Descent) was too computationally expensive and blocked the algorithm from effectively restarting and exploring other basins of attraction within the time limit.
#
#To improve upon Algorithm 1, we will implement a **Differential Evolution (DE) with Latin Hypercube Sampling (LHS) Initialization and Restarts**.
#
#**Key Improvements:**
#1.  **Latin Hypercube Sampling (LHS) Initialization**: Instead of uniform random initialization, LHS is used to generate the initial population. LHS ensures that the sample points are more evenly distributed across the search space (stratified), reducing the likelihood of bunching and providing a better starting coverage.
#2.  **Optimized "current-to-best" Strategy**: We retain the `current-to-best/1/bin` strategy as it showed the best convergence properties. We refine the mutation factor $F$ to be dithered in the range $[0.3, 0.9]$ to allow for both exploration and fine-tuning.
#3.  **Efficient Restart Mechanism**: The algorithm monitors population variance and stagnation. If the population converges (low variance) or fails to improve the local best for a set number of generations, it triggers a hard restart to explore new areas of the search space.
#4.  **Vectorized Operations**: The implementation relies on NumPy vectorization for mutation, crossover, and bound handling to maximize the number of function evaluations within `max_time`.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE) with 
    Latin Hypercube Sampling (LHS) initialization and a Restart mechanism.
    
    Strategy: DE/current-to-best/1/bin
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds_arr = np.array(bounds)
    lower_bound = bounds_arr[:, 0]
    upper_bound = bounds_arr[:, 1]
    
    # --- Configuration ---
    # Population size: Heuristic based on dimension, clamped for performance.
    # A size of 15*dim is generally robust, capped at 100 to ensure speed.
    pop_size = int(np.clip(15 * dim, 20, 100))
    
    best_fitness = float('inf')

    # Helper function to check remaining time
    def check_time():
        return (time.time() - start_time) >= max_time

    # --- Main Loop (Restarts) ---
    while True:
        # Check time before starting a new optimization run
        if check_time():
            return best_fitness
            
        # --- Initialization: Latin Hypercube Sampling (LHS) ---
        # LHS ensures better coverage of the search space than random uniform sampling.
        # We divide each dimension into 'pop_size' intervals and pick one point per interval.
        population = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(lower_bound[d], upper_bound[d], pop_size + 1)
            # Sample uniformly within each segment
            points = np.random.uniform(edges[:-1], edges[1:])
            # Shuffle points to uncorrelated dimensions
            np.random.shuffle(points)
            population[:, d] = points
            
        fitnesses = np.full(pop_size, float('inf'))
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if check_time(): return best_fitness
            
            val = func(population[i])
            fitnesses[i] = val
            
            if val < best_fitness:
                best_fitness = val
                
        # Track the best individual in the current population (Local Best)
        current_best_idx = np.argmin(fitnesses)
        current_best_val = fitnesses[current_best_idx]
        stagnation_counter = 0
        
        # --- Evolution Loop ---
        while True:
            if check_time(): return best_fitness
            
            # --- Parameters ---
            # F (Mutation Factor): Dithered between 0.3 and 0.9.
            # Lower values allow fine-tuning, higher values aid exploration.
            F = 0.3 + 0.6 * np.random.rand()
            # CR (Crossover Rate): 0.9 is generally effective for inseparable functions.
            CR = 0.9
            
            # --- Mutation: current-to-best/1 ---
            # Equation: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This strategy directs the search towards the best solution while maintaining diversity.
            
            # Generate random indices
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            x_best = population[current_best_idx]
            
            # Vectorized calculation of mutant vectors
            # (X_best - X) pulls individuals towards the best found so far
            diff_best = x_best - population
            # (X_r1 - X_r2) provides random perturbation
            diff_rand = population[r1] - population[r2]
            
            mutant = population + F * diff_best + F * diff_rand
            
            # --- Crossover (Binomial) ---
            # Create a mask for crossover
            mask = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one dimension is taken from the mutant (DE requirement)
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask, mutant, population)
            
            # --- Bound Handling ---
            # Clip values to stay within bounds
            trial_pop = np.clip(trial_pop, lower_bound, upper_bound)
            
            # --- Selection ---
            # Evaluate trial vectors and replace target if better (Greedy selection)
            for i in range(pop_size):
                if check_time(): return best_fitness
                
                f_trial = func(trial_pop[i])
                
                if f_trial <= fitnesses[i]:
                    fitnesses[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update global best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                    
                    # Update local best tracker
                    if f_trial < fitnesses[current_best_idx]:
                        current_best_idx = i
            
            # --- Stagnation & Convergence Check ---
            new_best_val = fitnesses[current_best_idx]
            
            # Check if the generation improved the best solution significantly
            if current_best_val - new_best_val < 1e-9:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                current_best_val = new_best_val
            
            # Restart Triggers:
            # 1. Population variance is very low (Converged)
            # 2. Best solution hasn't improved for many generations (Stuck)
            if np.std(fitnesses) < 1e-7 or stagnation_counter > 30:
                break
                
    return best_fitness
