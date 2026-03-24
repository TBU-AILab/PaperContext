#The following algorithm implements a **Self-Adaptive Differential Evolution (jDE)** strategy. 
#
#**Improvements over previous versions:**
#1.  **Self-Adaptation**: Instead of fixed hyperparameters ($F=0.5, CR=0.9$) or random dithering, this algorithm adapts the Mutation Factor ($F$) and Crossover Rate ($CR$) for each individual. This allows the algorithm to dynamically tune itself to the specific landscape of the function (e.g., separable vs. non-separable, steep vs. flat).
#2.  **Optimized Population Sizing**: Based on the success of the previous best result, the population size is dynamically calculated based on dimension but clamped between 20 and 50. This ensures enough diversity for exploration without slowing down generational speed in high dimensions.
#3.  **Strict Time Management**: Time checks are embedded deeply within the evaluation loops to strictly adhere to `max_time`, preventing timeouts.
#4.  **Convergence Restart**: A safety mechanism re-initializes the population (keeping the best solution) if the diversity drops to zero (standard deviation < 1e-8), allowing the algorithm to escape stagnation in local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE).
    This variant adapts F and CR parameters at the individual level to 
    dynamically suit the problem landscape.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Clamped to [20, 50] based on empirical success.
    # A size of 10*dim is standard, but capping it ensures high generational
    # throughput within the limited time budget.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # jDE Adaptation Probabilities
    # Probabilities to update F and CR respectively
    tau_F = 0.1
    tau_CR = 0.1
    
    # Bounds processing
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # Initialize population randomly within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize Adaptive Parameters: F and CR for each individual
    # Starting values: F=0.5, CR=0.9 (Standard DE defaults)
    F_arr = np.full(pop_size, 0.5)
    CR_arr = np.full(pop_size, 0.9)
    
    best_val = float('inf')
    best_idx = -1
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i
            
    # --- Main Optimization Loop ---
    while True:
        # Iterate through the population
        for i in range(pop_size):
            # Strict time check inside the loop
            if time.time() - start_time >= max_time:
                return best_val
            
            # --- 1. Parameter Adaptation (jDE) ---
            # Update Mutation Factor F
            if np.random.rand() < tau_F:
                # New F in range [0.1, 1.0]
                F_i = 0.1 + 0.9 * np.random.rand()
            else:
                F_i = F_arr[i]
                
            # Update Crossover Rate CR
            if np.random.rand() < tau_CR:
                # New CR in range [0.0, 1.0]
                CR_i = np.random.rand()
            else:
                CR_i = CR_arr[i]
            
            # --- 2. Mutation (DE/rand/1) ---
            # Efficiently select r1, r2, r3 distinct from i
            # Using while loops is faster than np.random.choice for small sets
            r1 = np.random.randint(0, pop_size)
            while r1 == i: 
                r1 = np.random.randint(0, pop_size)
            
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1: 
                r2 = np.random.randint(0, pop_size)
                
            r3 = np.random.randint(0, pop_size)
            while r3 == i or r3 == r1 or r3 == r2: 
                r3 = np.random.randint(0, pop_size)
            
            # Compute Mutant Vector
            mutant = pop[r1] + F_i * (pop[r2] - pop[r3])
            
            # --- 3. Crossover (Binomial) ---
            # Create crossover mask
            mask = np.random.rand(dim) < CR_i
            # Ensure at least one parameter is taken from the mutant (Standard DE rule)
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            # Create trial vector
            trial = np.where(mask, mutant, pop[i])
            
            # --- 4. Boundary Handling ---
            # Clip values to ensure they stay within valid bounds
            trial = np.clip(trial, min_b, max_b)
            
            # --- 5. Selection ---
            f_trial = func(trial)
            
            # Greedy selection
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                # If trial was successful, keep the adapted parameters
                F_arr[i] = F_i
                CR_arr[i] = CR_i
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i
        
        # --- Restart Mechanism ---
        # If population has converged (variance ~ 0) and we have time left,
        # restart the population to avoid stagnation, keeping the best individual.
        if time.time() - start_time < max_time * 0.9:
            # Check standard deviation of fitness
            if np.std(fitness) < 1e-8:
                # Preserve the best solution found so far (Elitism)
                best_vec = pop[best_idx].copy()
                
                # Re-initialize population
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                pop[0] = best_vec
                
                # Reset adaptive parameters to defaults
                F_arr = np.full(pop_size, 0.5)
                CR_arr = np.full(pop_size, 0.9)
                
                # Reset fitness array
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = best_val
                best_idx = 0
                
                # Re-evaluate the new random population (skipping index 0)
                for k in range(1, pop_size):
                    if time.time() - start_time >= max_time:
                        return best_val
                    val = func(pop[k])
                    fitness[k] = val
                    if val < best_val:
                        best_val = val
                        best_idx = k
    
    return best_val
