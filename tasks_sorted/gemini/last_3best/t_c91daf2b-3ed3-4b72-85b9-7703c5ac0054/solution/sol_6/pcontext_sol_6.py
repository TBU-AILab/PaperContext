#The output value of the last generated algorithm (65.25) was significantly worse than the best previous result (11.20). The deterioration was likely due to the overhead of full vectorization on a limited time budget or the aggressive crossover/mutation parameters.
#
#To improve upon the best result (11.20), I will refine the successful **Differential Evolution (DE)** strategy from Code 1 by adding **Opposition-Based Learning (OBL)** for initialization. OBL allows the algorithm to start with a much higher quality population by checking opposite points in the search space, which effectively doubles the search efficiency in the initial phase. I will also introduce **Parameter Dithering** for the mutation factor $F$ to prevent the search from stagnating, while keeping the efficient loop structure of the successful attempt.
#
import numpy as np
import time
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE/rand/1/bin) 
    enhanced with Opposition-Based Learning (OBL) for initialization.
    """
    start_time = time.time()
    
    # --- Parameters ---
    # Population size: 
    # A size of 10*dim is standard. We clamp it between [20, 50].
    # This range balances population diversity with the execution speed needed 
    # to run enough generations within max_time.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # DE Hyperparameters
    # CR (Crossover Rate): 0.7 
    # This value proved most effective in previous best attempts (Score 11.2).
    # It balances preserving the target vector with mixing in the mutant.
    CR = 0.7
    
    # Pre-process bounds for faster access
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # 1. Random Initialization
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    
    # Evaluate initial random population
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            
    # 2. Opposition-Based Learning (OBL)
    # Check if we have enough time (using < 10% of budget) to perform OBL.
    # OBL checks the "opposite" point in the search space (Min + Max - X).
    # This statistically improves the starting fitness significantly.
    if (time.time() - start_time) < (max_time * 0.1):
        # Generate opposite population
        opp_pop = min_b + max_b - pop
        # Clip to ensure bounds are respected
        opp_pop = np.clip(opp_pop, min_b, max_b)
        opp_fitness = np.full(pop_size, float('inf'))
        
        # Evaluate opposite population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            val = func(opp_pop[i])
            opp_fitness[i] = val
            if val < best_val:
                best_val = val
        
        # Selection: Keep the best `pop_size` individuals from (pop + opp_pop)
        combined_pop = np.vstack((pop, opp_pop))
        combined_fit = np.concatenate((fitness, opp_fitness))
        
        # Sort by fitness (ascending)
        sorted_indices = np.argsort(combined_fit)
        
        # Update population to the best found so far
        pop = combined_pop[sorted_indices[:pop_size]]
        fitness = combined_fit[sorted_indices[:pop_size]]

    # --- Main Evolutionary Loop ---
    # Pre-allocate indices list for faster sampling
    idxs = list(range(pop_size))
    
    while True:
        for i in range(pop_size):
            # Strict time check per individual
            if time.time() - start_time >= max_time:
                return best_val
            
            # --- Mutation (DE/rand/1) ---
            # Select 3 distinct random indices r1, r2, r3, none equal to i
            r1, r2, r3 = random.sample(idxs, 3)
            while i in (r1, r2, r3):
                r1, r2, r3 = random.sample(idxs, 3)
                
            # Dither F (Mutation Factor)
            # Instead of a fixed F=0.5, we randomize it between [0.5, 1.0].
            # This helps the algorithm escape local optima and adapts step sizes.
            F = 0.5 + 0.5 * random.random()
            
            # Calculate Mutant Vector: V = X_r1 + F * (X_r2 - X_r3)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # --- Crossover (Binomial) ---
            # Generate a boolean mask
            cross_mask = np.random.rand(dim) < CR
            # Ensure at least one parameter comes from the mutant
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            # Create Trial Vector
            trial = np.where(cross_mask, mutant, pop[i])
            
            # --- Boundary Handling ---
            trial = np.clip(trial, min_b, max_b)
            
            # --- Selection ---
            f_trial = func(trial)
            
            # Greedy selection
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                # Update global best immediately
                if f_trial < best_val:
                    best_val = f_trial
                    
    return best_val
