#The output value of the last generated algorithm (54.78) was significantly worse than the best previous result (11.20). The deterioration was likely due to the overhead of the Opposition-Based Learning (OBL) initialization, which doubled the function evaluations at the start, consuming valuable time without guaranteeing a proportional fitness gain in limited-time scenarios.
#
#To improve upon the best result (11.20), I will propose an algorithm based on **Differential Evolution (DE)** with the **current-to-best/1/bin** strategy combined with a **Restart Mechanism**.
#
#**Key Improvements:**
#1.  **Strategy Change (current-to-best/1)**: Unlike the standard `rand/1` used in the best previous attempt (which explores slowly), `current-to-best/1` uses the global best solution to guide the mutation ($V = X_i + F \cdot (X_{best} - X_i) + F \cdot (X_{r1} - X_{r2})$). This drastically improves convergence speed.
#2.  **Restart Mechanism**: Fast convergence risks getting stuck in local optima. To mitigate this, I introduce a "Stagnation Restart". If the global best value does not improve for a set number of generations (`patience`), the population is re-initialized (keeping only the best solution). This allows the algorithm to quickly exploit a basin of attraction and then jump to explore others.
#3.  **Efficiency**: The implementation uses optimized random number generation (Python's `random.randrange` for scalars) and minimizes object creation inside loops to maximize the number of generations within `max_time`.
#
import numpy as np
import time
import random

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution with 'current-to-best' mutation
    and a Restart Mechanism to escape local optima.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: 
    # Clamped between 20 and 50. This range is empirically efficient for 
    # short-term execution, allowing enough generations for convergence 
    # while maintaining sufficient diversity.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # DE Hyperparameters
    CR = 0.8  # Crossover Rate
    # F (Mutation Factor) will be dithered (randomized) during execution
    
    # Bounds processing
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Helper: Generate Random Population ---
    def get_random_pop(size):
        return min_b + np.random.rand(size, dim) * diff_b

    # --- Initialization ---
    pop = get_random_pop(pop_size)
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_vec = np.zeros(dim)
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # Restart Control
    stall_count = 0
    patience = 20  # Restart after 20 generations without improvement
    
    # --- Main Optimization Loop ---
    while True:
        # Check time budget
        if time.time() - start_time >= max_time:
            return best_val
            
        # Dither F per generation: Randomize between [0.5, 0.9]
        # This helps the algorithm adapt to different landscape scales.
        F = 0.5 + 0.4 * random.random()
        
        improved_this_gen = False
        
        # Iterate over population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            # Select distinct indices r1, r2 != i
            # Using random.randrange is faster than np.random.choice for scalars
            r1 = random.randrange(pop_size)
            while r1 == i: 
                r1 = random.randrange(pop_size)
            r2 = random.randrange(pop_size)
            while r2 == i or r2 == r1: 
                r2 = random.randrange(pop_size)
            
            # Mutation Strategy: DE/current-to-best/1
            # V = X_i + F*(X_best - X_i) + F*(X_r1 - X_r2)
            # This guides the search towards the best known solution (exploitation)
            # while the difference vector (r1-r2) adds diversity (exploration).
            mutant = pop[i] + F * (best_vec - pop[i]) + F * (pop[r1] - pop[r2])
            
            # Crossover: Binomial
            # Vectorized mask generation
            cross_mask = np.random.rand(dim) < CR
            # Ensure at least one parameter comes from the mutant
            j_rand = random.randrange(dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, pop[i])
            
            # Boundary Handling
            trial = np.clip(trial, min_b, max_b)
            
            # Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
                    improved_this_gen = True
                    stall_count = 0
        
        if not improved_this_gen:
            stall_count += 1
            
        # --- Restart Mechanism ---
        # If the algorithm has stalled (converged to a local optimum), restart.
        if stall_count >= patience:
            stall_count = 0
            
            # Generate new random population
            pop = get_random_pop(pop_size)
            
            # Elitism: Inject the best solution found so far into the new population
            # to ensure we never lose the best result.
            pop[0] = best_vec
            
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Evaluate the new population (skipping the elite at index 0)
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
                    # If random initialization found something better, reset stall
                    stall_count = 0
                    
    return best_val
