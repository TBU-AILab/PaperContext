#Here is a fully operational, self-contained implementation using the **Differential Evolution** algorithm. 
#
#Differential Evolution is a robust evolutionary algorithm suitable for black-box global optimization. It is significantly more efficient than random search for finding minimums in multi-dimensional spaces within a limited timeframe.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE).
    
    DE is a population-based optimization algorithm. It maintains a population of 
    candidate solutions and creates new candidates by combining existing ones 
    via mutation and crossover.
    """
    start_time = time.time()
    
    # --- Hyperparameters ---
    # Population size: Higher allows more exploration but fewer generations in limited time.
    # We adapt it to the dimension but cap it to ensuring responsiveness.
    pop_size = max(15, int(dim * 3))
    if pop_size > 60: 
        pop_size = 60
        
    mutation_factor = 0.8  # Scaling factor (F)
    crossover_prob = 0.7   # Crossover probability (CR)

    # --- Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population with random values within bounds
    # Shape: (pop_size, dim)
    pop = min_b + diff_b * np.random.rand(pop_size, dim)
    
    # Store fitness of population to minimize function calls
    pop_fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')

    # Evaluate initial population
    for i in range(pop_size):
        # Strict time check before every function evaluation
        if (time.time() - start_time) >= max_time:
            return best_fitness
            
        fitness = func(pop[i])
        pop_fitness[i] = fitness
        
        if fitness < best_fitness:
            best_fitness = fitness

    # --- Main Optimization Loop ---
    while True:
        # Loop through the population
        for i in range(pop_size):
            # Strict time check
            if (time.time() - start_time) >= max_time:
                return best_fitness

            # 1. Mutation (Strategy: rand/1)
            # Select 3 random unique indices distinct from current index 'i'
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]

            # Generate mutant vector
            mutant = a + mutation_factor * (b - c)
            
            # Enforce bounds (clipping)
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover (Binomial)
            # Create a mask for parameters to inherit from mutant
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter comes from the mutant (standard DE requirement)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # Construct trial vector
            trial = np.where(cross_points, mutant, pop[i])

            # 3. Selection
            # Calculate fitness of trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better, replace parent
            if f_trial < pop_fitness[i]:
                pop[i] = trial
                pop_fitness[i] = f_trial
                
                # Update global best found so far
                if f_trial < best_fitness:
                    best_fitness = f_trial

    # Fallback return (though logic dictates return occurs inside loop via timeout)
    return best_fitness
