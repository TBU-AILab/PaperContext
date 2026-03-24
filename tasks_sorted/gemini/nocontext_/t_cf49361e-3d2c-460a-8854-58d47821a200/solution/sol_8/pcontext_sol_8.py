#Here is the implementation of a **Differential Evolution (DE)** algorithm tailored to the provided template.
#
#Differential Evolution is a robust evolutionary algorithm suitable for continuous optimization problems. It evolves a population of candidate solutions by combining existing ones, making it efficient at escaping local minima and refining the solution within the bounded search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Finds the minimum output value of func using Differential Evolution
    within a limited execution time.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # ---------------------------
    # Differential Evolution Parameters
    # ---------------------------
    # Population size: Higher allows more diversity, lower allows more generations.
    # We clamp it to ensure we get iterations done within strict time limits.
    pop_size = min(max(10, dim * 5), 50)
    
    # F: Differential weight (mutation scale), usually between 0.5 and 1.0
    F = 0.8 
    # CR: Crossover probability, usually between 0.5 and 1.0
    CR = 0.9 

    # Prepare bounds as numpy arrays for vectorization
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize best found value
    best = float('inf')
    
    # ---------------------------
    # Initialization Phase
    # ---------------------------
    # We initialize the population one by one to check time constraints 
    # immediately if the function evaluation is very slow.
    population = np.zeros((pop_size, dim))
    fitnesses = np.full(pop_size, float('inf'))

    for i in range(pop_size):
        # Check time constraint
        if (datetime.now() - start_time) >= time_limit:
            return best

        # Generate random individual within bounds
        ind = min_b + np.random.rand(dim) * diff_b
        population[i] = ind
        
        # Evaluate
        val = func(ind)
        fitnesses[i] = val
        
        if val < best:
            best = val

    # ---------------------------
    # Evolution Phase
    # ---------------------------
    while True:
        for i in range(pop_size):
            # Check time constraint strictly before every evaluation
            if (datetime.now() - start_time) >= time_limit:
                return best

            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != i
            # We construct a pool of indices excluding current 'i'
            idxs = [idx for idx in range(pop_size) if idx != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]

            # Generate mutant vector: v = a + F * (b - c)
            mutant = a + F * (b - c)
            
            # Clip mutant to ensure it stays within bounds
            mutant = np.clip(mutant, min_b, max_b)

            # 2. Crossover: Create trial vector
            # Binomial crossover: pick parameter from mutant if rand < CR
            cross_points = np.random.rand(dim) < CR
            
            # Ensure at least one parameter comes from mutant to differ from parent
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True

            trial = np.where(cross_points, mutant, population[i])

            # 3. Selection: Evaluate trial vector
            f_trial = func(trial)

            # Greedy selection: if trial is better or equal, replace current
            if f_trial <= fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial
                
                # Update global best
                if f_trial < best:
                    best = f_trial

    # return fitness of the best found solution
    return best
