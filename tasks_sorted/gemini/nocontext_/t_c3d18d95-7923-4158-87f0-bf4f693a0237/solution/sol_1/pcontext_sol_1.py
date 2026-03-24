#Here is a fully operational, self-contained Python algorithm based on **Differential Evolution (DE)**.
#
#DE is a stochastic, population-based optimization algorithm that is generally more robust and efficient than random search, particularly for continuous search spaces. It maintains a population of candidate solutions and combines them to create new candidates, moving the population toward the global minimum.
#
#The implementation handles the time constraint strictly by checking the clock after every function evaluation.
#
import numpy as np
from datetime import datetime, timedelta
import random

def run(func, dim, bounds, max_time):
    """
    Finds the set of input parameter values that lead to a minimum output value
    within a limited time using Differential Evolution.
    """
    
    # --- Configuration ---
    # Population size: Balance between exploration and iteration speed.
    # We use a dynamic size based on dimension but capped to ensure fast startup.
    pop_size = max(10, int(dim * 10))
    if pop_size > 50: pop_size = 50  # Cap population to allow iterations within strict time limits
    
    # DE Hyperparameters
    mutation_factor = 0.8  # F: Weight of the difference vector
    crossover_prob = 0.7   # CR: Probability of crossover
    
    # Time management
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Pre-process bounds for numpy efficiency
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population
    # We generate the population one by one to ensure we don't timeout 
    # if func() is extremely slow or pop_size is massive.
    population = []
    fitnesses = []
    
    best = float('inf')
    best_idx = -1
    
    # --- Phase 1: Initialization ---
    for i in range(pop_size):
        # Check time constraint
        if (datetime.now() - start_time) >= time_limit:
            return best
        
        # Random vector within bounds
        # shape: (dim,)
        indv = min_b + np.random.rand(dim) * diff_b
        
        # Evaluate
        try:
            score = func(indv)
        except Exception:
            score = float('inf') # Handle potential function errors gracefully
            
        population.append(indv)
        fitnesses.append(score)
        
        # Update best found so far
        if score < best:
            best = score
            best_idx = i

    # Convert lists to numpy arrays for vectorized operations in Phase 2
    population = np.array(population)
    fitnesses = np.array(fitnesses)
    
    # --- Phase 2: Evolution Loop ---
    # We continue until time runs out
    while True:
        for i in range(pop_size):
            # Check time constraint before every evaluation
            if (datetime.now() - start_time) >= time_limit:
                return best
            
            # 1. Mutation: Select 3 distinct random individuals (a, b, c) != i
            # Create a pool of indices excluding current 'i'
            idxs = [idx for idx in range(pop_size) if idx != i]
            
            # Safety check: if population is too small (e.g. < 4), DE logic fails.
            # Fallback to pure random mutation if pop_size is tiny.
            if len(idxs) < 3:
                a_idx, b_idx, c_idx = idxs[0], idxs[0], idxs[0] # Degenerate case
            else:
                # Efficient random sampling
                a_idx, b_idx, c_idx = random.sample(idxs, 3)
            
            a = population[a_idx]
            b = population[b_idx]
            c = population[c_idx]
            
            # DE/rand/1 strategy: v = a + F * (b - c)
            mutant = a + mutation_factor * (b - c)
            
            # Constrain to bounds (Clip)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 2. Crossover: Create trial vector
            # Binomial crossover
            cross_points = np.random.rand(dim) < crossover_prob
            
            # Ensure at least one parameter changes
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            # trial[j] = mutant[j] if crossover else population[i][j]
            trial = np.where(cross_points, mutant, population[i])
            
            # 3. Selection
            try:
                f_trial = func(trial)
            except Exception:
                f_trial = float('inf')
            
            # Greedy selection
            if f_trial < fitnesses[i]:
                fitnesses[i] = f_trial
                population[i] = trial
                
                # Update global best
                if f_trial < best:
                    best = f_trial

    return best
