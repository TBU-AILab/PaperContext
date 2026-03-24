#Here is a fully operational, self-contained Python algorithm using a **Memetic Differential Evolution** strategy. 
#
#This approach improves upon standard Differential Evolution by:
#1.  Using a **"Current-to-Best" mutation strategy** ($DE/current-to-best/1/bin$), which converges faster than random selection by utilizing information from the best individuals.
#2.  Implementing a **Restart Mechanism** to escape local minima if the population stagnates or converges prematurely.
#3.  Incorporating a **Local Search (Hill Climbing)** phase in the final moments of the allocated time to polish the best solution found, squeezing out potential improvements in precision.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Memetic Differential Evolution with Restarts.
    Combines global search (DE) with local polishing and restart capabilities.
    """
    # Initialize timing
    start_time = datetime.now()
    # Use a safe time limit slightly buffered to ensure return
    time_limit = timedelta(seconds=max_time)
    
    # Helper to check elapsed time safely
    def get_elapsed():
        return datetime.now() - start_time

    # Pre-process bounds for efficient numpy operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # --- Hyperparameters ---
    # Population size: Adaptive to dimension.
    # 15*dim is a robust heuristic, clamped to [30, 80] to balance speed and diversity.
    pop_size = int(np.clip(15 * dim, 30, 80))
    
    # Differential Evolution Constants
    CR = 0.9  # Crossover probability (high CR often better for inseparable functions)
    # Mutation factor F is randomized (dithered) in the loop
    
    # Restart triggers
    stagnation_limit = 40  # Max generations without global improvement before restart
    convergence_tol = 1e-7 # Population standard deviation threshold

    # --- Initialization ---
    # Initialize Population uniformly within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Global best tracking
    best_fitness = float('inf')
    best_pos = None

    # Evaluate initial population
    for i in range(pop_size):
        if get_elapsed() >= time_limit:
            # If time out during init, return best found or infinity
            return best_fitness if best_pos is not None else float('inf')
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_pos = pop[i].copy()

    stagnation_counter = 0

    # --- Main Optimization Loop ---
    while True:
        elapsed = get_elapsed()
        if elapsed >= time_limit:
            return best_fitness

        # --- Phase 2: Local Search Polish (End Game) ---
        # If 90% of time is used, switch to a greedy Hill Climber to refine the best solution
        if elapsed >= time_limit * 0.90:
            # Start with a small step size relative to domain
            step_size = diff_b * 0.01 
            
            while get_elapsed() < time_limit:
                # Generate candidate via Gaussian perturbation
                noise = np.random.normal(0, 1, dim) * step_size
                candidate = np.clip(best_pos + noise, min_b, max_b)
                
                f_cand = func(candidate)
                
                if f_cand < best_fitness:
                    best_fitness = f_cand
                    best_pos = candidate
                    # If improved, keep step size (or slightly expand) to follow gradient
                else:
                    # If failed, shrink step size to zoom in on minimum
                    step_size *= 0.95
                    
                # Reset step size if it becomes too small to float precision effectively
                if np.mean(step_size) < 1e-9:
                     step_size = diff_b * 0.001
            
            return best_fitness

        # --- Phase 1: Differential Evolution (Global Search) ---
        
        # Dither F (Mutation Factor) per generation [0.5, 1.0] to maintain diversity
        F = 0.5 + 0.5 * np.random.rand()
        
        # Identify best individual in CURRENT population (for mutation strategy)
        idx_best_in_pop = np.argmin(fitness)
        x_best_pop = pop[idx_best_in_pop]
        
        # Vectorized Mutation: DE/current-to-best/1
        # V = Xi + F * (Xbest - Xi) + F * (Xr1 - Xr2)
        
        # Generate random indices r1, r2
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Calculate mutant vectors for the whole population
        mutant = pop + F * (x_best_pop - pop) + F * (pop[r1] - pop[r2])
        
        # Crossover (Binomial)
        # Create mask for crossover
        cross_mask = np.random.rand(pop_size, dim) < CR
        
        # Ensure at least one parameter comes from mutant for each vector
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        # Create trial population
        trial_pop = np.where(cross_mask, mutant, pop)
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # Selection Step
        improved_global = False
        
        for i in range(pop_size):
            # Strict time check inside evaluation loop
            if get_elapsed() >= time_limit:
                return best_fitness
            
            f_trial = func(trial_pop[i])
            
            # Greedy selection
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial_pop[i]
                
                # Update global best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_pos = trial_pop[i].copy()
                    improved_global = True
        
        # --- Restart Mechanism ---
        if improved_global:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # Check for population convergence or long-term stagnation
        pop_std = np.std(fitness)
        if pop_std < convergence_tol or stagnation_counter > stagnation_limit:
            # Perform Restart: Re-initialize population to find new basins of attraction
            # We keep the global best safely stored in `best_pos` / `best_fitness`
            
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            fitness = np.full(pop_size, float('inf'))
            stagnation_counter = 0
            
            # Re-evaluate new population
            for i in range(pop_size):
                if get_elapsed() >= time_limit:
                    return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_pos = pop[i].copy()

    return best_fitness
