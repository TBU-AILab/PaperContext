#Here is an improved optimization algorithm. 
#
#It implements a **Restart Differential Evolution (DE) with a 'Current-to-Best' mutation strategy**. This approach is significantly more aggressive at converging than standard DE (which often led to the result ~40 in the previous attempt) and uses a restart mechanism to escape local optima if the population stagnates or converges too early. It also utilizes vectorized `numpy` operations for the mathematical steps to maximize the number of generations possible within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Restart Differential Evolution with current-to-best mutation.
    
    Key improvements over standard DE:
    1. 'current-to-best/1/bin' strategy accelerates convergence by using gradient information 
       implicit in the population's best member.
    2. Restart mechanism: If the population converges (low variance) or stagnates, 
       the algorithm resets with new random candidates to explore different basins of attraction.
    3. Vectorized trial vector generation reduces overhead.
    """
    
    # 1. Setup Timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # 2. Parameters
    # Population size: Balance between diversity (high) and speed (low).
    # 15*dim is a robust rule of thumb, clipped to keep iterations fast.
    pop_size = int(15 * dim)
    pop_size = np.clip(pop_size, 10, 100)
    
    # Dithering range for F (Scaling Factor). 
    # Randomizing F per individual helps maintain diversity.
    F_min, F_max = 0.5, 0.9 
    CR = 0.9 # Crossover Probability
    
    # 3. Pre-process Bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # Global best tracking
    best = float('inf')
    
    # 4. Main Loop (Restarts)
    while True:
        # Check time before starting a new population
        if datetime.now() - start_time >= time_limit:
            return best

        # --- Initialize Population ---
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Local best for the current restart (needed for mutation strategy)
        local_best_val = float('inf')
        local_best_vec = np.zeros(dim)

        # Evaluate Initial Population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best
            
            val = func(population[i])
            fitness[i] = val
            
            if val < local_best_val:
                local_best_val = val
                local_best_vec = population[i].copy()
            
            if val < best:
                best = val

        # --- Evolution Loop ---
        stagnation_count = 0
        
        while True:
            # Time check inside generation loop
            if datetime.now() - start_time >= time_limit:
                return best
            
            # --- Mutation: DE/current-to-best/1 ---
            # Formula: V = X + F * (X_best - X) + F * (X_r1 - X_r2)
            # This pulls individuals towards the best found so far while maintaining exploration.
            
            # 1. Generate F for each individual (Dithering)
            F = F_min + (F_max - F_min) * np.random.rand(pop_size, 1)
            
            # 2. Select Indices for r1 and r2
            # We use array shifting to ensure r1 != r2 != i efficiently without loops
            idxs = np.arange(pop_size)
            
            shift1 = np.random.randint(1, pop_size)
            shift2 = np.random.randint(1, pop_size)
            # Ensure shift1 != shift2 to guarantee r1 != r2
            while shift1 == shift2:
                shift2 = np.random.randint(1, pop_size)
                
            idx_r1 = np.roll(idxs, shift1)
            idx_r2 = np.roll(idxs, shift2)
            
            # 3. Compute Mutant Vectors
            # X_best is broadcasted. 'population' is X_current.
            x_r1 = population[idx_r1]
            x_r2 = population[idx_r2]
            
            mutant = population + F * (local_best_vec - population) + F * (x_r1 - x_r2)
            
            # --- Crossover (Binomial) ---
            rand_vals = np.random.rand(pop_size, dim)
            mask = rand_vals < CR
            
            # Guarantee at least one parameter is inherited from mutant
            j_rand = np.random.randint(0, dim, pop_size)
            mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(mask, mutant, population)
            
            # --- Boundary Constraints ---
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection ---
            improved_in_gen = False
            
            # Evaluate trials one by one (cannot vectorize func call)
            for i in range(pop_size):
                if datetime.now() - start_time >= time_limit:
                    return best
                
                f_trial = func(trial_pop[i])
                
                # Greedy selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial_pop[i]
                    
                    # Update Local Best
                    if f_trial < local_best_val:
                        local_best_val = f_trial
                        local_best_vec = trial_pop[i].copy()
                        improved_in_gen = True
                        
                        # Update Global Best
                        if f_trial < best:
                            best = f_trial
            
            # --- Stagnation Detection ---
            # If population variance is very low, or no improvement for many generations,
            # break to outer loop to restart.
            
            if not improved_in_gen:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            # Heuristic thresholds for restart
            if stagnation_count > 20: 
                break
                
            if np.std(fitness) < 1e-6:
                break

    return best
