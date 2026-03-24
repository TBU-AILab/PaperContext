#The following algorithm implements a **Self-Adaptive Differential Evolution (jDE)** strategy with a **Restart Mechanism**. 
#
#**Key Improvements over standard algorithms:**
#1.  **Self-Adaptation**: Instead of fixed parameters, the mutation factor ($F$) and crossover probability ($CR$) are encoded into each individual and evolve. This allows the algorithm to automatically tune itself to the specific function landscape.
#2.  **Restart Mechanism**: To utilize the available `max_time` effectively, the algorithm detects convergence (when the population variance becomes negligible) and restarts the search while preserving the best solution found so far. This significantly reduces the risk of getting stuck in local minima (which likely caused the previous score of ~98).
#3.  **Robust Initialization**: Uses a population size scaled to the dimension to ensure adequate coverage.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # --- Configuration ---
    # Self-Adaptive Differential Evolution (jDE) with Restart
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # Population size: 15*dim is a robust baseline, clipped for performance constraints
    # We ensure a minimum of 20 and maximum of 100 to balance exploration speed.
    pop_size = max(20, min(100, 15 * dim))
    
    # Parse bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Helper to Initialize Population ---
    def init_population(p_size, d):
        return min_b + np.random.rand(p_size, d) * diff_b

    # Initialize variables
    population = init_population(pop_size, dim)
    
    # jDE Control Parameters initialized randomly
    # F in [0.1, 1.0], CR in [0.0, 1.0]
    F = 0.1 + 0.9 * np.random.rand(pop_size)
    CR = np.random.rand(pop_size)
    
    # Fitness tracking
    fitness = np.zeros(pop_size)
    best_val = float('inf')
    best_idx = -1
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_val if best_val != float('inf') else func(population[i])
            
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_idx = i
            
    # --- Main Optimization Loop ---
    while (datetime.now() - start_time) < time_limit:
        
        # 1. Restart Mechanism
        # If the population has converged (low variance), restart to find better minima
        # We keep the single best individual and re-initialize the rest.
        if np.std(fitness) < 1e-6 and (datetime.now() - start_time) < time_limit / 2:
            # Preserve best
            best_ind = population[best_idx].copy()
            best_f = F[best_idx]
            best_cr = CR[best_idx]
            
            # Re-init population and parameters
            population = init_population(pop_size, dim)
            F = 0.1 + 0.9 * np.random.rand(pop_size)
            CR = np.random.rand(pop_size)
            
            # Place best back at index 0
            population[0] = best_ind
            F[0] = best_f
            CR[0] = best_cr
            fitness = np.zeros(pop_size)
            fitness[0] = best_val
            
            # Re-evaluate the new individuals (skipping index 0)
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                val = func(population[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_idx = i
            continue # Skip to next iteration
            
        # 2. Parameter Adaptation Logic (Pre-calculation)
        # Generate random values to decide if F/CR should change
        rand1 = np.random.rand(pop_size)
        rand2 = np.random.rand(pop_size)
        rand3 = np.random.rand(pop_size) # New F value source
        rand4 = np.random.rand(pop_size) # New CR value source
        
        # Trial parameters
        # If rand < 0.1, pick new parameter, else keep old
        tau1 = 0.1
        tau2 = 0.1
        
        trial_F = np.where(rand1 < tau1, 0.1 + 0.9 * rand3, F)
        trial_CR = np.where(rand2 < tau2, rand4, CR)
        
        # 3. Evolution Cycle
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_val
            
            # Mutation: DE/rand/1
            # Pick 3 distinct random indices != i
            # Simple rejection sampling is efficient for small k=3
            idxs = np.random.randint(0, pop_size, 3)
            while i in idxs or idxs[0] == idxs[1] or idxs[0] == idxs[2] or idxs[1] == idxs[2]:
                idxs = np.random.randint(0, pop_size, 3)
            r1, r2, r3 = idxs
            
            # Create mutant vector: v = x_r1 + F * (x_r2 - x_r3)
            f_curr = trial_F[i]
            mutant = population[r1] + f_curr * (population[r2] - population[r3])
            
            # Boundary Constraint: Clip
            mutant = np.clip(mutant, min_b, max_b)
            
            # Crossover: Binomial
            cr_curr = trial_CR[i]
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < cr_curr
            mask[j_rand] = True # Ensure at least one parameter is from mutant
            
            trial = np.where(mask, mutant, population[i])
            
            # Selection
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                # Trial is better or equal; replace parent
                population[i] = trial
                fitness[i] = f_trial
                # Adopt the successful parameters
                F[i] = trial_F[i]
                CR[i] = trial_CR[i]
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i

    return best_val
