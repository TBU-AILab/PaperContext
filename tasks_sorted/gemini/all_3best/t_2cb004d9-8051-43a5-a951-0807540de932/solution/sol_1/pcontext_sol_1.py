#The following Python code implements a **Self-Adaptive Differential Evolution (jDE) algorithm with Restarts**.
#
#### Improvements over standard Differential Evolution:
#1.  **Self-Adaptation (jDE)**: Instead of using fixed values for mutation ($F$) and crossover ($CR$), each individual in the population carries its own $F$ and $CR$ values. These parameters evolve along with the solution, allowing the algorithm to automatically adapt to the specific function landscape (e.g., changing search step sizes dynamically).
#2.  **Restart Mechanism**: If the population converges (i.e., variance in fitness becomes negligible) before the time limit expires, the algorithm restarts the population while preserving the global best found so far. This prevents the solver from idling in a local optimum and allows it to explore other basins of attraction within the available time.
#3.  **Optimized Time Management**: Time is checked aggressively (per individual evaluation) to ensure the strict `max_time` constraint is respected.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE) with Restarts.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Initialization ---
    # Convert bounds to numpy arrays for efficient vectorized operations
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Population size: Dynamic based on dimension
    # We use a moderate size to balance diversity and generation speed.
    # Capped at 50 to ensure rapid iterations within time limits.
    pop_size = int(max(10, 10 * dim))
    if pop_size > 50:
        pop_size = 50

    # Global best fitness tracker
    best_fitness = float('inf')
    
    # jDE Control Parameter probabilities (probabilities to update F and CR)
    tau_f = 0.1
    tau_cr = 0.1
    
    # --- Restart Loop ---
    # The algorithm restarts with a new random population if it converges 
    # (stagnates) while time still remains.
    while True:
        
        # Check time before starting initialization
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness
            
        # Initialize Population randomly within bounds
        population = min_b + np.random.rand(pop_size, dim) * diff_b
        fitnesses = np.full(pop_size, float('inf'))
        
        # Initialize jDE Control Parameters (F and CR) for each individual
        # F (Mutation factor) initialized to 0.5
        # CR (Crossover prob) initialized to 0.9
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # Evaluate Initial Population
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            val = func(population[i])
            fitnesses[i] = val
            
            if val < best_fitness:
                best_fitness = val
                
        # --- Evolution Loop ---
        while True:
            # Check time
            if (datetime.now() - start_time) >= time_limit:
                return best_fitness
            
            # Check for Convergence (Stagnation)
            # If population diversity is lost (fitnesses are extremely close), 
            # break the inner loop to trigger a restart.
            if np.std(fitnesses) < 1e-6 or (np.max(fitnesses) - np.min(fitnesses)) < 1e-6:
                break 
                
            # Iterate through population (Generation)
            for i in range(pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_fitness
                
                # 1. Parameter Adaptation (jDE logic)
                # Update Mutation Factor F
                f_new = F[i]
                if np.random.rand() < tau_f:
                    f_new = 0.1 + np.random.rand() * 0.9 # F in range [0.1, 1.0]
                
                # Update Crossover Probability CR
                cr_new = CR[i]
                if np.random.rand() < tau_cr:
                    cr_new = np.random.rand() # CR in range [0.0, 1.0]
                
                # 2. Mutation Strategy: rand/1/bin
                # Select 3 unique random indices distinct from current i
                while True:
                    idxs = np.random.randint(0, pop_size, 3)
                    if np.unique(idxs).size == 3 and i not in idxs:
                        r1, r2, r3 = idxs
                        break
                
                a, b, c = population[r1], population[r2], population[r3]
                
                # Generate mutant vector
                mutant = a + f_new * (b - c)
                
                # 3. Crossover
                # Create trial vector by mixing mutant and target
                # Ensure at least one dimension is taken from mutant to avoid duplication
                rand_j = np.random.randint(dim)
                mask = np.random.rand(dim) < cr_new
                mask[rand_j] = True 
                
                trial = np.where(mask, mutant, population[i])
                
                # Boundary Constraint Handling (Clipping)
                trial = np.clip(trial, min_b, max_b)
                
                # 4. Selection
                trial_fitness = func(trial)
                
                if trial_fitness <= fitnesses[i]:
                    # Trial is better or equal; accept it
                    population[i] = trial
                    fitnesses[i] = trial_fitness
                    # Update control parameters for this individual (learning)
                    F[i] = f_new
                    CR[i] = cr_new
                    
                    # Update Global Best
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
    
    return best_fitness
