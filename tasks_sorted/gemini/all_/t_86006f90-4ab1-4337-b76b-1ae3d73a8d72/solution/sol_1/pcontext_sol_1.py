#An improved algorithm based on **Self-Adaptive Differential Evolution (jDE)** with a **Current-to-Best** mutation strategy and a **Restart Mechanism**.
#
#### Improvements Explanation:
#1.  **Current-to-Best Mutation**: Instead of random mutation (`rand/1/bin`), this algorithm moves individuals towards the best solution found so far (`current-to-best/1/bin`). This significantly speeds up convergence for minimization tasks.
#2.  **Self-Adaptation (jDE)**: The control parameters $F$ (mutation factor) and $CR$ (crossover probability) are not fixed. Each individual carries its own parameters which evolve. This eliminates the need for manual tuning and adapts to the function landscape.
#3.  **Restart Mechanism**: A critical flaw in standard DE is getting stuck in local optima (stagnation), likely the cause of the previous 105.4 result. This algorithm detects convergence (when population variance is low) and triggers a restart: it keeps the best solution found but re-initializes the rest of the population to explore new areas of the search space.
#4.  **Vectorized & Efficient**: Designed to maximize the number of generations within the `max_time` constraint by managing population size dynamically and checking time limits strictly.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Self-Adaptive Differential Evolution (jDE) 
    with 'current-to-best' mutation and population restart capabilities.
    """
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Algorithm Configuration ---
    # Population size: Adaptive to dimension but capped for performance
    # 'current-to-best' strategy works well with smaller populations (5*dim) compared to 'rand'
    pop_size = int(max(20, 5 * dim))
    if pop_size > 50: 
        pop_size = 50  # Cap population to ensure high generation count in limited time
    
    # Restart threshold: If population standard deviation drops below this, we restart
    convergence_tol = 1e-6
    
    # Bounds processing
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])
    diff_b = max_b - min_b
    
    # --- Initialization ---
    # Initialize population within bounds
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # jDE Control Parameters (Self-Adaptive)
    # Each individual has its own F and CR
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Global best tracking
    best_idx = -1
    best_val = float('inf')
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            # Return best found so far if time runs out during initialization
            return best_val if best_val != float('inf') else float('inf')
            
        val = func(pop[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_idx = i
            
    # --- Main Optimization Loop ---
    while True:
        # Check time at start of each generation
        if (datetime.now() - start_time) >= time_limit:
            return best_val

        # --- Restart Mechanism ---
        # If population has converged (low variance), restart to escape local optima
        # np.std returns nan if fitness contains inf, condition becomes False, so safe.
        if np.std(fitness) < convergence_tol:
            # Preserve the global best individual
            best_sol = pop[best_idx].copy()
            best_v = best_val
            
            # Re-initialize the rest of the population randomly
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_sol # Place best at index 0
            
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_v
            best_idx = 0
            
            # Reset jDE parameters for exploration
            F = np.full(pop_size, 0.5)
            CR = np.full(pop_size, 0.9)
            
            # Evaluate new population (skipping the preserved best at idx 0)
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_idx = i
            continue # Skip directly to next generation

        # --- Evolution Step ---
        for i in range(pop_size):
            # Strict time check before every function evaluation
            if (datetime.now() - start_time) >= time_limit:
                return best_val
                
            # 1. Parameter Adaptation (jDE Logic)
            # Generate trial parameters based on history
            f_i = F[i]
            cr_i = CR[i]
            
            # With 10% probability, generate new F in [0.1, 1.0]
            if np.random.rand() < 0.1:
                f_i = 0.1 + 0.9 * np.random.rand()
            
            # With 10% probability, generate new CR in [0.0, 1.0]
            if np.random.rand() < 0.1:
                cr_i = np.random.rand()
                
            # 2. Mutation: Current-to-Best/1
            # Formula: V = Xi + F*(Xbest - Xi) + F*(Xr1 - Xr2)
            # This balances exploitation (move to best) and exploration (diff between randoms)
            
            # Select distinct r1, r2 != i
            # (Simple rejection sampling is fast enough for small pop_size)
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
                
            x_best = pop[best_idx]
            x_i = pop[i]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            mutant = x_i + f_i * (x_best - x_i) + f_i * (x_r1 - x_r2)
            
            # 3. Crossover (Binomial)
            rand_vals = np.random.rand(dim)
            cross_mask = rand_vals < cr_i
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # 4. Boundary Constraint Handling (Clip)
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                # Trial is better, replace current
                fitness[i] = f_trial
                pop[i] = trial
                # Adopt the successful parameters
                F[i] = f_i
                CR[i] = cr_i
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_idx = i
    
    return best_val
