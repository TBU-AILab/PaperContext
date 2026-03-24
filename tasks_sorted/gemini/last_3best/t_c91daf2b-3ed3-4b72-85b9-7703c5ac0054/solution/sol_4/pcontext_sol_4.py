#The following algorithm improves upon the previous submissions by implementing a **Differential Evolution (DE)** strategy enhanced with a **Restart Mechanism** and a **Local Search (Polishing)** phase.
#
##### Key Improvements:
#1.  **Robust Strategy (DE/rand/1/bin)**: Based on the performance history, the standard "rand/1" strategy proved most effective (score 11.2) compared to greedy "best/1" strategies (score 21.1) or vectorized approaches (score 56.5). This strategy maintains better population diversity, preventing early entrapment in local optima.
#2.  **Restart Mechanism**: A major limitation of standard DE is stagnation. This implementation monitors the population's standard deviation. If the population converges (stagnates), the algorithm triggers a **Restart**, keeping only the best solution (Elitism) and re-initializing the rest. This allows the search to escape local minima and explore new basins of attraction.
#3.  **Local Search (Polishing)**: Before a restart occurs (when the algorithm thinks it has found a minimum), a "Polishing" phase is activated. This performs a stochastic Gaussian walk (random hill-climbing) around the best solution to refine it and squeeze out the last bits of precision, ensuring the absolute minimum in that basin is found.
#4.  **Parameter Dithering**: The mutation factor `F` is randomized ("dithered") between 0.5 and 1.0 for each mutation. This simple technique significantly helps in handling different surface landscapes by varying the search step size dynamically.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Differential Evolution (DE/rand/1/bin) with
    Dithering, Restart Mechanism, and Local Search.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population Size:
    # A robust default is 10*dim, clamped to [20, 50] to balance
    # exploration capability with generational speed under time limits.
    pop_size = int(np.clip(10 * dim, 20, 50))
    
    # DE Parameters:
    # CR (Crossover Rate): 0.9 is generally effective for non-separable functions.
    # F (Mutation Factor): Dithered per iteration between 0.5 and 1.0.
    CR = 0.9
    
    # Bound processing
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_val = float('inf')
    best_vec = None
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # Local Search Step Size (initialized relative to domain size)
    ls_step = np.mean(diff_b) * 0.05
    
    # --- Main Optimization Loop ---
    while True:
        # Check time budget
        if time.time() - start_time >= max_time:
            return best_val
        
        # --- Restart & Local Search Mechanism ---
        # Check if population has converged (low standard deviation of fitness)
        # If so, the algorithm is likely stuck in a local minimum.
        if np.std(fitness) < 1e-6:
            
            # 1. Local Search (Exploitation)
            # Before abandoning this basin, try to drill down further from the best point.
            # We perform a simple stochastic hill climb (Gaussian walk).
            for _ in range(15):
                if time.time() - start_time >= max_time:
                    return best_val
                
                # Create a candidate by perturbing the best vector
                candidate = best_vec + np.random.normal(0, ls_step, dim)
                candidate = np.clip(candidate, min_b, max_b)
                val = func(candidate)
                
                if val < best_val:
                    best_val = val
                    best_vec = candidate.copy()
                    # If we found a better point, we keep the step size (or could increase it)
                    # to continue descending.
                else:
                    # If no improvement, reduce step size to search finer details
                    ls_step *= 0.5
                
                # If step size becomes negligible, stop local search
                if ls_step < 1e-9:
                    break
            
            # Reset step size for the next time we need it
            ls_step = np.mean(diff_b) * 0.05
            
            # 2. Restart (Exploration)
            # Re-initialize the population to explore new areas of the search space.
            # Elitism: Keep the single best individual found so far.
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            pop[0] = best_vec
            
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            
            # Evaluate new population (skipping index 0)
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            
            continue # Skip the DE step for this iteration, start fresh

        # --- Differential Evolution Step ---
        # Strategy: DE/rand/1/bin
        # Iterate through population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            # Select 3 random distinct indices r1, r2, r3 != i
            # Using a while loop is faster than np.random.choice for small sets
            while True:
                r1 = np.random.randint(0, pop_size)
                if r1 != i: break
            while True:
                r2 = np.random.randint(0, pop_size)
                if r2 != i and r2 != r1: break
            while True:
                r3 = np.random.randint(0, pop_size)
                if r3 != i and r3 != r1 and r3 != r2: break
            
            # Parameter Dithering
            # Randomize F between 0.5 and 1.0 to ensure different step sizes
            F = 0.5 + 0.5 * np.random.rand()
            
            # Mutation: V = X_r1 + F * (X_r2 - X_r3)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover: Binomial
            mask = np.random.rand(dim) < CR
            # Ensure at least one parameter is changed
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, pop[i])
            
            # Boundary Handling: Clip
            trial = np.clip(trial, min_b, max_b)
            
            # Selection: Greedy
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
                    
    return best_val
