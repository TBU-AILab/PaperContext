#To improve upon the previous algorithm, I propose an implementation of **Differential Evolution (DE) with a "Best/1/Bin" strategy, Parameter Dithering, and Automatic Restarts**.
#
#**Why this is better:**
#1.  **Fast Convergence (`best/1/bin`)**: Unlike the previous `rand/1/bin` (which is slow and explores randomly), this strategy pulls the population toward the best known solution. This finds minima much faster.
#2.  **Automatic Restarts**: Aggressive convergence can get stuck in local minima. To solve this, the algorithm detects stagnation (when fitness variance is low or no improvement occurs) and restarts the population entirely. This effectively turns the available `max_time` into a multi-start global search.
#3.  **Vectorization**: The generation of mutant vectors and crossover is vectorized using NumPy, reducing Python loop overhead significantly compared to the previous solution.
#4.  **Parameter Dithering**: The scaling factor `F` is randomized (dithered) per generation. This prevents the search pattern from becoming rigid, helping to escape shallow basins of attraction.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Pre-process bounds for vectorization
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    bound_range = ub - lb
    
    # Track global best
    best_fitness = float('inf')
    
    # Hyperparameters
    # A slightly larger population than standard (15*dim) allows better diversity 
    # before the aggressive "best" strategy converges.
    pop_size = max(20, 15 * dim) 
    
    # Crossover Rate (CR) fixed high for preserving structure of good solutions
    CR = 0.9 

    # Main optimization loop (Restart Loop)
    while True:
        # Check strict time limit
        if datetime.now() >= end_time:
            return best_fitness

        # --- Initialization Phase ---
        # Generate random population within bounds
        pop = lb + np.random.rand(pop_size, dim) * bound_range
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        current_best_idx = 0
        for i in range(pop_size):
            if datetime.now() >= end_time: return best_fitness
            
            val = func(pop[i])
            fitness[i] = val
            
            if val < best_fitness:
                best_fitness = val
            if val < fitness[current_best_idx]:
                current_best_idx = i

        # --- Evolution Phase ---
        stagnation_counter = 0
        last_gen_best_val = fitness[current_best_idx]

        # Run generations until stagnation or time limit
        while True:
            if datetime.now() >= end_time:
                return best_fitness

            # Dithering: Randomize F slightly to prevent search pattern rigidity
            # F in range [0.5, 1.0]
            F = 0.5 + 0.5 * np.random.rand()

            # --- Vectorized Mutation & Crossover ---
            # Strategy: DE/best/1/bin
            # V = Best + F * (r1 - r2)
            
            # Select random indices r1, r2
            # (ignoring self/collision checks for speed, negligible impact in larger pops)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Calculate difference vectors (Vectorized)
            diffs = pop[r1] - pop[r2]
            
            # Create mutant vectors based on the BEST individual in current generation
            # Broadcasting pop[current_best_idx] across the matrix
            best_vec = pop[current_best_idx]
            mutants = best_vec + F * diffs
            
            # Crossover (Binomial)
            # Create a mask where we accept the mutant
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < CR
            
            # Create trial population
            trials = np.where(cross_mask, mutants, pop)
            
            # Boundary constraint (Clip)
            trials = np.clip(trials, lb, ub)
            
            # --- Selection Phase ---
            # Evaluate trials and update population
            # This part requires a loop as func is not guaranteed to be vectorized
            improved_gen = False
            
            for i in range(pop_size):
                if datetime.now() >= end_time: return best_fitness
                
                # Evaluate trial
                f_trial = func(trials[i])
                
                # Greedy selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trials[i]
                    improved_gen = True
                    
                    # Update global best
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                    
                    # Update generation best index
                    if f_trial < fitness[current_best_idx]:
                        current_best_idx = i

            # --- Stagnation / Restart Logic ---
            # Check if the best value in the population has improved
            gen_best_val = fitness[current_best_idx]
            
            if gen_best_val < last_gen_best_val:
                last_gen_best_val = gen_best_val
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Calculate population standard deviation to detect convergence
            pop_std = np.std(fitness)
            
            # Restart if:
            # 1. Standard deviation is tiny (all individuals converged to one spot)
            # 2. No improvement for significant number of generations (stuck in local min)
            if pop_std < 1e-6 or stagnation_counter > 20:
                break # Break inner loop to trigger restart in outer loop

    return best_fitness
