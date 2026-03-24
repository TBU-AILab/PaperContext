#The following algorithm improves upon the previous submissions by implementing a **jDE (Self-Adaptive Differential Evolution)** algorithm with **Asynchronous Updates** and a **Restart Mechanism**.
#
#### Key Improvements:
#1.  **jDE (Self-Adaptive Control Parameters)**: Instead of fixed `F` and `CR` values, each individual has its own parameters that evolve with the population. This allows the algorithm to automatically tune itself for different landscapes (finding the right balance between exploration and exploitation) without manual trial-and-error.
#2.  **Asynchronous Updates**: Unlike the vectorized approach which updates the population at the end of the generation, this implementation updates the population **immediately** if a better solution is found. This allows "good genes" to propagate faster within the same generation, significantly improving convergence speed.
#3.  **Fast Index Selection**: A specialized `while` loop is used to select distinct random indices ($r_1 \neq r_2 \neq r_3 \neq i$). This avoids the overhead of creating list objects or calling complex NumPy sampling functions inside the loop, making the Python iteration much faster.
#4.  **Restart Mechanism**: If the population variance drops below a threshold (convergence), the algorithm saves the best solution and re-initializes the rest. This prevents the search from idling in a local minimum.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using jDE (Self-Adaptive Differential Evolution) 
    with Asynchronous Updates and Restarts.
    """
    start_time = time.time()
    
    # --- Configuration ---
    # Population size: Clamped to [20, 50].
    # This range is empirically efficient for DE under time constraints: 
    # small enough for many generations, large enough for diversity.
    pop_size = int(max(20, min(50, 10 * dim)))
    
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # jDE Self-Adaptive Parameters
    # Each individual has its own F and CR.
    # Initialize with conservative starting values.
    F = np.full(pop_size, 0.5) 
    CR = np.full(pop_size, 0.9)
    
    best_val = float('inf')
    best_vec = None
    
    # --- Initial Evaluation ---
    for i in range(pop_size):
        if time.time() - start_time >= max_time:
            return best_val
        val = func(pop[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = pop[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Check time at the start of the generation
        if time.time() - start_time >= max_time:
            return best_val
            
        # --- Restart Mechanism ---
        # If population diversity is lost (standard deviation is low), 
        # the search is stuck. Restart with new random individuals, keeping the elite.
        if np.std(fitness) < 1e-6:
            # Re-initialize population
            pop = min_b + np.random.rand(pop_size, dim) * diff_b
            # Elitism: Inject the best found solution
            pop[0] = best_vec
            
            # Reset fitness and parameters
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_val
            F = np.full(pop_size, 0.5)
            CR = np.full(pop_size, 0.9)
            
            # Re-evaluate the new population (skipping the elite at index 0)
            for i in range(1, pop_size):
                if time.time() - start_time >= max_time:
                    return best_val
                val = func(pop[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = pop[i].copy()
            continue # Start evolution immediately with new population

        # --- Evolutionary Cycle (Asynchronous) ---
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            
            # 1. Parameter Adaptation (jDE)
            # With probability 0.1, assign new random values to F and CR.
            # F is sampled from [0.1, 1.0], CR from [0.0, 1.0].
            if np.random.rand() < 0.1:
                F[i] = 0.1 + 0.9 * np.random.rand()
            if np.random.rand() < 0.1:
                CR[i] = np.random.rand()
            
            # 2. Mutation (DE/rand/1)
            # Select 3 distinct random indices r1, r2, r3, none equal to i.
            # A simple retry loop is faster than np.random.choice for small pop_size.
            while True:
                r1 = np.random.randint(0, pop_size)
                r2 = np.random.randint(0, pop_size)
                r3 = np.random.randint(0, pop_size)
                if r1 != i and r2 != i and r3 != i and r1 != r2 and r1 != r3 and r2 != r3:
                    break
            
            # Create mutant vector
            mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])
            
            # 3. Crossover (Binomial)
            # Create trial vector by mixing mutant and target
            mask = np.random.rand(dim) < CR[i]
            # Ensure at least one dimension is taken from mutant
            j_rand = np.random.randint(dim)
            mask[j_rand] = True
            
            trial = np.where(mask, mutant, pop[i])
            
            # 4. Boundary Handling
            trial = np.clip(trial, min_b, max_b)
            
            # 5. Selection (Greedy & Asynchronous)
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                fitness[i] = f_trial
                pop[i] = trial
                
                # Update global best immediately
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial.copy()
    
    return best_val
