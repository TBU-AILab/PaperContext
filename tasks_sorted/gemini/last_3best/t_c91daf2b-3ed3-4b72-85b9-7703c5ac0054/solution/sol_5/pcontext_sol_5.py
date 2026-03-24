#The following algorithm improves upon the previous submissions by implementing a **Vectorized Differential Evolution (DE)** with a **Restart Mechanism**.
#
#### Key Improvements:
#1.  **Full Vectorization**: Unlike previous iterations that used Python loops for mutation and crossover, this implementation uses NumPy's array operations to process the entire population at once. This drastically reduces overhead, allowing for significantly more generations (and thus better convergence) within the limited `max_time`.
#2.  **Robust Strategy (DE/rand/1/bin)**: Based on the history, this strategy (score 11.2) outperformed greedy strategies. It is retained but enhanced with **Parameter Dithering**, where the mutation factor `F` varies per individual (0.5 to 1.0) to maintain diversity and adaptability.
#3.  **Smart Restart**: The algorithm monitors the population's standard deviation. If the population converges (`std < 1e-6`) and time permits, it triggers a hard restart to explore different basins of attraction, preventing stagnation in local optima.
#4.  **Distinct Index Selection**: A fast, vectorized collision-resolution mechanism ensures that mutation indices ($r_1, r_2, r_3$) are distinct from each other and the target index $i$, ensuring valid differential perturbations.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using Vectorized Differential Evolution (DE/rand/1/bin)
    with Dithering and Restart Mechanism.
    """
    start_time = time.time()
    
    # --- Pre-processing ---
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # --- Configuration ---
    # Population Size: 
    # 15 * dim is a robust balance between diversity and speed.
    # Clamped to [20, 100] to handle various dimensions efficiently.
    pop_size = int(np.clip(15 * dim, 20, 100))
    
    # Crossover Rate (CR):
    # 0.9 is generally effective for non-separable functions.
    CR = 0.9
    
    # Global best tracker
    best_val = float('inf')
    
    # Pre-allocated index array for vectorization
    idxs = np.arange(pop_size)
    
    # --- Main Loop (Restart Mechanism) ---
    while True:
        # Check remaining time before starting a new restart
        if time.time() - start_time >= max_time:
            return best_val
            
        # 1. Initialization
        # Random initialization within bounds
        pop = min_b + np.random.rand(pop_size, dim) * diff_b
        fitness = np.full(pop_size, float('inf'))
        
        # Evaluate initial population
        for i in range(pop_size):
            if time.time() - start_time >= max_time:
                return best_val
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
        
        # 2. Evolutionary Cycle
        while True:
            # Check time
            if time.time() - start_time >= max_time:
                return best_val
            
            # --- Vectorized Index Selection ---
            # Select r1, r2, r3 such that r1 != r2 != r3 != i
            # Using rejection sampling on vectors (very fast for these sizes)
            
            # Select r1 != i
            r1 = np.random.randint(0, pop_size, pop_size)
            collision = (r1 == idxs)
            while np.any(collision):
                r1[collision] = np.random.randint(0, pop_size, np.sum(collision))
                collision = (r1 == idxs)
            
            # Select r2 != r1 and r2 != i
            r2 = np.random.randint(0, pop_size, pop_size)
            collision = (r2 == idxs) | (r2 == r1)
            while np.any(collision):
                r2[collision] = np.random.randint(0, pop_size, np.sum(collision))
                collision = (r2 == idxs) | (r2 == r1)
                
            # Select r3 != r2 and r3 != r1 and r3 != i
            r3 = np.random.randint(0, pop_size, pop_size)
            collision = (r3 == idxs) | (r3 == r1) | (r3 == r2)
            while np.any(collision):
                r3[collision] = np.random.randint(0, pop_size, np.sum(collision))
                collision = (r3 == idxs) | (r3 == r1) | (r3 == r2)

            # --- Mutation (DE/rand/1) ---
            # Dither F per individual between 0.5 and 1.0 to diversify step sizes
            F = 0.5 + 0.5 * np.random.rand(pop_size, 1)
            
            # Vectorized mutation: V = X_r1 + F * (X_r2 - X_r3)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # --- Crossover (Binomial) ---
            # Create a boolean mask for crossover
            cross_mask = np.random.rand(pop_size, dim) < CR
            
            # Ensure at least one dimension is inherited from the mutant
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[idxs, j_rand] = True
            
            # Create trial population
            trial_pop = np.where(cross_mask, mutant, pop)
            
            # --- Boundary Handling ---
            trial_pop = np.clip(trial_pop, min_b, max_b)
            
            # --- Selection (Evaluation) ---
            # Evaluate trial solutions
            for i in range(pop_size):
                # Strict time check inside the loop
                if time.time() - start_time >= max_time:
                    return best_val
                
                f_trial = func(trial_pop[i])
                
                # Greedy selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial_pop[i]
                    
                    # Update global best
                    if f_trial < best_val:
                        best_val = f_trial
            
            # --- Convergence Check ---
            # If the population has converged (low standard deviation), 
            # break the inner loop to trigger a restart.
            if np.std(fitness) < 1e-6:
                break

    return best_val
