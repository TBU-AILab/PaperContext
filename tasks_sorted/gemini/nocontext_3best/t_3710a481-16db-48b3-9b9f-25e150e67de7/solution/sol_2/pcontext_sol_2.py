#Based on the performance of the provided algorithms, the logic of **Self-Adaptive Differential Evolution (jDE)** proved superior (Result: 32.32) to standard DE (Result: 87.40).
#
#To improve this further, I propose upgrading the mutation strategy from `DE/rand/1` to **`DE/current-to-pbest/1`**. This is a strategy used in state-of-the-art algorithms like SHADE. It balances exploration (searching the whole space) and exploitation (converging towards the best found solutions) much more effectively.
#
#Additionally, I have added a **Local Search (Polishing)** phase. When the population converges (just before a restart), the algorithm performs a fine-grained Gaussian search around the best solution to "drain" the local minimum of any remaining potential before exploring new areas.
#
#Here is the improved algorithm:
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # Initialize timing
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Hyperparameters ---
    # Population size: Robust choice for adaptive DE
    pop_size = int(max(20, 10 * dim))
    
    # Top % to use for p-best mutation (0.05 to 0.20 is standard)
    p_best_rate = 0.10 
    
    # Pre-process bounds
    bounds_arr = np.array(bounds)
    min_b = bounds_arr[:, 0]
    max_b = bounds_arr[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Adaptive Parameters (jDE style)
    # F: Mutation factor, CR: Crossover rate
    # Each individual has its own F and CR
    F = np.full(pop_size, 0.5)
    CR = np.full(pop_size, 0.9)
    
    # Track Global Best
    global_best_val = float('inf')
    global_best_vec = np.zeros(dim)
    
    # Evaluate Initial Population
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return global_best_val
        
        val = func(pop[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_vec = pop[i].copy()

    # --- Main Loop ---
    while True:
        if datetime.now() - start_time >= time_limit:
            return global_best_val

        # 1. Parameter Adaptation (jDE logic)
        # -----------------------------------
        # Update F with prob 0.1
        mask_f = np.random.rand(pop_size) < 0.1
        F_new = F.copy()
        F_new[mask_f] = 0.1 + 0.9 * np.random.rand(np.sum(mask_f))
        
        # Update CR with prob 0.1
        mask_cr = np.random.rand(pop_size) < 0.1
        CR_new = CR.copy()
        CR_new[mask_cr] = np.random.rand(np.sum(mask_cr))

        # 2. Mutation: DE/current-to-pbest/1/bin
        # --------------------------------------
        # This strategy moves current vectors towards the best vectors in the population,
        # speeding up convergence compared to DE/rand/1.
        
        # Sort population by fitness to find p-best
        sorted_indices = np.argsort(fitness)
        num_p_best = max(1, int(pop_size * p_best_rate))
        top_p_indices = sorted_indices[:num_p_best]
        
        # Select p-best indices for each individual
        p_best_choices = np.random.choice(top_p_indices, pop_size)
        x_pbest = pop[p_best_choices]
        
        # Select r1 and r2 distinct from current i
        # We use random permutation approximation for speed
        r1_indices = np.random.randint(0, pop_size, pop_size)
        r2_indices = np.random.randint(0, pop_size, pop_size)
        
        # Calculate mutant vectors
        # V = X_current + F * (X_pbest - X_current) + F * (X_r1 - X_r2)
        mutant = pop + F_new[:, None] * (x_pbest - pop) + \
                 F_new[:, None] * (pop[r1_indices] - pop[r2_indices])

        # 3. Crossover
        # ------------
        rand_cross = np.random.rand(pop_size, dim)
        cross_mask = rand_cross < CR_new[:, None]
        
        # Ensure at least one dimension is inherited from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, pop)
        
        # 4. Bound Constraint Handling
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 5. Selection and Evaluation
        # ---------------------------
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            f_trial = func(trial_pop[i])
            
            if f_trial <= fitness[i]:
                pop[i] = trial_pop[i]
                fitness[i] = f_trial
                # Adopt new parameters on success
                F[i] = F_new[i]
                CR[i] = CR_new[i]
                
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trial_pop[i].copy()

        # 6. Restart Strategy with Local Polish
        # -------------------------------------
        # If population diversity is lost, we are likely in a local optimum.
        if not np.isinf(fitness).any():
            std_dev = np.std(fitness)
            # Convergence threshold
            if std_dev < 1e-6:
                
                # A. Local Polish (Fine-tuning)
                # Before destroying the population, try to improve the best solution
                # with a small local Gaussian search
                polish_attempts = 15  # Small number to not waste time
                polish_sigma = 1e-4   # Small step size
                
                current_polish_vec = global_best_vec.copy()
                
                for _ in range(polish_attempts):
                    if datetime.now() - start_time >= time_limit:
                        return global_best_val
                        
                    # Generate neighbor
                    noise = np.random.normal(0, polish_sigma, dim)
                    neighbor = np.clip(current_polish_vec + noise, min_b, max_b)
                    f_neighbor = func(neighbor)
                    
                    if f_neighbor < global_best_val:
                        global_best_val = f_neighbor
                        global_best_vec = neighbor
                        current_polish_vec = neighbor # Move center
                
                # B. Restart Population
                # Keep the best, randomize the rest
                pop = min_b + np.random.rand(pop_size, dim) * diff_b
                pop[0] = global_best_vec # Inject best found so far
                
                fitness = np.full(pop_size, float('inf'))
                fitness[0] = global_best_val
                
                # Reset Adaptive Parameters
                F = np.full(pop_size, 0.5)
                CR = np.full(pop_size, 0.9)

    return global_best_val
