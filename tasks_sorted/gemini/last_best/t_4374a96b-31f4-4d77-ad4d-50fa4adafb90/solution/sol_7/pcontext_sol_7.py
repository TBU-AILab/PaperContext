#Here is an improved, fully operational, and self-contained algorithm.
#
#### Improvements & Logic
#The previous best-performing algorithm (DE/rand/1 with Restarts) achieved a score of **18.65**, while the DE/current-to-pbest/1 variant stagnated at **26.85**. This suggests that while exploration is key (DE/rand/1), the static parameters ($F$ and $CR$) of the "best" algorithm were likely suboptimal for the specific landscape.
#
#This new algorithm implements **jDE (Self-Adaptive Differential Evolution)**. Instead of fixed parameters, every individual carries its own $F$ and $CR$ values. These parameters evolve: successful values are propagated to the next generation, while unsuccessful ones are re-sampled. This allows the algorithm to automatically learn whether the function is separable (low $CR$) or non-separable (high $CR$), and the appropriate step size ($F$), leading to better convergence than the static DE/rand/1.
#
#It retains the **Latin Hypercube Sampling (LHS)** initialization and **Restarts** mechanism to robustly handle multimodal landscapes and escape local optima within the time limit.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    """
    Minimizes func using jDE (Self-Adaptive Differential Evolution) with Restarts.
    
    Key Features:
    1. jDE: Self-adapts control parameters F and CR for each individual.
    2. DE/rand/1/bin: Robust mutation strategy for exploration.
    3. Latin Hypercube Sampling: Stratified initialization.
    4. Restarts: Triggers when population converges to explore new basins.
    """
    
    # --- Time Management ---
    start_time = time.time()
    # Buffer to ensure return before timeout
    time_limit = max_time - 0.05
    
    # --- Setup ---
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]
    diff = ub - lb
    
    # Population Size
    # Adaptive to dimension but clamped to ensure speed.
    # 10*dim is standard; capped at 70 to allow many generations within time limit.
    pop_size = int(np.clip(10 * dim, 20, 70))
    
    # jDE Adaptation Probabilities
    tau_F = 0.1
    tau_CR = 0.1
    
    # Global Best Tracker
    best_val = float('inf')
    
    # --- Main Loop (Restarts) ---
    while True:
        # Check time before starting a new restart
        if time.time() - start_time > time_limit:
            return best_val
            
        # 1. Initialization: Latin Hypercube Sampling (LHS)
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            jitter = np.random.rand(pop_size)
            pop[:, d] = lb[d] + (perm + jitter) / pop_size * diff[d]
            
        # Evaluate Initial Population
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if time.time() - start_time > time_limit:
                return best_val
            
            val = func(pop[i])
            fitness[i] = val
            if val < best_val:
                best_val = val
                
        # 2. Initialize jDE Parameters
        # Start with standard DE assumptions: F=0.5, CR=0.9
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        # --- Evolutionary Cycle ---
        while True:
            # Time check
            if time.time() - start_time > time_limit:
                return best_val
            
            # Convergence Check (Restart criteria)
            # If population fitness variance is negligible, we are stuck.
            if np.ptp(fitness) < 1e-7:
                break
                
            # --- Parameter Adaptation (jDE) ---
            # Create trial parameters
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            F_trial = F.copy()
            CR_trial = CR.copy()
            
            # Update F: Uniform in [0.1, 1.0]
            if np.any(mask_F):
                F_trial[mask_F] = 0.1 + 0.9 * np.random.rand(np.sum(mask_F))
            
            # Update CR: Uniform in [0.0, 1.0]
            if np.any(mask_CR):
                CR_trial[mask_CR] = np.random.rand(np.sum(mask_CR))
            
            # --- Mutation (DE/rand/1) ---
            # Generate distinct indices r1, r2, r3 != i
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            r3 = np.random.randint(0, pop_size, pop_size)
            
            # Fix collisions using a fast loop
            for i in range(pop_size):
                while r1[i] == i:
                    r1[i] = np.random.randint(0, pop_size)
                while r2[i] == i or r2[i] == r1[i]:
                    r2[i] = np.random.randint(0, pop_size)
                while r3[i] == i or r3[i] == r1[i] or r3[i] == r2[i]:
                    r3[i] = np.random.randint(0, pop_size)
            
            # Calculate Mutants: V = X_r1 + F * (X_r2 - X_r3)
            # F_trial needs shape (N, 1) for broadcasting
            mutants = pop[r1] + F_trial[:, None] * (pop[r2] - pop[r3])
            
            # Bound Handling (Clip)
            mutants = np.clip(mutants, lb, ub)
            
            # --- Crossover (Binomial) ---
            rand_matrix = np.random.rand(pop_size, dim)
            cross_mask = rand_matrix < CR_trial[:, None]
            
            # Ensure at least one dimension comes from mutant
            j_rand = np.random.randint(0, dim, size=pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trials = np.where(cross_mask, mutants, pop)
            
            # --- Selection ---
            fitness_trials = np.zeros(pop_size)
            
            # Strict time check during evaluation loop
            for i in range(pop_size):
                if time.time() - start_time > time_limit:
                    return best_val
                
                f_t = func(trials[i])
                fitness_trials[i] = f_t
                
                if f_t < best_val:
                    best_val = f_t
            
            # Greedy Selection
            mask_better = fitness_trials <= fitness
            
            # Update Population
            pop[mask_better] = trials[mask_better]
            fitness[mask_better] = fitness_trials[mask_better]
            
            # Update Parameters: Successful F and CR are kept
            F[mask_better] = F_trial[mask_better]
            CR[mask_better] = CR_trial[mask_better]
            
    return best_val
