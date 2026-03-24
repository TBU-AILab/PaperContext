#Here is a fully operational, self-contained implementation using a **Self-Adaptive Differential Evolution (jDE) with Current-to-pBest Strategy and Restarts**.
#
#### Improvements over the previous algorithm:
#1.  **Current-to-pBest Mutation**: Instead of targeting the single best solution (which causes premature convergence), this strategy targets a random solution from the top `p%` (e.g., top 10%). This balances fast convergence with genetic diversity.
#2.  **Self-Adaptation (jDE)**: Mutation factor (`F`) and Crossover rate (`CR`) are not just randomized but **adapted** per individual. Successful parameters are inherited by the next generation, allowing the algorithm to "learn" the best search settings for the specific function landscape.
#3.  **Vectorized Operations**: Mutation and crossover trials are generated using Numpy vectorization for the entire population at once, significantly reducing interpretation overhead compared to looping inside Python.
#4.  **Robust Restart Mechanism**: The algorithm monitors both population diversity (standard deviation) and fitness stagnation. If the search stalls or converges to a local optimum, it triggers a full restart (keeping the global best safe) to explore new areas of the search space.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes func using Self-Adaptive Differential Evolution (jDE) 
    with Current-to-pBest mutation and Restart mechanism.
    """
    # --- Configuration ---
    start_time = datetime.now()
    # Use 98% of available time to ensure we return safely before timeout
    time_limit = timedelta(seconds=max_time * 0.98) 

    # Population size: Adaptive based on dimension
    # 10x-15x dim is standard, clamped to [20, 100] for speed/efficiency trade-off
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # jDE Control Parameter Adaptation Settings
    tau_F = 0.1   # Probability to update F
    tau_CR = 0.1  # Probability to update CR
    
    # Pre-process bounds for vectorized math
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    global_best_val = float('inf')

    # Helper: Check if time is almost up
    def is_time_up():
        return (datetime.now() - start_time) >= time_limit

    # --- Main Optimization Loop (Outer Loop for Restarts) ---
    while not is_time_up():
        
        # 1. Initialization: Latin Hypercube Sampling (LHS)
        # Stratified sampling ensures better initial coverage than pure random
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            # Create strata edges
            edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
            # Sample uniformly within strata
            samp = np.random.uniform(edges[:-1], edges[1:])
            # Shuffle to mix dimensions
            np.random.shuffle(samp)
            pop[:, d] = samp
            
        # Initialize jDE parameters (F=0.5, CR=0.9 are standard starting points)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        fitness = np.full(pop_size, float('inf'))
        
        # Initial Evaluation
        for i in range(pop_size):
            if is_time_up(): return global_best_val
            val = func(pop[i])
            fitness[i] = val
            if val < global_best_val:
                global_best_val = val
        
        # --- Evolution Cycle (Inner Loop) ---
        stall_counter = 0
        gen_best = np.min(fitness)
        
        while not is_time_up():
            # 2. Parameter Adaptation (jDE)
            # Create trial parameters. If mask is True, generate new params, else keep old.
            rand_F = np.random.rand(pop_size)
            rand_CR = np.random.rand(pop_size)
            
            mask_F = np.random.rand(pop_size) < tau_F
            mask_CR = np.random.rand(pop_size) < tau_CR
            
            # Temporary arrays for trial generation
            F_trial = F.copy()
            CR_trial = CR.copy()
            
            # F updates to [0.1, 1.0], CR updates to [0.0, 1.0]
            F_trial[mask_F] = 0.1 + 0.9 * rand_F[mask_F]
            CR_trial[mask_CR] = rand_CR[mask_CR]
            
            # 3. Mutation Strategy: Current-to-pBest/1 (Vectorized Preparation)
            # Sort population by fitness to identify "p-best"
            sorted_idx = np.argsort(fitness)
            
            # Select p-best (top 10% of population, minimum 2 individuals)
            p_limit = max(2, int(pop_size * 0.1))
            top_p_idxs = sorted_idx[:p_limit]
            
            # Randomly select a pbest for each individual
            pbest_idxs = np.random.choice(top_p_idxs, pop_size)
            
            # Select r1, r2 distinct from i
            # (Fast approximate method with collision fix)
            r1 = np.random.randint(0, pop_size, pop_size)
            r2 = np.random.randint(0, pop_size, pop_size)
            
            # Fix collisions where r1==i, r2==i, or r1==r2
            for k in range(pop_size):
                while r1[k] == k: 
                    r1[k] = np.random.randint(0, pop_size)
                while r2[k] == k or r2[k] == r1[k]: 
                    r2[k] = np.random.randint(0, pop_size)
            
            # Vector math for mutation
            x_i = pop
            x_pbest = pop[pbest_idxs]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            F_col = F_trial[:, np.newaxis]
            
            # Mutation Equation: V = Xi + F*(Xpbest - Xi) + F*(Xr1 - Xr2)
            mutant = x_i + F_col * (x_pbest - x_i) + F_col * (x_r1 - x_r2)
            
            # Boundary Constraint (Clip)
            mutant = np.clip(mutant, min_b, max_b)
            
            # 4. Crossover (Binomial)
            CR_col = CR_trial[:, np.newaxis]
            cross_mask = np.random.rand(pop_size, dim) < CR_col
            # Forced primary parameter constraint (ensure at least 1 dim changes)
            j_rand = np.random.randint(0, dim, pop_size)
            cross_mask[np.arange(pop_size), j_rand] = True
            
            trial_pop = np.where(cross_mask, mutant, pop)
            
            # 5. Selection & Evaluation
            for i in range(pop_size):
                if is_time_up(): return global_best_val
                
                f_new = func(trial_pop[i])
                
                if f_new <= fitness[i]:
                    fitness[i] = f_new
                    pop[i] = trial_pop[i]
                    # Success: adopt the trial parameters for next gen
                    F[i] = F_trial[i]
                    CR[i] = CR_trial[i]
                    
                    if f_new < global_best_val:
                        global_best_val = f_new
            
            # 6. Restart Detection
            current_gen_best = np.min(fitness)
            # Check for improvement
            if current_gen_best < gen_best:
                gen_best = current_gen_best
                stall_counter = 0
            else:
                stall_counter += 1
                
            # Calculate population diversity (normalized avg std dev)
            std_metric = np.mean(np.std(pop, axis=0)) / np.mean(diff_b)
            
            # Trigger Restart if:
            # a) Population has converged (low diversity)
            # b) Best fitness hasn't improved for 30 generations (Stagnation)
            if std_metric < 1e-4 or stall_counter > 30:
                break 

    return global_best_val
