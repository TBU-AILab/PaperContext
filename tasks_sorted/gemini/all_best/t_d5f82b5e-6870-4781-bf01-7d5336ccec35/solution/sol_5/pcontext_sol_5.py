#The output value is: 32.1154829103857
#
#The improved algorithm code is:
#The improved algorithm implements a **Self-Adaptive Differential Evolution (jDE)** with the **Current-to-pBest/1/bin** strategy.
#
#Key Improvements:
#1.  **Strategy: Current-to-pBest**: Instead of greedily following the single best individual (which can lead to local optima), individuals are attracted to a random member of the top p% (top 10%). This balances convergence speed with population diversity.
#2.  **Self-Adaptation (jDE)**: The control parameters `F` (Scaling Factor) and `CR` (Crossover Rate) are encoded into each individual and evolve. The algorithm "learns" the best mutation strength and crossover probability for the specific function landscape on the fly, rather than relying on fixed or dithered constants.
#3.  **Soft Restart**: Monitors population fitness variance. If the population collapses (variance -> 0), it keeps the elite half and re-initializes the worst half. This prevents total stagnation while preserving the progress made.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start_time = datetime.now()
    # Subtract a small buffer to ensure safe return
    time_limit = timedelta(seconds=max_time)
    
    # -----------------------------------------------------
    # Algorithm Configuration
    # -----------------------------------------------------
    # Population Size: Adaptive to dimension
    # Larger than standard DE/rand to support p-best selection logic
    pop_size = int(15 * dim)
    # Clamp population size for time efficiency
    if pop_size < 20: pop_size = 20
    if pop_size > 70: pop_size = 70
    
    # jDE Self-Adaptation Probabilities
    tau_F = 0.1
    tau_CR = 0.1
    
    # Current-to-pBest parameter (Top 10%)
    p_best_rate = 0.1 
    
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Initialize Population: Uniform random within bounds
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    # Initialize Parameter Arrays (jDE mechanism)
    # Each individual carries its own F and CR
    F = np.full(pop_size, 0.5) 
    CR = np.full(pop_size, 0.9)
    
    # Track Global Best
    global_best_val = float('inf')
    global_best_vec = np.zeros(dim)
    
    # -----------------------------------------------------
    # Initial Evaluation
    # -----------------------------------------------------
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return global_best_val
            
        val = func(population[i])
        fitness[i] = val
        
        if val < global_best_val:
            global_best_val = val
            global_best_vec = population[i].copy()

    # -----------------------------------------------------
    # Main Optimization Loop
    # -----------------------------------------------------
    while datetime.now() - start_time < time_limit:
        
        # 1. Sort population to facilitate p-best selection and restart logic
        sorted_indices = np.argsort(fitness)
        
        # Determine the pool of best individuals (p-best)
        # Ensure at least 2 individuals are in the pool for diversity
        num_p_best = max(2, int(pop_size * p_best_rate))
        top_p_indices = sorted_indices[:num_p_best]
        
        # 2. Iterate through population
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return global_best_val
            
            # --- jDE Parameter Adaptation ---
            # Create trial parameters F_new, CR_new
            # With probability tau, reset to random values, else keep memory
            if np.random.rand() < tau_F:
                F_new = 0.1 + 0.9 * np.random.rand()
            else:
                F_new = F[i]
            
            if np.random.rand() < tau_CR:
                CR_new = np.random.rand()
            else:
                CR_new = CR[i]
                
            # --- Mutation Strategy: DE/current-to-pbest/1 ---
            # Formula: V = Xi + F*(Xpbest - Xi) + F*(Xr1 - Xr2)
            
            # Select target Xpbest from top p%
            p_idx = top_p_indices[np.random.randint(0, num_p_best)]
            x_pbest = population[p_idx]
            
            # Select random r1, r2 distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
                
            r2 = np.random.randint(0, pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(0, pop_size)
            
            x_i = population[i]
            x_r1 = population[r1]
            x_r2 = population[r2]
            
            # Calculate Mutant Vector
            mutant = x_i + F_new * (x_pbest - x_i) + F_new * (x_r1 - x_r2)
            
            # --- Crossover: Binomial ---
            # Randomly mix mutant with existing vector
            cross_mask = np.random.rand(dim) < CR_new
            # Guarantee at least one parameter comes from mutant
            j_rand = np.random.randint(dim)
            cross_mask[j_rand] = True
            
            trial = np.where(cross_mask, mutant, x_i)
            
            # --- Bound Handling ---
            # Clip to specified bounds
            trial = np.clip(trial, min_b, max_b)
            
            # --- Evaluation ---
            f_trial = func(trial)
            
            # --- Selection (Greedy) ---
            if f_trial <= fitness[i]:
                # Trial is better or equal; replace individual
                fitness[i] = f_trial
                population[i] = trial
                
                # Update individual's control parameters (Learning)
                F[i] = F_new
                CR[i] = CR_new
                
                # Update Global Best
                if f_trial < global_best_val:
                    global_best_val = f_trial
                    global_best_vec = trial.copy()
        
        # 3. Soft Restart Mechanism (Anti-Stagnation)
        # If the population has converged (low standard deviation), 
        # re-inject diversity by re-initializing the worst 50%.
        if np.std(fitness) < 1e-8:
            # Indices of the worst 50%
            worst_indices = sorted_indices[pop_size//2:]
            
            for idx in worst_indices:
                if datetime.now() - start_time >= time_limit:
                    return global_best_val
                
                # Re-initialize random position
                new_vec = min_b + np.random.rand(dim) * diff_b
                population[idx] = new_vec
                
                val = func(new_vec)
                fitness[idx] = val
                
                # Reset parameters to defaults to encourage exploration
                F[idx] = 0.5
                CR[idx] = 0.9
                
                # Check if random re-init found something better
                if val < global_best_val:
                    global_best_val = val
                    global_best_vec = new_vec.copy()

    return global_best_val
