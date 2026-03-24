#Here is a fully operational, self-contained implementation using a **Self-Adaptive Differential Evolution (JADE variant) with Restart**.
#
#### Improvements:
#1.  **Adaptive Parameters (JADE Strategy):** Instead of fixed or randomly dithered parameters, this algorithm "learns" the best mutation factor ($F$) and crossover rate ($CR$) over time. It maintains averages ($\mu_F, \mu_{CR}$) and updates them based on successful mutations that improved fitness.
#2.  **Current-to-pBest Mutation:** It directs agents towards the top $p\%$ of the population (not just the single best). This balances convergence speed with population diversity better than the standard "current-to-best".
#3.  **Adaptive Restart:** Detects stagnation by monitoring fitness variance. If the population converges to a local optimum, it triggers a restart, keeping the best solution found so far (elite) to ensure monotonic improvement.
#4.  **Vectorized Operations:** Optimizes math operations using NumPy broadcasting for speed, allowing more generations within the time limit.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    # --- Configuration ---
    # Population size: Standard is 10*dim. 
    # Clipped between 20 and 75 to ensure speed on expensive functions while maintaining diversity.
    pop_size = int(np.clip(10 * dim, 20, 75))
    
    # Time Management
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=max_time)
    
    # Pre-process bounds
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Adaptive Parameters (JADE) ---
    mu_f = 0.5      # Initial mean for Mutation Factor
    mu_cr = 0.5     # Initial mean for Crossover Rate
    c_rate = 0.1    # Learning rate for parameter updates
    p_best = 0.05   # Top 5% used for current-to-pbest mutation

    # Helper: Check time
    def is_time_up():
        return datetime.now() >= end_time

    # --- Initialization: Latin Hypercube Sampling ---
    # Stratified sampling for better initial coverage of the search space
    population = np.empty((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(0, 1, pop_size + 1)
        offsets = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(offsets)
        population[:, d] = min_b[d] + offsets * diff_b[d]
        
    fitness = np.full(pop_size, float('inf'))
    
    # Tracking the Global Best
    best_val = float('inf')
    best_vec = None

    # Initial Evaluation
    for i in range(pop_size):
        if is_time_up(): return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            best_vec = population[i].copy()

    # --- Main Optimization Loop ---
    while not is_time_up():
        
        # 1. Restart Mechanism
        # If population diversity (std dev) is too low, we are likely stuck.
        # Restart population but keep the elite solution.
        std_fit = np.std(fitness)
        if std_fit < 1e-8 * (abs(best_val) + 1e-5):
            # Keep elite
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_vec
            fitness[:] = float('inf')
            fitness[0] = best_val
            
            # Reset adaptive memory to encourage exploration
            mu_f, mu_cr = 0.5, 0.5
            
            # Re-evaluate new random agents
            for i in range(1, pop_size):
                if is_time_up(): return best_val
                val = func(population[i])
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_vec = population[i].copy()
            continue

        # 2. Sort Population
        # Necessary for 'current-to-pbest' to find the top p%
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Ensure best_vec is synced (index 0 is best)
        best_vec = population[0].copy()
        best_val = fitness[0]

        # 3. Parameter Generation
        # CR_i ~ Normal(mu_cr, 0.1)
        cr_vals = np.random.normal(mu_cr, 0.1, pop_size)
        cr_vals = np.clip(cr_vals, 0, 1)
        
        # F_i ~ Cauchy(mu_f, 0.1)
        # Standard Cauchy (loc=0, scale=1) -> loc + scale * sample
        f_vals = mu_f + 0.1 * np.random.standard_cauchy(pop_size)
        
        # Constraint handling for F
        f_vals[f_vals > 1] = 1.0
        # For F <= 0, we can regenerate or clamp. Clamping to 0.1 keeps mutation active.
        f_vals[f_vals <= 0] = 0.1

        # 4. Mutation: DE/current-to-pbest/1
        # V = Xi + F * (X_pbest - Xi) + F * (X_r1 - X_r2)
        
        # Select X_pbest from top p%
        top_n = max(1, int(pop_size * p_best))
        pbest_idxs = np.random.randint(0, top_n, pop_size)
        x_pbest = population[pbest_idxs]
        
        # Select r1, r2 (random indices)
        # Note: Collision with i is possible but ignored for vectorization speed.
        r1 = np.random.randint(0, pop_size, pop_size)
        r2 = np.random.randint(0, pop_size, pop_size)
        
        # Compute Mutation Vectors
        # Reshape F for broadcasting (N, 1)
        F_col = f_vals[:, np.newaxis]
        
        # Vectorized mutation equation
        # current is population itself (sorted)
        mutants = population + F_col * (x_pbest - population) + F_col * (population[r1] - population[r2])
        
        # 5. Crossover (Binomial)
        rand_matrix = np.random.rand(pop_size, dim)
        cross_mask = rand_matrix < cr_vals[:, np.newaxis]
        
        # Ensure at least one dimension is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutants, population)
        
        # Boundary constraints
        trial_pop = np.clip(trial_pop, min_b, max_b)
        
        # 6. Selection and Parameter Adaptation
        successful_f = []
        successful_cr = []
        
        for i in range(pop_size):
            if is_time_up(): return best_val
            
            f_trial = func(trial_pop[i])
            
            # Greedy selection: if trial is better or equal, replace parent
            if f_trial < fitness[i]:
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                # Record successful parameters
                successful_f.append(f_vals[i])
                successful_cr.append(cr_vals[i])
                
                if f_trial < best_val:
                    best_val = f_trial
                    best_vec = trial_pop[i].copy()
        
        # 7. Update Adaptive Means
        if len(successful_f) > 0:
            sf = np.array(successful_f)
            scr = np.array(successful_cr)
            
            # Update mu_cr (Arithmetic Mean)
            mu_cr = (1 - c_rate) * mu_cr + c_rate * np.mean(scr)
            
            # Update mu_f (Lehmer Mean: sum(x^2)/sum(x))
            denom = np.sum(sf)
            if denom == 0: denom = 1e-10
            mean_lehmer = np.sum(sf**2) / denom
            mu_f = (1 - c_rate) * mu_f + c_rate * mean_lehmer

    return best_val
