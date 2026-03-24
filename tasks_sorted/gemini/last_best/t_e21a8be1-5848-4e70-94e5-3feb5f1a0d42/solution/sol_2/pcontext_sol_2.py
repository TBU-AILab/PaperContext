#The output value of the last generated algorithm is: 43.6522096413823
#
#The following algorithm implements **JADE (Adaptive Differential Evolution with Optional External Archive)**.
#
#**Improvements over previous version:**
#1.  **Mutation Strategy `current-to-pbest/1`**: Instead of blindly following the single best solution (which risks premature convergence), this strategy directs the search towards a random solution selected from the top $p\%$ (e.g., top 5%). This balances greediness with diversity.
#2.  **Self-Adaptive Parameters**: Instead of randomizing mutation factor $F$ and crossover $CR$, the algorithm adapts the mean values ($\mu_F, \mu_{CR}$) based on the parameters that successfully generated better offspring in the previous generation. This allows the algorithm to "learn" the optimal step sizes for the specific landscape.
#3.  **External Archive**: "Loser" parent vectors are stored in an archive rather than being discarded. These vectors are used in the mutation operator to define difference vectors, significantly increasing the diversity of directions available without increasing population size.
#4.  **Optimized Restart**: The restart mechanism detects convergence (low standard deviation) and resets the population while preserving the global best, clearing the archive to prevent bias from the previous local optimum.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using JADE (Adaptive Differential Evolution) with Archive and LHS.
    """
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    # --- Configuration ---
    # Population size: JADE works well with smaller populations than classic DE.
    # We constrain it to ensure enough generations fit within max_time.
    pop_size = int(max(20, min(100, 10 * dim)))
    
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # Adaptive Parameter Initialization
    mu_cr = 0.5    # Mean Crossover Rate
    mu_f = 0.5     # Mean Mutation Factor
    c_adapt = 0.1  # Learning rate for parameter adaptation
    p_greedy = 0.05 # Top p% fraction for current-to-pbest strategy
    
    # Archive to store inferior solutions (maintains diversity)
    archive = []
    
    # --- Initialization: Latin Hypercube Sampling (LHS) ---
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        edges = np.linspace(min_b[d], max_b[d], pop_size + 1)
        samples = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(samples)
        population[:, d] = samples
        
    fitness = np.full(pop_size, float('inf'))
    best_val = float('inf')
    
    # Initial Evaluation
    for i in range(pop_size):
        if datetime.now() - start_time >= time_limit:
            return best_val
        val = func(population[i])
        fitness[i] = val
        if val < best_val:
            best_val = val
            
    # --- Main Optimization Loop ---
    while True:
        if datetime.now() - start_time >= time_limit:
            return best_val
            
        # Sort population by fitness to identify top p%
        sorted_indices = np.argsort(fitness)
        
        # Lists to store successful parameters for adaptation
        good_crs = []
        good_fs = []
        
        # Next generation containers
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Iterate over each individual
        for i in range(pop_size):
            if datetime.now() - start_time >= time_limit:
                return best_val
                
            # --- 1. Parameter Generation ---
            # CR ~ Normal(mu_cr, 0.1), clipped [0, 1]
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            # F ~ Cauchy(mu_f, 0.1), clipped (0, 1]
            while True:
                f = mu_f + 0.1 * np.random.standard_cauchy()
                if f > 0:
                    if f > 1: f = 1.0
                    break
                    
            # --- 2. Mutation: current-to-pbest/1 with Archive ---
            # x_pbest: Randomly chosen from top p%
            top_cnt = max(1, int(pop_size * p_greedy))
            pbest_idx = sorted_indices[np.random.randint(0, top_cnt)]
            x_pbest = population[pbest_idx]
            
            # x_r1: Random distinct from i
            r1 = np.random.randint(0, pop_size)
            while r1 == i:
                r1 = np.random.randint(0, pop_size)
            x_r1 = population[r1]
            
            # x_r2: Random distinct from i and r1, selected from (Population U Archive)
            pop_len = pop_size
            arch_len = len(archive)
            total_len = pop_len + arch_len
            
            r2_idx = np.random.randint(0, total_len)
            # Make sure r2 is not i or r1 (if r2 is in current population)
            while (r2_idx < pop_len and (r2_idx == i or r2_idx == r1)):
                r2_idx = np.random.randint(0, total_len)
                
            if r2_idx < pop_len:
                x_r2 = population[r2_idx]
            else:
                x_r2 = archive[r2_idx - pop_len]
                
            # Mutation Equation: v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            mutant = population[i] + f * (x_pbest - population[i]) + f * (x_r1 - x_r2)
            
            # Boundary constraint handling (clamping)
            mutant = np.clip(mutant, min_b, max_b)
            
            # --- 3. Crossover (Binomial) ---
            cross_mask = np.random.rand(dim) < cr
            # Ensure at least one dimension is taken from mutant
            if not np.any(cross_mask):
                cross_mask[np.random.randint(dim)] = True
                
            trial = np.where(cross_mask, mutant, population[i])
            
            # --- 4. Selection ---
            f_trial = func(trial)
            
            if f_trial < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                
                # Record successful control parameters
                good_crs.append(cr)
                good_fs.append(f)
                
                # Add parent to archive (improves diversity for x_r2 selection)
                archive.append(population[i].copy())
                if len(archive) > pop_size:
                    # Remove random element if archive is full
                    archive.pop(np.random.randint(0, len(archive)))
                
                if f_trial < best_val:
                    best_val = f_trial
            
        # Update population arrays
        population = new_population
        fitness = new_fitness
        
        # --- 5. Parameter Adaptation ---
        if len(good_crs) > 0:
            # Update mu_cr using weighted arithmetic mean
            mu_cr = (1 - c_adapt) * mu_cr + c_adapt * np.mean(good_crs)
            
            # Update mu_f using Lehmer mean (gives more weight to larger F values)
            fs = np.array(good_fs)
            sum_fs = np.sum(fs)
            if sum_fs > 0:
                lehmer_mean = np.sum(fs**2) / sum_fs
                mu_f = (1 - c_adapt) * mu_f + c_adapt * lehmer_mean
                
        # --- 6. Restart Mechanism ---
        # If population converges (low std dev), restart to explore new basins
        if np.std(fitness) < 1e-6:
             # Preserve the single best individual
             best_curr_idx = np.argmin(fitness)
             best_indiv = population[best_curr_idx].copy()
             
             # Re-initialize the rest randomly
             population = min_b + np.random.rand(pop_size, dim) * diff_b
             population[0] = best_indiv
             
             fitness = np.full(pop_size, float('inf'))
             fitness[0] = best_val
             
             # Clear archive as it belongs to the old basin
             archive = []
             
             # Evaluate new population (skipping index 0)
             for k in range(1, pop_size):
                 if datetime.now() - start_time >= time_limit:
                    return best_val
                 val = func(population[k])
                 fitness[k] = val
                 if val < best_val:
                     best_val = val
