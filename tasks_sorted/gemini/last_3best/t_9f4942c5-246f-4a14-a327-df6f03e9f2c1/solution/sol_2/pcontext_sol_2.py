#The following algorithm implements **Vectorized JADE (Adaptive Differential Evolution) with Archive and Restart**.
#
##### Key Improvements:
#1.  **Vectorized Operations**: Unlike the loop-based approaches, this implementation vectorizes the mutation and crossover steps using NumPy. This significantly reduces the overhead of the Python interpreter, allowing for many more generations within the `max_time` limit.
#2.  **JADE Strategy (`current-to-pbest/1`)**: This mutation strategy balances exploration and exploitation better than `rand/1` (too slow) or `best/1` (prone to local optima). It guides the search towards the top $p\%$ of individuals.
#3.  **Adaptive Parameters**: The algorithm automatically adapts the crossover rate ($CR$) and mutation factor ($F$) based on the success of previous generations, removing the need for manual tuning.
#4.  **External Archive**: It maintains an archive of inferior solutions recently replaced. This preserves diversity in the mutation step, preventing the population from becoming too similar too quickly.
#5.  **Restart Mechanism**: If the population's fitness variance collapses (convergence), it triggers a soft restart, keeping the best solution and re-initializing the rest to escape local optima.
#
import numpy as np
import time

def run(func, dim, bounds, max_time):
    start_time = time.time()
    
    # --- Hyperparameters ---
    # Adjust population size based on dimension
    pop_size = 50
    if dim > 20:
        pop_size = 100
        
    archive_size = pop_size  # Capacity of the archive
    p_best_rate = 0.1        # Top 10% used for pbest
    c = 0.1                  # Learning rate for parameter adaptation
    
    # Adaptive parameter initialization
    mu_cr = 0.5
    mu_f = 0.5
    
    # --- Pre-processing ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b
    
    # --- Initialization ---
    population = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, np.inf)
    
    # Global best tracking
    best_val = np.inf
    best_sol = None
    
    # Archive initialization (stores historic vectors to maintain diversity)
    archive = np.zeros((archive_size, dim))
    n_archive = 0
    
    # Initial Evaluation
    for i in range(pop_size):
        if (time.time() - start_time) >= max_time:
            return best_val
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_val:
            best_val = val
            best_sol = population[i].copy()
            
    # --- Main Optimization Loop ---
    while True:
        # Check time at start of generation
        if (time.time() - start_time) >= max_time:
            return best_val
            
        # 1. Parameter Adaptation (Vectorized)
        # Generate CR_i ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0.0, 1.0)
        
        # Generate F_i ~ Cauchy(mu_f, 0.1) -> Approximated by Normal + clipping
        f = np.random.normal(mu_f, 0.1, pop_size)
        f = np.clip(f, 0.1, 1.0) # Clip to valid range avoiding 0
        
        # 2. Mutation: DE/current-to-pbest/1 with Archive
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Identify p-best individuals
        sorted_indices = np.argsort(fitness)
        num_top = max(1, int(pop_size * p_best_rate))
        top_indices = sorted_indices[:num_top]
        
        # Select pbest for each individual
        pbest_indices = np.random.choice(top_indices, pop_size)
        x_pbest = population[pbest_indices]
        
        # Select r1 (distinct from current is ideal, but random is fast & sufficient)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_indices]
        
        # Select r2 from Union(Population, Archive)
        # Total pool size
        pool_size = pop_size + n_archive
        r2_indices = np.random.randint(0, pool_size, pop_size)
        
        # Construct x_r2 based on indices
        x_r2 = np.zeros((pop_size, dim))
        
        # Mask for indices belonging to current population
        mask_pop = r2_indices < pop_size
        if np.any(mask_pop):
            x_r2[mask_pop] = population[r2_indices[mask_pop]]
            
        # Mask for indices belonging to archive
        mask_arch = ~mask_pop
        if np.any(mask_arch):
            # Map global index to archive index
            arch_idx = r2_indices[mask_arch] - pop_size
            x_r2[mask_arch] = archive[arch_idx]
            
        # Calculate mutant vectors (Vectorized Math)
        # Formula: X + F*(X_pbest - X) + F*(X_r1 - X_r2)
        diff_pbest = x_pbest - population
        diff_r1_r2 = x_r1 - x_r2
        mutant = population + f[:, None] * diff_pbest + f[:, None] * diff_r1_r2
        
        # 3. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        
        # Ensure at least one parameter is changed
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial = np.where(cross_mask, mutant, population)
        
        # 4. Boundary Handling (Clip)
        trial = np.clip(trial, min_b, max_b)
        
        # 5. Selection and Archive Update
        successful_f = []
        successful_cr = []
        
        # Evaluation loop (func is likely not vectorized)
        for i in range(pop_size):
            if (time.time() - start_time) >= max_time:
                return best_val
            
            f_trial = func(trial[i])
            
            if f_trial <= fitness[i]:
                # Archive the replaced solution
                if n_archive < archive_size:
                    archive[n_archive] = population[i].copy()
                    n_archive += 1
                else:
                    # Replace random member if full
                    rand_idx = np.random.randint(0, archive_size)
                    archive[rand_idx] = population[i].copy()
                
                # Update population
                population[i] = trial[i]
                fitness[i] = f_trial
                
                # Store successful parameters
                successful_f.append(f[i])
                successful_cr.append(cr[i])
                
                # Update global best
                if f_trial < best_val:
                    best_val = f_trial
                    best_sol = trial[i].copy()
                    
        # 6. Adapt Parameters (Lehmer Mean for F, Arithmetic Mean for CR)
        if len(successful_f) > 0:
            sf = np.array(successful_f)
            scr = np.array(successful_cr)
            
            # Lehmer mean for F
            mean_lehmer = np.sum(sf**2) / np.sum(sf)
            mu_f = (1 - c) * mu_f + c * mean_lehmer
            
            # Arithmetic mean for CR
            mu_cr = (1 - c) * mu_cr + c * np.mean(scr)
            
        # 7. Restart Mechanism
        # If population has converged (low variance), restart to search new areas
        if np.std(fitness) < 1e-6:
            # Keep only the best solution (Elitism)
            b_idx = np.argmin(fitness)
            best_current = population[b_idx].copy()
            best_fit_current = fitness[b_idx]
            
            # Re-initialize population
            population = min_b + np.random.rand(pop_size, dim) * diff_b
            population[0] = best_current
            fitness[:] = np.inf
            fitness[0] = best_fit_current
            
            # Reset Archive and Parameters
            n_archive = 0
            mu_f = 0.5
            mu_cr = 0.5
            
            # Re-evaluate new individuals (skipping index 0)
            for k in range(1, pop_size):
                if (time.time() - start_time) >= max_time:
                    return best_val
                
                val = func(population[k])
                fitness[k] = val
                if val < best_val:
                    best_val = val
                    best_sol = population[k].copy()

    return best_val
