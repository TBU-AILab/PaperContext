#The output values and codes for the last generated algorithms are as follows:
#1. output value is: 14.848924371508517
#2. output value is: 30.274162611025417
#3. output value is: 120.07359399137354
#
#The best generated algorithm code was the JADE-like variant. The following algorithm improves upon it by implementing a full **JADE (Joint Adaptive Differential Evolution) with an External Archive**.
#
#**Key Improvements:**
#1.  **External Archive**: Unlike the previous version, this implementation maintains an archive of inferior solutions recently replaced by better ones. This preserves diversity and provides alternative difference vectors for mutation, significantly reducing the risk of premature convergence (getting stuck in local minima).
#2.  **Latin Hypercube Sampling (LHS)**: The population is initialized using LHS (stratified sampling) rather than simple random uniform. This guarantees better initial coverage of the search space.
#3.  **Robust Parameter Adaptation**: It uses the full JADE logic with **Cauchy distribution** for the mutation factor $F$. The heavy tail of the Cauchy distribution allows for occasional long jumps in the search space, which helps escape local basins of attraction.
#4.  **Refined Restart Mechanism**: A soft restart strategy is used that clears the archive and resets adaptive parameters upon stagnation, ensuring the algorithm can effectively explore new regions if it gets stuck.
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Optimizes a black-box function using JADE (Adaptive Differential Evolution)
    with an External Archive and Latin Hypercube Initialization.
    """
    # ---------------- Setup ----------------
    start_time = datetime.now()
    # Safety buffer: stop slightly before max_time to ensure return
    time_limit = timedelta(seconds=max_time * 0.98)
    
    # ---------------- Hyperparameters ----------------
    # Population size: Standard heuristic 15*dim, clipped for performance constraints
    pop_size = int(max(20, min(15 * dim, 100)))
    
    # JADE Settings
    p_best_rate = 0.10     # Top 10% guide the mutation (greediness)
    archive_rate = 1.0     # Archive size relative to population size
    c = 0.1                # Learning rate for parameter adaptation
    
    # Restart Threshold (Variance in fitness)
    restart_tol = 1e-9

    # ---------------- Initialization ----------------
    bounds = np.array(bounds)
    min_b = bounds[:, 0]
    max_b = bounds[:, 1]
    diff_b = max_b - min_b
    
    # 1. Latin Hypercube Sampling (LHS)
    # Stratified sampling: divide each dimension into pop_size intervals
    # and pick one random point per interval.
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        population[:, d] = (np.random.permutation(pop_size) + np.random.rand(pop_size)) / pop_size
    population = min_b + population * diff_b

    # Archive initialization (empty)
    max_archive_size = int(pop_size * archive_rate)
    archive = np.empty((0, dim))
    
    # Adaptive Parameters Init (mu_cr, mu_f)
    mu_cr = 0.5
    mu_f = 0.5
    
    # Initial Evaluation
    fitness = np.full(pop_size, float('inf'))
    best_fitness = float('inf')
    best_sol = np.zeros(dim)
    
    for i in range(pop_size):
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness if best_sol is not None else float('inf')
        
        val = func(population[i])
        fitness[i] = val
        
        if val < best_fitness:
            best_fitness = val
            best_sol = population[i].copy()
            
    # ---------------- Main Loop ----------------
    while True:
        # Check Time
        if (datetime.now() - start_time) >= time_limit:
            return best_fitness

        # 1. Restart Check (Stagnation Detection)
        # If population variance is negligible, we are stuck.
        if (np.max(fitness) - np.min(fitness)) < restart_tol:
            # Soft Restart: Keep the elite, re-LHS the rest, clear archive
            new_pop = np.zeros((pop_size, dim))
            for d in range(dim):
                new_pop[:, d] = (np.random.permutation(pop_size) + np.random.rand(pop_size)) / pop_size
            population = min_b + new_pop * diff_b
            
            # Preserve elite
            population[0] = best_sol.copy()
            
            # Reset Archive and Adaptive Means
            archive = np.empty((0, dim))
            mu_cr = 0.5
            mu_f = 0.5
            
            # Re-evaluate (skipping elite)
            fitness = np.full(pop_size, float('inf'))
            fitness[0] = best_fitness
            
            for i in range(1, pop_size):
                if (datetime.now() - start_time) >= time_limit: return best_fitness
                val = func(population[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = population[i].copy()
            continue

        # 2. Parameter Generation
        # Generate CR_i ~ Normal(mu_cr, 0.1), clipped [0, 1]
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F_i ~ Cauchy(mu_f, 0.1)
        # Cauchy helps escape local optima via larger steps.
        # Implementation: loc + scale * tan(pi * (rand - 0.5))
        f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        
        # F must be positive and <= 1. Regenerate non-positive values.
        while True:
            bad_mask = f <= 0
            if not np.any(bad_mask): break
            f[bad_mask] = mu_f + 0.1 * np.tan(np.pi * (np.random.rand(np.sum(bad_mask)) - 0.5))
        f = np.minimum(f, 1.0)
        
        # 3. Mutation: DE/current-to-pbest/1 with Archive
        # V = X + F * (X_pbest - X) + F * (X_r1 - X_r2)
        
        # Identify p-best
        sorted_indices = np.argsort(fitness)
        num_p = max(1, int(pop_size * p_best_rate))
        p_best_indices = np.random.choice(sorted_indices[:num_p], pop_size)
        x_pbest = population[p_best_indices]
        
        # Select r1 (from population)
        r1_indices = np.random.randint(0, pop_size, pop_size)
        x_r1 = population[r1_indices]
        
        # Select r2 (from Population U Archive)
        if len(archive) > 0:
            union_pop = np.vstack((population, archive))
        else:
            union_pop = population
            
        r2_indices = np.random.randint(0, len(union_pop), pop_size)
        x_r2 = union_pop[r2_indices]
        
        # Calculate Mutant Vector
        f_col = f[:, None] # Reshape for broadcasting
        mutant = population + f_col * (x_pbest - population) + f_col * (x_r1 - x_r2)
        
        # Bound Handling (Clip)
        mutant = np.clip(mutant, min_b, max_b)
        
        # 4. Crossover (Binomial)
        rand_vals = np.random.rand(pop_size, dim)
        cross_mask = rand_vals < cr[:, None]
        
        # Ensure at least one parameter is taken from mutant
        j_rand = np.random.randint(0, dim, pop_size)
        cross_mask[np.arange(pop_size), j_rand] = True
        
        trial_pop = np.where(cross_mask, mutant, population)
        
        # 5. Selection & Adaptation Update
        successful_cr = []
        successful_f = []
        new_archive_candidates = []
        
        for i in range(pop_size):
            if (datetime.now() - start_time) >= time_limit: return best_fitness
            
            f_trial = func(trial_pop[i])
            
            # Greedy Selection
            if f_trial <= fitness[i]:
                # Archive the parent solution being replaced
                new_archive_candidates.append(population[i].copy())
                
                # Record successful parameters
                successful_cr.append(cr[i])
                successful_f.append(f[i])
                
                # Update Population
                fitness[i] = f_trial
                population[i] = trial_pop[i]
                
                # Update Global Best
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_sol = trial_pop[i].copy()
        
        # 6. Update Archive
        if new_archive_candidates:
            candidates = np.array(new_archive_candidates)
            if len(archive) == 0:
                archive = candidates
            else:
                archive = np.vstack((archive, candidates))
            
            # Randomly remove elements if archive exceeds size limit
            if len(archive) > max_archive_size:
                indices = np.random.choice(len(archive), max_archive_size, replace=False)
                archive = archive[indices]
                
        # 7. Update Adaptive Parameters (Lehmer Mean)
        if len(successful_cr) > 0:
            scr = np.array(successful_cr)
            sf = np.array(successful_f)
            
            # Update mu_cr (Arithmetic Mean)
            mu_cr = (1 - c) * mu_cr + c * np.mean(scr)
            
            # Update mu_f (Lehmer Mean)
            # sum(f^2) / sum(f) - gives more weight to larger successful F values
            mean_lehmer = np.sum(sf**2) / np.sum(sf)
            mu_f = (1 - c) * mu_f + c * mean_lehmer

    return best_fitness
