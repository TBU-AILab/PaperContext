# Here is a fully operational, self-contained Python implementation of an improved optimization algorithm.
#
# This algorithm implements **JADE (Adaptive Differential Evolution) with a Restart Mechanism**.
# 
# Key Improvements over standard Differential Evolution:
# 1. **Self-Adaptation**: Automatically adjusts the crossover probability (CR) and mutation factor (F) based on successful generations, removing the need for manual tuning.
# 2. **Current-to-pBest Mutation**: Guides the search towards the top p% best solutions found so far, offering faster convergence than random mutation.
# 3. **Restart Strategy**: Detects population stagnation (convergence to a local optimum) and restarts the search while preserving the global best solution. This prevents getting stuck in local minima, utilizing the full `max_time` effectively.
# 4. **Strict Time Management**: Checks the timer before every function evaluation to strictly respect the `max_time` constraint.

import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    """
    Minimizes a function using JADE (Adaptive Differential Evolution) with Restart.
    """
    
    # --- 1. Setup & Time Management ---
    start_time = datetime.now()
    time_limit = timedelta(seconds=max_time)
    
    def check_timeout():
        return (datetime.now() - start_time) >= time_limit

    # --- 2. Configuration ---
    # Population size: 10 * dim is a robust heuristic, clamped to reasonable limits
    pop_size = int(np.clip(10 * dim, 20, 100))
    
    # JADE Hyperparameters
    p = 0.05        # Top percentage (p-best) to guide mutation
    c = 0.1         # Learning rate for parameter adaptation
    
    # Adaptive Memory (Initial values)
    mu_cr = 0.5     # Mean Crossover Probability
    mu_f = 0.5      # Mean Mutation Factor

    # --- 3. Initialization ---
    bounds_np = np.array(bounds)
    min_b = bounds_np[:, 0]
    max_b = bounds_np[:, 1]
    diff_b = max_b - min_b

    # Initialize population randomly
    pop = min_b + np.random.rand(pop_size, dim) * diff_b
    fitness = np.full(pop_size, float('inf'))
    
    best_fitness = float('inf')
    best_sol = None

    # Helper to re-initialize population (used at start and restarts)
    def init_population(keep_best=None):
        new_pop = min_b + np.random.rand(pop_size, dim) * diff_b
        new_fitness = np.full(pop_size, float('inf'))
        start_idx = 0
        
        # Elitism: Inject the best solution found so far into the new population
        if keep_best is not None:
            new_pop[0] = keep_best['vec']
            new_fitness[0] = keep_best['val']
            start_idx = 1
            
        return new_pop, new_fitness, start_idx

    pop, fitness, start_idx = init_population()

    # Initial Evaluation Loop
    for i in range(start_idx, pop_size):
        if check_timeout(): return best_fitness
        val = func(pop[i])
        fitness[i] = val
        if val < best_fitness:
            best_fitness = val
            best_sol = pop[i].copy()

    # --- 4. Main Optimization Loop ---
    while not check_timeout():
        
        # --- Restart Mechanism ---
        # Check if population has converged (variance is too low)
        # If converged, restart to explore other areas, but keep the global best.
        if np.std(fitness) < 1e-6 and (np.max(fitness) - np.min(fitness)) < 1e-6:
            # Save current best state for injection
            saved_best = {'vec': best_sol.copy(), 'val': best_fitness}
            
            # Re-initialize
            pop, fitness, start_idx = init_population(keep_best=saved_best)
            
            # Reset adaptive parameters to encourage exploration
            mu_cr = 0.5
            mu_f = 0.5
            
            # Evaluate new population
            for i in range(start_idx, pop_size):
                if check_timeout(): return best_fitness
                val = func(pop[i])
                fitness[i] = val
                if val < best_fitness:
                    best_fitness = val
                    best_sol = pop[i].copy()
            
            # Skip to next iteration to start evolution with new pop
            continue

        # --- Adaptive Parameter Generation ---
        # Generate CR_i ~ Normal(mu_cr, 0.1)
        cr = np.random.normal(mu_cr, 0.1, pop_size)
        cr = np.clip(cr, 0, 1)
        
        # Generate F_i ~ Cauchy(mu_f, 0.1)
        # standard_cauchy creates distr centered at 0. Scale by 0.1, shift by mu_f.
        f = np.random.standard_cauchy(pop_size) * 0.1 + mu_f
        
        # Truncate F values: if > 1 clamp to 1, if <= 0 clamp to 0.1 (robustness)
        f[f > 1] = 1.0
        f[f <= 0] = 0.1

        # --- Sorting for p-Best Selection ---
        sorted_indices = np.argsort(fitness)
        top_k = max(1, int(pop_size * p))
        top_indices = sorted_indices[:top_k]
        
        successful_cr = []
        successful_f = []

        # --- Evolution Cycle ---
        for i in range(pop_size):
            if check_timeout(): return best_fitness

            # 1. Mutation Strategy: DE/current-to-pbest/1
            # v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
            
            # Pick p-best
            pbest_idx = np.random.choice(top_indices)
            
            # Pick r1, r2 distinct from i
            candidates = [idx for idx in range(pop_size) if idx != i]
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            x_i = pop[i]
            x_pbest = pop[pbest_idx]
            x_r1 = pop[r1]
            x_r2 = pop[r2]
            
            F_i = f[i]
            CR_i = cr[i]
            
            # Compute Mutant
            mutant = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)
            
            # 2. Crossover (Binomial)
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR_i
            mask[j_rand] = True # Ensure at least one dimension changes
            
            trial = np.where(mask, mutant, x_i)
            
            # 3. Bound Constraint (Clamping)
            trial = np.clip(trial, min_b, max_b)
            
            # 4. Selection
            trial_fitness = func(trial)
            
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                pop[i] = trial
                successful_cr.append(CR_i)
                successful_f.append(F_i)
                
                # Update global best
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_sol = trial.copy()

        # --- Parameter Adaptation ---
        # Update mu_cr and mu_f based on successful individuals
        if len(successful_cr) > 0:
            # Update mu_cr (Arithmetic Mean)
            mu_cr = (1 - c) * mu_cr + c * np.mean(successful_cr)
            
            # Update mu_f (Lehmer Mean)
            sf = np.array(successful_f)
            mean_lehmer = np.sum(sf**2) / (np.sum(sf) + 1e-10)
            mu_f = (1 - c) * mu_f + c * mean_lehmer
            
    return best_fitness