#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    # --- Parameters ---
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8       # differential weight
    CR = 0.9      # crossover probability
    
    # --- Initialize population ---
    population = np.random.uniform(0, 1, (pop_size, dim)) * ranges + lower
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_pos = population[best_idx].copy()
    
    if best_val < best:
        best = best_val
    
    # --- CMA-ES-like restart tracking ---
    stagnation_count = 0
    last_best = best
    
    generation = 0
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        generation += 1
        
        # Adaptive parameters
        # Use jittered F and CR for diversity
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # Mutation: DE/current-to-best/1 with jitter
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            # Select 2 random distinct indices different from i and best_idx
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # DE/current-to-best/1
            mutant = population[i] + Fi * (best_pos - population[i]) + Fi * (population[r1] - population[r2])
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bound handling: bounce back
            for d in range(dim):
                if trial[d] < lower[d] or trial[d] > upper[d]:
                    trial[d] = lower[d] + np.random.rand() * ranges[d]
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_pos = trial.copy()
        
        # Local search around best (Nelder-Mead style simplex or pattern search)
        if generation % 5 == 0:
            # Do a local search around best_pos
            local_best = best_pos.copy()
            local_best_val = best
            step = ranges * 0.01
            
            for _ in range(dim * 2):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                
                # Random perturbation (coordinate-wise)
                d = np.random.randint(dim)
                trial = local_best.copy()
                delta = step[d] * np.random.randn()
                trial[d] += delta
                trial[d] = np.clip(trial[d], lower[d], upper[d])
                
                trial_fitness = func(trial)
                if trial_fitness < local_best_val:
                    local_best = trial
                    local_best_val = trial_fitness
            
            if local_best_val < best:
                best = local_best_val
                best_pos = local_best.copy()
                # Replace worst in population
                worst_idx = np.argmax(fitness)
                population[worst_idx] = local_best
                fitness[worst_idx] = local_best_val
        
        # Check stagnation and restart partially
        if generation % 20 == 0:
            if abs(best - last_best) < 1e-12:
                stagnation_count += 1
            else:
                stagnation_count = 0
                last_best = best
            
            if stagnation_count >= 3:
                # Restart half the population randomly, keep best
                n_restart = pop_size // 2
                indices = np.argsort(fitness)
                for j in range(n_restart):
                    idx = indices[-(j+1)]
                    population[idx] = lower + np.random.rand(dim) * ranges
                    fitness[idx] = func(population[idx])
                    if fitness[idx] < best:
                        best = fitness[idx]
                        best_pos = population[idx].copy()
                stagnation_count = 0
                last_best = best
    
    return best
