#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Parameters
    pop_size = min(max(20, 10 * dim), 100)
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability
    
    # Initialize population
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best = fitness[best_idx]
    best_params = population[best_idx].copy()
    
    # CMA-ES inspired restart + DE hybrid
    generation = 0
    stagnation_count = 0
    prev_best = best
    
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
            
            # DE/current-to-best/1/bin with jitter
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.0, 1.0)
            
            # Select 3 random individuals different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # DE/current-to-best/1
            mutant = population[i] + Fi * (best_params - population[i]) + Fi * (population[r1] - population[r2])
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
                # Final clip just in case
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        # Check stagnation
        if abs(prev_best - best) < 1e-12:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        # If stagnated, do a partial restart keeping best solutions
        if stagnation_count > 20:
            stagnation_count = 0
            # Keep top 20% and reinitialize rest
            sorted_idx = np.argsort(fitness)
            keep = max(2, pop_size // 5)
            for i in range(keep, pop_size):
                # Local search around best with decreasing radius
                elapsed_ratio = (datetime.now() - start).total_seconds() / max_time
                radius = max(0.01, 1.0 - elapsed_ratio)
                population[i] = best_params + radius * (upper - lower) * (np.random.rand(dim) - 0.5)
                population[i] = np.clip(population[i], lower, upper)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_params = population[i].copy()
        
        # Occasional local search around best
        if generation % 5 == 0:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            elapsed_ratio = passed_time.total_seconds() / max_time
            sigma = max(0.001, 0.1 * (1 - elapsed_ratio)) * (upper - lower)
            
            for _ in range(dim):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                local_trial = best_params + sigma * np.random.randn(dim)
                local_trial = np.clip(local_trial, lower, upper)
                local_fitness = func(local_trial)
                if local_fitness < best:
                    best = local_fitness
                    best_params = local_trial.copy()
    
    return best
