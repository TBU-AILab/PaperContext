#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    def time_remaining():
        elapsed = (datetime.now() - start).total_seconds()
        return elapsed < max_time * 0.95
    
    # Phase 1: Latin Hypercube-like initial sampling
    pop_size = min(20 * dim, 200)
    
    # Initialize population
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        population[:, i] = lower[i] + population[:, i] * (upper[i] - lower[i])
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if not time_remaining():
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # Phase 2: Differential Evolution with restarts and local search
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while time_remaining():
        generation += 1
        
        # Sort population by fitness
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if not time_remaining():
                return best
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            
            mutant = population[i] + Fi * (population[0] - population[i]) + Fi * (population[a] - population[b])
            
            # Binomial crossover
            CRi = np.clip(CR + 0.1 * np.random.randn(), 0.1, 1.0)
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back bounds handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            trial_fit = func(trial)
            
            if trial_fit <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = trial_fit
                if trial_fit < best:
                    best = trial_fit
                    best_params = trial.copy()
        
        population = new_pop
        fitness = new_fit
        
        # Local search on best solution periodically
        if generation % 10 == 0 and best_params is not None and time_remaining():
            # Nelder-Mead-like local search
            step = (upper - lower) * 0.01
            for ls_iter in range(dim * 5):
                if not time_remaining():
                    return best
                direction = np.random.randn(dim)
                direction = direction / (np.linalg.norm(direction) + 1e-30)
                candidate = best_params + step * direction
                candidate = np.clip(candidate, lower, upper)
                cand_fit = func(candidate)
                if cand_fit < best:
                    best = cand_fit
                    best_params = candidate.copy()
                    step *= 1.2
                else:
                    step *= 0.8
        
        # Check stagnation
        if abs(best - prev_best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
            prev_best = best
        
        # Restart bottom half if stagnant
        if stagnation > 20:
            half = pop_size // 2
            for i in range(half, pop_size):
                population[i] = lower + np.random.rand(dim) * (upper - lower)
                if not time_remaining():
                    return best
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_params = population[i].copy()
            stagnation = 0
    
    return best
