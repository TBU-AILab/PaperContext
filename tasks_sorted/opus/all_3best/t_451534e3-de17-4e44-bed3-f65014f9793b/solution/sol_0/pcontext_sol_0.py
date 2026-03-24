#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # ---- Parameters ----
    pop_size = min(max(20, 10 * dim), 200)
    F = 0.8       # differential weight
    CR = 0.9      # crossover probability
    
    # ---- Initialize population ----
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([func(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_sol = population[best_idx].copy()
    if best_val < best:
        best = best_val
    
    generation = 0
    
    # ---- CMA-ES-like restart tracking ----
    stagnation_count = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        generation += 1
        
        # Adaptive parameters
        # Jitter F and CR slightly
        Fs = np.clip(F + 0.1 * np.random.randn(pop_size), 0.1, 1.5)
        CRs = np.clip(CR + 0.1 * np.random.randn(pop_size), 0.0, 1.0)
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # DE/current-to-best/1/bin
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2 = np.random.choice(indices, 2, replace=False)
            
            mutant = population[i] + Fs[i] * (best_sol - population[i]) + Fs[i] * (population[r1] - population[r2])
            
            # Crossover
            cross_points = np.random.rand(dim) < CRs[i]
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
                # Clip as safety net
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_sol = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if abs(prev_best - best) < 1e-15:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best = best
        
        # Restart mechanism: reinitialize part of population if stagnant
        if stagnation_count > 50:
            # Keep best solution, reinitialize rest
            n_keep = max(2, pop_size // 5)
            sorted_idx = np.argsort(fitness)
            keep_idx = sorted_idx[:n_keep]
            
            new_pop = np.random.uniform(lower, upper, (pop_size, dim))
            new_fit = np.full(pop_size, float('inf'))
            
            for j, ki in enumerate(keep_idx):
                new_pop[j] = population[ki]
                new_fit[j] = fitness[ki]
            
            # Also add some local search around best
            for j in range(n_keep, min(n_keep + 5, pop_size)):
                perturbation = best_sol + 0.01 * (upper - lower) * np.random.randn(dim)
                perturbation = np.clip(perturbation, lower, upper)
                new_pop[j] = perturbation
            
            for j in range(pop_size):
                if new_fit[j] == float('inf'):
                    passed_time = (datetime.now() - start)
                    if passed_time >= timedelta(seconds=max_time * 0.95):
                        return best
                    new_fit[j] = func(new_pop[j])
                    if new_fit[j] < best:
                        best = new_fit[j]
                        best_sol = new_pop[j].copy()
            
            population = new_pop
            fitness = new_fit
            stagnation_count = 0
    
    return best
