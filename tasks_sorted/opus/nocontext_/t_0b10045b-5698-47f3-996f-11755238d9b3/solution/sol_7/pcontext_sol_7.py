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
    best_solution = population[best_idx].copy()
    
    # Track stagnation for restart
    stagnation_counter = 0
    last_best = best
    
    generation = 0
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            return best
        
        generation += 1
        
        # Adaptive parameters
        # Use jittered F and CR for diversity
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                return best
            
            # Mutation: DE/current-to-best/1 with jitter
            Fi = F + 0.1 * (np.random.random() - 0.5)
            CRi = CR + 0.1 * (np.random.random() - 0.5)
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # Select 2 random distinct individuals (not i, not best_idx)
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # DE/current-to-best/1
            mutant = population[i] + Fi * (best_solution - population[i]) + Fi * (population[r1] - population[r2])
            
            # Binomial crossover
            cross_points = np.random.random(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
                # Clamp as fallback
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best:
                    best = trial_fitness
                    best_solution = trial.copy()
        
        # Check stagnation
        if abs(best - last_best) < 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_best = best
        
        # Restart mechanism: keep best, reinitialize rest
        if stagnation_counter > 50:
            stagnation_counter = 0
            # Partial restart: reinitialize worst half around best
            sorted_indices = np.argsort(fitness)
            half = pop_size // 2
            for j in range(half, pop_size):
                idx = sorted_indices[j]
                # Reinitialize with shrinking radius around best
                radius = 0.3 * (upper - lower)
                population[idx] = best_solution + np.random.uniform(-radius, radius)
                population[idx] = np.clip(population[idx], lower, upper)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_solution = population[idx].copy()
        
        # Periodically do local search around best (Nelder-Mead style simplex step)
        if generation % 20 == 0:
            # Simple local search: try small perturbations
            for _ in range(dim):
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.98):
                    return best
                
                perturbation = np.random.normal(0, 0.01 * (upper - lower))
                candidate = np.clip(best_solution + perturbation, lower, upper)
                cand_fitness = func(candidate)
                if cand_fitness < best:
                    best = cand_fitness
                    best_solution = candidate.copy()
    
    return best
