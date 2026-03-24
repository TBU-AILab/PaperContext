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
    
    # Parameters
    pop_size = min(max(20, 10 * dim), 100)
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover probability
    
    # Initialize population using Latin Hypercube-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        population[:, i] = lower[i] + population[:, i] * (upper[i] - lower[i])
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # Main loop: hybrid DE with restarts and local search
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.95):
            return best
        
        generation += 1
        improved = False
        
        # Adaptive parameters
        # Use jittered F and CR per individual
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.95):
                return best
            
            # DE/current-to-best/1 with jitter
            Fi = F + 0.1 * (np.random.random() - 0.5)
            CRi = CR + 0.1 * (np.random.random() - 0.5)
            CRi = np.clip(CRi, 0.1, 1.0)
            
            # Select 3 distinct random indices different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            best_idx = np.argmin(fitness)
            
            # DE/current-to-best/1
            mutant = population[i] + Fi * (population[best_idx] - population[i]) + Fi * (population[r1] - population[r2])
            
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
                # Clip as safety
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
                    improved = True
        
        # Local search on best solution periodically
        if generation % 5 == 0 and best_params is not None:
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.90):
                # Do a quick Nelder-Mead style local search
                pass
            
            # Simple coordinate-wise local search
            step = 0.01 * (upper - lower)
            current = best_params.copy()
            current_fit = best
            for _ in range(3):
                for d in range(dim):
                    passed_time = (datetime.now() - start)
                    if passed_time >= timedelta(seconds=max_time * 0.95):
                        return best
                    for direction in [1, -1]:
                        candidate = current.copy()
                        candidate[d] = np.clip(candidate[d] + direction * step[d], lower[d], upper[d])
                        cf = func(candidate)
                        if cf < current_fit:
                            current = candidate
                            current_fit = cf
                step *= 0.5
            
            if current_fit < best:
                best = current_fit
                best_params = current.copy()
                # Update population
                worst_idx = np.argmax(fitness)
                population[worst_idx] = current.copy()
                fitness[worst_idx] = current_fit
        
        # Check stagnation and restart partially
        if not improved:
            stagnation += 1
        else:
            stagnation = 0
        
        if stagnation > 15:
            # Restart half the population
            n_restart = pop_size // 2
            worst_indices = np.argsort(fitness)[-n_restart:]
            for idx in worst_indices:
                population[idx] = lower + np.random.random(dim) * (upper - lower)
                passed_time = (datetime.now() - start)
                if passed_time >= timedelta(seconds=max_time * 0.95):
                    return best
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
            stagnation = 0
    
    return best
