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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
        return f, x
    
    # Phase 1: Latin Hypercube Sampling for initial population
    pop_size = min(20 * dim, 200)
    
    # Generate initial population
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = lower[i] + (perm[j] + np.random.random()) / pop_size * ranges[i]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i], population[i] = evaluate(population[i])
    
    # Phase 2: Differential Evolution with adaptive parameters
    F = 0.8
    CR = 0.9
    
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]
    
    generation = 0
    stagnation = 0
    prev_best = best_f
    
    while elapsed() < max_time * 0.85:
        generation += 1
        
        # Adaptive parameters
        if stagnation > 10:
            F = np.random.uniform(0.4, 1.0)
            CR = np.random.uniform(0.1, 1.0)
            stagnation = 0
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            # Select mutation strategy
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            strategy = np.random.randint(0, 3)
            
            if strategy == 0:
                # DE/best/1
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                Fi = F * (1 + 0.1 * np.random.randn())
                mutant = best_x + Fi * (a - b)
            elif strategy == 1:
                # DE/rand/1
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                Fi = F * (1 + 0.1 * np.random.randn())
                mutant = a + Fi * (b - c)
            else:
                # DE/current-to-best/1
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                Fi = F * (1 + 0.1 * np.random.randn())
                mutant = population[i] + Fi * (best_x - population[i]) + Fi * (a - b)
            
            # Crossover
            CRi = min(1.0, max(0.0, CR + 0.1 * np.random.randn()))
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            trial_f, trial = evaluate(trial)
            
            if trial_f <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_f
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial.copy()
                    best_idx = i
        
        if abs(prev_best - best_f) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best_f
    
    # Phase 3: Local search around best solution (Nelder-Mead style)
    step = ranges * 0.01
    current_x = best_x.copy()
    current_f = best_f
    
    while elapsed() < max_time * 0.95:
        improved = False
        for d in range(dim):
            if elapsed() >= max_time * 0.95:
                break
            for sign in [1, -1]:
                trial = current_x.copy()
                trial[d] += sign * step[d]
                trial_f, trial = evaluate(trial)
                if trial_f < current_f:
                    current_x = trial
                    current_f = trial_f
                    improved = True
                    break
        if not improved:
            step *= 0.5
            if np.max(step / ranges) < 1e-12:
                break
    
    return best
