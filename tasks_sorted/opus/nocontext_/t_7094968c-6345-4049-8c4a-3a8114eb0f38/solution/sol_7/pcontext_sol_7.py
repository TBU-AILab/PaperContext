#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    ranges = upper - lower
    
    # --- Phase 1: CMA-ES inspired search ---
    # Initialize population
    pop_size = min(4 + int(3 * np.log(dim)), 50)
    if pop_size % 2 != 0:
        pop_size += 1
    
    # Multiple restarts with different strategies
    def time_left():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip_to_bounds(x):
        return np.clip(x, lower, upper)
    
    # --- Differential Evolution + Local Search Hybrid ---
    
    # DE parameters
    F = 0.8
    CR = 0.9
    de_pop_size = max(10, min(10 * dim, 100))
    
    # Initialize population with Latin Hypercube-like sampling
    population = np.random.uniform(0, 1, (de_pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(de_pop_size)
        population[:, i] = (perm + population[:, i]) / de_pop_size
    population = lower + population * ranges
    
    fitness = np.full(de_pop_size, float('inf'))
    
    for i in range(de_pop_size):
        if time_left() < 0.1:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    # Main DE loop
    while time_left() > 0.5:
        generation += 1
        
        # Adaptive parameters
        sorted_idx = np.argsort(fitness)
        
        improved = False
        for i in range(de_pop_size):
            if time_left() < 0.2:
                return best
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(de_pop_size))
            idxs.remove(i)
            
            best_idx = sorted_idx[0]
            a, b = np.random.choice(idxs, 2, replace=False)
            
            # Adaptive F
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            
            mutant = population[i] + Fi * (population[best_idx] - population[i]) + Fi * (population[a] - population[b])
            mutant = clip_to_bounds(mutant)
            
            # Binomial crossover
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            trial = clip_to_bounds(trial)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
                    improved = True
        
        if not improved:
            stagnation += 1
        else:
            stagnation = 0
        
        # If stagnant, do local search around best
        if stagnation >= 5 and time_left() > 1.0:
            stagnation = 0
            # Nelder-Mead-like local search around best
            center = best_params.copy()
            step = ranges * 0.01
            
            for _ in range(dim * 10):
                if time_left() < 0.3:
                    return best
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-30)
                scale = np.random.exponential(1.0)
                candidate = center + step * direction * scale
                candidate = clip_to_bounds(candidate)
                cand_fit = func(candidate)
                if cand_fit < best:
                    best = cand_fit
                    best_params = candidate.copy()
                    center = candidate.copy()
            
            # Reinject some diversity
            n_replace = de_pop_size // 4
            worst_idx = np.argsort(fitness)[-n_replace:]
            for idx in worst_idx:
                population[idx] = best_params + ranges * 0.1 * np.random.randn(dim)
                population[idx] = clip_to_bounds(population[idx])
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
    
    # Final local refinement with shrinking search
    if best_params is not None and time_left() > 0.1:
        step_size = ranges * 0.001
        while time_left() > 0.05:
            candidate = best_params + step_size * np.random.randn(dim)
            candidate = clip_to_bounds(candidate)
            cand_fit = func(candidate)
            if cand_fit < best:
                best = cand_fit
                best_params = candidate.copy()
    
    return best
