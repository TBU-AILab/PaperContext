#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
    F = 0.8
    CR = 0.9
    
    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        remaining = max_time - elapsed
        if remaining < max_time * 0.02:
            break
        
        generation += 1
        
        # Adaptive F and CR
        if stagnation > 10:
            F = np.random.uniform(0.4, 1.0)
            CR = np.random.uniform(0.1, 1.0)
        
        for i in range(pop_size):
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.85:
                break
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Use best individual
            best_idx = 0  # population is sorted
            
            F_i = F + 0.1 * np.random.randn()
            F_i = np.clip(F_i, 0.1, 1.5)
            
            mutant = population[i] + F_i * (population[best_idx] - population[i]) + F_i * (population[r1] - population[r2])
            
            # Crossover
            CR_i = np.clip(CR + 0.1 * np.random.randn(), 0.0, 1.0)
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back if out of bounds
            for d in range(dim):
                if trial[d] < lower[d] or trial[d] > upper[d]:
                    trial[d] = lower[d] + np.random.rand() * (upper[d] - lower[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        # Re-sort population
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        # If stagnating, inject some random individuals
        if stagnation > 20:
            n_replace = pop_size // 3
            for j in range(pop_size - n_replace, pop_size):
                population[j] = lower + np.random.rand(dim) * (upper - lower)
                fitness[j] = func(population[j])
                if fitness[j] < best:
                    best = fitness[j]
                    best_params = population[j].copy()
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: Local search (Nelder-Mead-like) around best ---
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.95:
            break
        
        # Small perturbation around best
        scale = 0.01 * (upper - lower)
        candidate = best_params + scale * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        f_val = func(candidate)
        if f_val < best:
            best = f_val
            best_params = candidate.copy()
    
    return best
