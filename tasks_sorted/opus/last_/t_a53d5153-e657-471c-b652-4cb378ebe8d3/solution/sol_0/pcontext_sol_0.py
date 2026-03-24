#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initialization ---
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate initial population using LHS-like approach
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        fitness[i] = func(population[i])
        if fitness[i] < best:
            best = fitness[i]
            best_params = population[i].copy()
    
    # --- Phase 2: CMA-ES inspired + Differential Evolution hybrid ---
    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # DE parameters
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.90):
            break
        
        generation += 1
        
        # Adaptive parameters
        F_adapt = 0.5 + 0.3 * np.random.random()
        CR_adapt = 0.8 + 0.2 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.90):
                return best
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best
            jitter = 0.001 * np.random.randn(dim)
            mutant = population[i] + F_adapt * (population[0] - population[i]) + F_adapt * (population[r1] - population[r2]) + jitter
            
            # Crossover
            cross_points = np.random.rand(dim) < CR_adapt
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= new_fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_params = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Sort
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Check stagnation
        if abs(prev_best - best) < 1e-15:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst half if stagnant
        if stagnation > 20:
            half = pop_size // 2
            for i in range(half, pop_size):
                population[i] = lower + np.random.rand(dim) * (upper - lower)
                # Local perturbation around best for some
                if np.random.random() < 0.5:
                    scale = 0.1 * (upper - lower)
                    population[i] = best_params + scale * np.random.randn(dim)
                    population[i] = np.clip(population[i], lower, upper)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_params = population[i].copy()
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: Nelder-Mead local search around best ---
    while True:
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
            return best
        
        # Simple local search with shrinking radius
        scale = 0.01 * (upper - lower)
        candidate = best_params + scale * np.random.randn(dim)
        candidate = np.clip(candidate, lower, upper)
        f_val = func(candidate)
        if f_val < best:
            best = f_val
            best_params = candidate.copy()
    
    return best
