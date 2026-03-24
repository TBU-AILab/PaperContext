#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # --- Phase 1: Latin Hypercube Sampling for initial population ---
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate initial population using LHS-like sampling
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    
    for i in range(pop_size):
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.95):
            return best
        f = func(population[i])
        fitness[i] = f
        if f < best:
            best = f
            best_x = population[i].copy()
    
    # --- Phase 2: CMA-ES inspired search + Differential Evolution hybrid ---
    # Sort population
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # DE parameters
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    last_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.90):
            break
        
        generation += 1
        
        # Adaptive parameters
        F_cur = 0.5 + 0.3 * np.random.random()
        CR_cur = 0.8 + 0.2 * np.random.random()
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.90):
                return best
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best
            jitter = 0.001 * np.random.randn(dim)
            mutant = population[i] + F_cur * (population[0] - population[i]) + F_cur * (population[a] - population[b]) + jitter
            
            # Crossover
            cross_points = np.random.random(dim) < CR_cur
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            if f_trial < new_fit[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
            
            if f_trial < best:
                best = f_trial
                best_x = trial.copy()
        
        population = new_pop
        fitness = new_fit
        
        # Sort
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Check stagnation
        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1
        
        # If stagnating, restart worst half with perturbation around best
        if stagnation > 10:
            stagnation = 0
            half = pop_size // 2
            for i in range(half, pop_size):
                scale = (upper - lower) * 0.1 * np.random.random()
                population[i] = best_x + scale * np.random.randn(dim)
                population[i] = np.clip(population[i], lower, upper)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_x = population[i].copy()
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.90):
                    return best
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
    
    # --- Phase 3: Local refinement with Nelder-Mead style simplex ---
    if best_x is not None:
        # Simple coordinate descent refinement
        step = (upper - lower) * 0.01
        improved = True
        while improved:
            improved = False
            for i in range(dim):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                    return best
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[i] = trial[i] + direction * step[i]
                    trial = np.clip(trial, lower, upper)
                    f_trial = func(trial)
                    if f_trial < best:
                        best = f_trial
                        best_x = trial.copy()
                        improved = True
        
        # Reduce step and try again
        step *= 0.1
        improved = True
        while improved:
            improved = False
            for i in range(dim):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.99):
                    return best
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[i] = trial[i] + direction * step[i]
                    trial = np.clip(trial, lower, upper)
                    f_trial = func(trial)
                    if f_trial < best:
                        best = f_trial
                        best_x = trial.copy()
                        improved = True
    
    return best
