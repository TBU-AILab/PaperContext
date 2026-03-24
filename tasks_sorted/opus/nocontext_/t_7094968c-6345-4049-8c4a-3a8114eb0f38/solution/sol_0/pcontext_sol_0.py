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
    pop_size = min(max(20, 10 * dim), 200)
    
    # Generate initial population
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[perm[j], i] = (perm[j] + np.random.uniform()) / pop_size
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
    
    # --- Phase 2: Differential Evolution with restarts and local search ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    last_best = best
    
    while True:
        elapsed = (datetime.now() - start).total_seconds()
        if elapsed >= max_time * 0.90:
            break
        
        generation += 1
        
        # Adaptive parameters
        F_gen = 0.5 + 0.3 * np.random.random()
        CR_gen = 0.8 + 0.2 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        # Sort population by fitness for current-to-best strategy
        sorted_idx = np.argsort(fitness)
        x_best_pop = population[sorted_idx[0]]
        
        for i in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.90:
                return best
            
            # Strategy selection
            strategy = np.random.randint(0, 3)
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            if strategy == 0:
                # DE/rand/1
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F_gen * (b - c)
            elif strategy == 1:
                # DE/best/1
                chosen = np.random.choice(idxs, 2, replace=False)
                b, c = population[chosen[0]], population[chosen[1]]
                mutant = x_best_pop + F_gen * (b - c)
            else:
                # DE/current-to-best/1
                chosen = np.random.choice(idxs, 2, replace=False)
                b, c = population[chosen[0]], population[chosen[1]]
                mutant = population[i] + F_gen * (x_best_pop - population[i]) + F_gen * (b - c)
            
            # Binomial crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR_gen or j == j_rand:
                    trial[j] = mutant[j]
            
            # Clip to bounds
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            if f_trial < new_fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if best < last_best - 1e-12:
            stagnation = 0
            last_best = best
        else:
            stagnation += 1
        
        # If stagnant, do partial restart keeping best individuals
        if stagnation > max(15, 50 // max(1, dim // 5)):
            stagnation = 0
            sorted_idx = np.argsort(fitness)
            keep = max(3, pop_size // 5)
            for i in range(keep, pop_size):
                population[i] = lower + np.random.uniform(0, 1, dim) * (upper - lower)
                # Some near best
                if np.random.random() < 0.3 and best_x is not None:
                    scale = 0.1 * (upper - lower) * np.random.random()
                    population[i] = best_x + np.random.randn(dim) * scale
                    population[i] = np.clip(population[i], lower, upper)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_x = population[i].copy()
    
    # --- Phase 3: Local search (Nelder-Mead style) around best ---
    if best_x is not None:
        # Simple coordinate descent / pattern search
        step = 0.01 * (upper - lower)
        
        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed >= max_time * 0.98:
                break
            
            improved = False
            for j in range(dim):
                if (datetime.now() - start).total_seconds() >= max_time * 0.98:
                    break
                
                # Try positive direction
                trial = best_x.copy()
                trial[j] = min(upper[j], best_x[j] + step[j])
                f_trial = func(trial)
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
                    improved = True
                    continue
                
                # Try negative direction
                trial = best_x.copy()
                trial[j] = max(lower[j], best_x[j] - step[j])
                f_trial = func(trial)
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
                    improved = True
                    continue
            
            if not improved:
                step *= 0.5
                if np.max(step / (upper - lower + 1e-30)) < 1e-12:
                    break
    
    return best
