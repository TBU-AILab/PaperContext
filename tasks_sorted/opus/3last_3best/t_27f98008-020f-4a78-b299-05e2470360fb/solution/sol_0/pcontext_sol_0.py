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
    pop_size = min(max(20 * dim, 50), 200)
    
    # Generate LHS-like initial population
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
    
    # --- Phase 2: Differential Evolution with adaptive parameters ---
    F = 0.8
    CR = 0.9
    
    # Sort population by fitness
    sort_idx = np.argsort(fitness)
    population = population[sort_idx]
    fitness = fitness[sort_idx]
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        elapsed = (datetime.now() - start)
        if elapsed >= timedelta(seconds=max_time * 0.92):
            break
        
        generation += 1
        
        # Adaptive F and CR
        if stagnation > 10:
            F = np.random.uniform(0.4, 1.0)
            CR = np.random.uniform(0.1, 1.0)
        
        improved = False
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.92):
                break
            
            # Strategy selection: mix of DE/rand/1, DE/best/1, DE/current-to-best/1
            strategy = np.random.randint(0, 3)
            
            # Select 3 distinct individuals different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r = np.random.choice(candidates, 3, replace=False)
            r1, r2, r3 = r[0], r[1], r[2]
            
            Fi = F * np.random.uniform(0.5, 1.5)
            
            if strategy == 0:
                # DE/rand/1
                mutant = population[r1] + Fi * (population[r2] - population[r3])
            elif strategy == 1:
                # DE/best/1
                mutant = population[0] + Fi * (population[r1] - population[r2])
            else:
                # DE/current-to-best/1
                mutant = population[i] + Fi * (population[0] - population[i]) + Fi * (population[r1] - population[r2])
            
            # Binomial crossover
            CRi = CR
            cross_points = np.random.rand(dim) < CRi
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.rand() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.rand() * (upper[d] - population[i][d])
                # Clamp as safety
                trial[d] = np.clip(trial[d], lower[d], upper[d])
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    improved = True
        
        if not improved:
            stagnation += 1
        else:
            stagnation = 0
        
        # Re-sort population
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Restart worst members if stagnation is high
        if stagnation > 20:
            n_restart = pop_size // 3
            for j in range(pop_size - n_restart, pop_size):
                population[j] = lower + np.random.rand(dim) * (upper - lower)
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.92):
                    break
                fitness[j] = func(population[j])
                if fitness[j] < best:
                    best = fitness[j]
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            stagnation = 0
    
    # --- Phase 3: Local search (Nelder-Mead style) around best solution ---
    best_x = population[0].copy()
    
    # Simple pattern search refinement
    step = (upper - lower) * 0.01
    while True:
        if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
            break
        improved_local = False
        for d in range(dim):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.98):
                break
            for sign in [1, -1]:
                trial = best_x.copy()
                trial[d] = np.clip(trial[d] + sign * step[d], lower[d], upper[d])
                trial_f = func(trial)
                if trial_f < best:
                    best = trial_f
                    best_x = trial.copy()
                    improved_local = True
                    break
        if not improved_local:
            step *= 0.5
            if np.max(step / (upper - lower + 1e-30)) < 1e-12:
                break
    
    return best
