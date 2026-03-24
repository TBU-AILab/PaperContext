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
    
    # --- Phase 2: CMA-ES inspired + Differential Evolution hybrid ---
    # Sort population
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
        if elapsed >= timedelta(seconds=max_time * 0.92):
            break
        
        generation += 1
        
        # Adaptive parameters
        F_adapt = 0.5 + 0.3 * np.random.random()
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.92):
                return best
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best with random component
            jitter = 0.001 * np.random.randn(dim)
            mutant = population[i] + F_adapt * (population[0] - population[i]) + F_adapt * (population[r1] - population[r2]) + jitter
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
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
        
        # Sort
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Check stagnation
        if best >= prev_best - 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        # Restart worst half if stagnated
        if stagnation > 15:
            half = pop_size // 2
            for i in range(half, pop_size):
                # Reinitialize around best with some randomness
                if np.random.random() < 0.5:
                    sigma = 0.1 * (upper - lower) * (0.5 ** (generation / 50.0))
                    population[i] = best_x + sigma * np.random.randn(dim)
                else:
                    population[i] = np.array([np.random.uniform(l, u) for l, u in bounds])
                population[i] = np.clip(population[i], lower, upper)
                fitness[i] = func(population[i])
                if fitness[i] < best:
                    best = fitness[i]
                    best_x = population[i].copy()
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            stagnation = 0
    
    # --- Phase 3: Local refinement with Nelder-Mead style simplex ---
    if best_x is not None:
        # Simple coordinate descent refinement
        step = 0.01 * (upper - lower)
        improved = True
        while improved:
            if (datetime.now() - start) >= timedelta(seconds=max_time * 0.99):
                break
            improved = False
            for j in range(dim):
                if (datetime.now() - start) >= timedelta(seconds=max_time * 0.99):
                    break
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[j] = trial[j] + direction * step[j]
                    trial = np.clip(trial, lower, upper)
                    f_trial = func(trial)
                    if f_trial < best:
                        best = f_trial
                        best_x = trial.copy()
                        improved = True
    
    return best
