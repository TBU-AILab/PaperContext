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
    
    # Generate initial population via LHS-like sampling
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
    
    # --- Phase 2: Differential Evolution with restarts and local search ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
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
        
        # Sort by fitness for current-to-best strategies
        sorted_idx = np.argsort(fitness)
        x_best_pop = population[sorted_idx[0]]
        
        for i in range(pop_size):
            if (datetime.now() - start).total_seconds() >= max_time * 0.90:
                return best
            
            # DE/current-to-best/1/bin with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            # Mutation: current-to-best
            F_i = F_gen + 0.1 * np.random.randn()
            F_i = np.clip(F_i, 0.1, 1.5)
            
            mutant = population[i] + F_i * (x_best_pop - population[i]) + F_i * (population[a] - population[b])
            
            # Crossover
            cross_points = np.random.random(dim) < CR_gen
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce-back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_x = trial.copy()
        
        population = new_population
        fitness = new_fitness
        
        # Check stagnation
        if abs(best - prev_best) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
            prev_best = best
        
        # If stagnated, do local search around best and partial restart
        if stagnation > 15:
            stagnation = 0
            # Nelder-Mead-like local search: just do a simple coordinate descent
            if best_x is not None:
                local_x = best_x.copy()
                scale = 0.01 * (upper - lower)
                for _ in range(3):
                    for d in range(dim):
                        if (datetime.now() - start).total_seconds() >= max_time * 0.90:
                            return best
                        for direction in [1, -1]:
                            trial = local_x.copy()
                            trial[d] = trial[d] + direction * scale[d]
                            trial = np.clip(trial, lower, upper)
                            f_trial = func(trial)
                            if f_trial < best:
                                best = f_trial
                                best_x = trial.copy()
                                local_x = trial.copy()
                    scale *= 0.5
            
            # Partial restart: replace worst half
            sorted_idx = np.argsort(fitness)
            half = pop_size // 2
            for i in range(half, pop_size):
                idx = sorted_idx[i]
                # Generate near best or random
                if np.random.random() < 0.5:
                    population[idx] = best_x + 0.1 * (upper - lower) * (np.random.random(dim) - 0.5)
                    population[idx] = np.clip(population[idx], lower, upper)
                else:
                    population[idx] = lower + np.random.random(dim) * (upper - lower)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_x = population[idx].copy()
    
    # --- Phase 3: Final local refinement around best ---
    if best_x is not None:
        scale = 0.001 * (upper - lower)
        for iteration in range(10):
            if (datetime.now() - start).total_seconds() >= max_time * 0.98:
                break
            improved = False
            for d in range(dim):
                if (datetime.now() - start).total_seconds() >= max_time * 0.98:
                    break
                for direction in [1, -1]:
                    trial = best_x.copy()
                    trial[d] = trial[d] + direction * scale[d]
                    trial = np.clip(trial, lower, upper)
                    f_trial = func(trial)
                    if f_trial < best:
                        best = f_trial
                        best_x = trial.copy()
                        improved = True
            if not improved:
                scale *= 0.5
    
    return best
