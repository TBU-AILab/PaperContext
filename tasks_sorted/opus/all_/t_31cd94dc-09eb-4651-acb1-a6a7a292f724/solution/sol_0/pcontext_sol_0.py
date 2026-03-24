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
            best_params = population[i].copy()
    
    # --- Phase 2: Differential Evolution with restarts and local search ---
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.85):
            break
        
        generation += 1
        improved = False
        
        # Adaptive parameters
        F_gen = 0.5 + 0.3 * np.random.random()
        CR_gen = 0.8 + 0.2 * np.random.random()
        
        for i in range(pop_size):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.85):
                break
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            best_idx = np.argmin(fitness)
            
            # Mutation: current-to-best/1
            F_i = F_gen * (0.8 + 0.4 * np.random.random())
            mutant = population[i] + F_i * (population[best_idx] - population[i]) + F_i * (population[a] - population[b])
            
            # Crossover
            cross_points = np.random.random(dim) < CR_gen
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # Bounce back boundary handling
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = lower[d] + np.random.random() * (population[i][d] - lower[d])
                if trial[d] > upper[d]:
                    trial[d] = upper[d] - np.random.random() * (upper[d] - population[i][d])
            trial = np.clip(trial, lower, upper)
            
            f_trial = func(trial)
            
            if f_trial <= fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                if f_trial < best:
                    best = f_trial
                    best_params = trial.copy()
                    improved = True
        
        if not improved:
            stagnation += 1
        else:
            stagnation = 0
        
        # Restart worst half if stagnating
        if stagnation > 15:
            sorted_idx = np.argsort(fitness)
            half = pop_size // 2
            for idx in sorted_idx[half:]:
                population[idx] = lower + np.random.random(dim) * (upper - lower)
                # Bias some towards best
                if np.random.random() < 0.3:
                    population[idx] = best_params + 0.1 * (upper - lower) * np.random.randn(dim)
                    population[idx] = np.clip(population[idx], lower, upper)
                fitness[idx] = func(population[idx])
                if fitness[idx] < best:
                    best = fitness[idx]
                    best_params = population[idx].copy()
            stagnation = 0
    
    # --- Phase 3: Nelder-Mead local search around best solution ---
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time * 0.98):
            break
        
        # Simple coordinate-wise local search
        step = 0.01 * (upper - lower)
        for d in range(dim):
            passed_time = (datetime.now() - start)
            if passed_time >= timedelta(seconds=max_time * 0.98):
                break
            
            trial = best_params.copy()
            trial[d] = min(best_params[d] + step[d], upper[d])
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_params = trial.copy()
                continue
            
            trial = best_params.copy()
            trial[d] = max(best_params[d] - step[d], lower[d])
            f_trial = func(trial)
            if f_trial < best:
                best = f_trial
                best_params = trial.copy()
    
    return best
