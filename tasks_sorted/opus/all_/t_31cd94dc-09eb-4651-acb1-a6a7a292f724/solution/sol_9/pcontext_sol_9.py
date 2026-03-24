#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def eval_func(x):
        nonlocal best, best_params
        x = clip(x)
        val = func(x)
        if val < best:
            best = val
            best_params = x.copy()
        return val
    
    best_params = None
    
    # Phase 1: Latin Hypercube-like initialization
    n_init = min(50 * dim, 500)
    pop_size = min(20 * dim, 200)
    
    # Initial sampling
    init_pop = np.random.uniform(0, 1, (n_init, dim))
    for i in range(dim):
        init_pop[:, i] = lower[i] + init_pop[:, i] * (upper[i] - lower[i])
    
    init_fitness = np.full(n_init, float('inf'))
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        init_fitness[i] = eval_func(init_pop[i])
    
    # Select best individuals for DE population
    sorted_idx = np.argsort(init_fitness)
    pop = init_pop[sorted_idx[:pop_size]].copy()
    fitness = init_fitness[sorted_idx[:pop_size]].copy()
    
    # Phase 2: Differential Evolution with adaptive parameters
    F = 0.8
    CR = 0.9
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.85:
        generation += 1
        
        # Adaptive parameters
        if stagnation > 5:
            F = np.random.uniform(0.4, 1.0)
            CR = np.random.uniform(0.5, 1.0)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            jF = F * (1 + 0.1 * np.random.randn())
            
            mutant = pop[i] + jF * (best_params - pop[i]) + jF * (pop[r1] - pop[r2])
            mutant = clip(mutant)
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            trial = clip(trial)
            
            trial_fitness = eval_func(trial)
            
            if trial_fitness <= fitness[i]:
                pop[i] = trial.copy()
                fitness[i] = trial_fitness
        
        if best < prev_best - 1e-12:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        # Restart worst members if stagnating
        if stagnation > 15:
            n_replace = pop_size // 2
            worst_idx = np.argsort(fitness)[-n_replace:]
            for idx in worst_idx:
                pop[idx] = best_params + np.random.randn(dim) * (upper - lower) * 0.1
                pop[idx] = clip(pop[idx])
                fitness[idx] = eval_func(pop[idx])
                if elapsed() >= max_time * 0.85:
                    break
            stagnation = 0
    
    # Phase 3: Local refinement with Nelder-Mead-like simplex
    if best_params is not None:
        scale = (upper - lower) * 0.01
        while elapsed() < max_time * 0.95:
            # Random perturbation around best
            perturbation = np.random.randn(dim) * scale
            trial = clip(best_params + perturbation)
            trial_fitness = eval_func(trial)
            
            if trial_fitness < best:
                scale = scale * 1.1
            else:
                scale = scale * 0.95
                scale = np.maximum(scale, (upper - lower) * 1e-8)
    
    return best
