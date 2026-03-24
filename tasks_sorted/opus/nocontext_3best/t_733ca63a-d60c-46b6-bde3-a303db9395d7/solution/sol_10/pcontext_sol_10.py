#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    
    bounds = np.array(bounds)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    # Phase 1: Latin Hypercube-like initialization
    pop_size = min(100, max(20, dim * 10))
    
    # Initialize population
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            pop[perm[j], i] = (j + np.random.random()) / pop_size
    
    # Scale to bounds
    for i in range(pop_size):
        pop[i] = lower + pop[i] * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time:
            return best
        fitness[i] = func(pop[i])
        if fitness[i] < best:
            best = fitness[i]
            best_x = pop[i].copy()
    
    # Phase 2: Differential Evolution with adaptive parameters
    F = 0.5
    CR = 0.9
    
    generation = 0
    while True:
        if elapsed() >= max_time * 0.85:
            break
        
        # Sort population by fitness
        idx = np.argsort(fitness)
        pop = pop[idx]
        fitness = fitness[idx]
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                return best
            
            # Adaptive parameters
            Fi = F + 0.1 * np.random.randn()
            Fi = min(Fi, 1.0)
            Fi = max(Fi, 0.1)
            CRi = CR + 0.1 * np.random.randn()
            CRi = min(CRi, 1.0)
            CRi = max(CRi, 0.0)
            
            # Strategy: current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.randint(0, p)
            
            # Select 2 random indices different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - pop[r2])
            mutant = clip(mutant)
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CRi
            j_rand = np.random.randint(dim)
            cross_points[j_rand] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            trial = clip(trial)
            
            trial_fitness = func(trial)
            
            if trial_fitness <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = trial_fitness
                if trial_fitness < best:
                    best = trial_fitness
                    best_x = trial.copy()
        
        pop = new_pop
        fitness = new_fitness
        generation += 1
    
    # Phase 3: Local search around best solution using Nelder-Mead-like simplex
    if elapsed() < max_time * 0.95 and best < float('inf'):
        # Simple coordinate descent refinement
        step = (upper - lower) * 0.01
        improved = True
        while improved and elapsed() < max_time * 0.98:
            improved = False
            for d in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                for sign in [1, -1]:
                    candidate = best_x.copy()
                    candidate[d] = candidate[d] + sign * step[d]
                    candidate = clip(candidate)
                    val = func(candidate)
                    if val < best:
                        best = val
                        best_x = candidate.copy()
                        improved = True
            step *= 0.5
    
    return best
