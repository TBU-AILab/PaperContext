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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        val = func(x)
        if val < best:
            best = val
            best_x = x.copy()
        return val
    
    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(100, max(20, dim * 10))
    pop_size = min(60, max(15, dim * 5))
    
    population = np.zeros((n_init, dim))
    for i in range(dim):
        perm = np.random.permutation(n_init)
        for j in range(n_init):
            population[j, i] = lower[i] + (perm[j] + np.random.random()) / n_init * (upper[i] - lower[i])
    
    fitness = np.array([evaluate(population[j]) for j in range(n_init)])
    
    if elapsed() >= max_time:
        return best
    
    # Keep the best pop_size individuals
    indices = np.argsort(fitness)[:pop_size]
    population = population[indices].copy()
    fitness = fitness[indices].copy()
    
    # Phase 2: Differential Evolution with adaptive parameters
    F = 0.8
    CR = 0.9
    
    generation = 0
    while elapsed() < max_time * 0.85:
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            # Select three distinct individuals different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Adaptive F and CR
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # Mutation: DE/current-to-best/1
            best_idx = np.argmin(fitness)
            mutant = population[i] + Fi * (population[best_idx] - population[i]) + Fi * (population[r1] - population[r2])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_fitness = evaluate(trial)
            
            if trial_fitness <= fitness[i]:
                population[i] = trial.copy()
                fitness[i] = trial_fitness
        
        generation += 1
    
    # Phase 3: Local search (Nelder-Mead-like simplex) around the best solution
    if best_x is not None and elapsed() < max_time * 0.98:
        # Simple coordinate-wise local search
        x_current = best_x.copy()
        f_current = best
        
        step_sizes = (upper - lower) * 0.01
        
        while elapsed() < max_time * 0.98:
            improved = False
            for i in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                
                # Try positive step
                x_trial = x_current.copy()
                x_trial[i] = x_current[i] + step_sizes[i]
                f_trial = evaluate(x_trial)
                if f_trial < f_current:
                    x_current = x_trial.copy()
                    f_current = f_trial
                    improved = True
                    continue
                
                # Try negative step
                x_trial = x_current.copy()
                x_trial[i] = x_current[i] - step_sizes[i]
                f_trial = evaluate(x_trial)
                if f_trial < f_current:
                    x_current = x_trial.copy()
                    f_current = f_trial
                    improved = True
                    continue
            
            if not improved:
                step_sizes *= 0.5
                if np.max(step_sizes / (upper - lower + 1e-30)) < 1e-10:
                    break
    
    return best
