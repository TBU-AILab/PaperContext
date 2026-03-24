#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_x
        x_clipped = np.clip(x, lower, upper)
        f = func(x_clipped)
        if f < best:
            best = f
            best_x = x_clipped.copy()
        return f
    
    # Phase 1: Latin Hypercube-like initialization with DE
    pop_size = min(max(10 * dim, 40), 200)
    
    # Initialize population
    pop = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        pop[:, i] = lower[i] + pop[:, i] * (upper[i] - lower[i])
    
    fitness = np.array([evaluate(pop[i]) for i in range(pop_size)])
    
    if elapsed() >= max_time:
        return best
    
    # Phase 2: Differential Evolution with adaptive parameters
    F = 0.8
    CR = 0.9
    generation = 0
    
    while elapsed() < max_time * 0.85:
        # Sort population by fitness for current-to-best strategies
        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[0]
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            # Adaptive parameters
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            CRi = CR + 0.1 * np.random.randn()
            CRi = np.clip(CRi, 0.0, 1.0)
            
            # Strategy: current-to-pbest/1/bin
            p = max(2, int(0.1 * pop_size))
            pbest_idx = sorted_idx[np.random.randint(0, p)]
            
            # Select 2 distinct random indices != i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            mutant = pop[i] + Fi * (pop[pbest_idx] - pop[i]) + Fi * (pop[r1] - pop[r2])
            
            # Binomial crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CRi or j == j_rand:
                    trial[j] = mutant[j]
            
            # Bounce-back boundary handling
            for j in range(dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + np.random.rand() * (pop[i][j] - lower[j])
                if trial[j] > upper[j]:
                    trial[j] = upper[j] - np.random.rand() * (upper[j] - pop[i][j])
            
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            
            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
        
        generation += 1
        
        # Occasionally restart worst members
        if generation % 50 == 0 and pop_size > 10:
            worst_idx = sorted_idx[-pop_size // 4:]
            for idx in worst_idx:
                if elapsed() >= max_time * 0.85:
                    break
                pop[idx] = lower + np.random.rand(dim) * (upper - lower)
                fitness[idx] = evaluate(pop[idx])
    
    # Phase 3: Local search around best solution using Nelder-Mead-like simplex
    if best_x is not None and elapsed() < max_time * 0.98:
        # Simple local search: coordinate-wise refinement
        x_local = best_x.copy()
        step = (upper - lower) * 0.01
        
        while elapsed() < max_time * 0.98:
            improved = False
            for j in range(dim):
                if elapsed() >= max_time * 0.98:
                    break
                for direction in [1, -1]:
                    x_new = x_local.copy()
                    x_new[j] = x_local[j] + direction * step[j]
                    x_new = np.clip(x_new, lower, upper)
                    f_new = evaluate(x_new)
                    if f_new < evaluate(x_local):
                        x_local = x_new
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if np.max(step / (upper - lower + 1e-30)) < 1e-12:
                    break
    
    return best
