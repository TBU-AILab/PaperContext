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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        val = func(x)
        if val < best:
            best = val
            best_x = x.copy()
        return val
    
    # Phase 1: Latin Hypercube Sampling for initial population
    pop_size = min(50, max(20, dim * 5))
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * (upper[d] - lower[d])
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = eval_func(population[i])
    
    # Phase 2: Differential Evolution
    F = 0.8
    CR = 0.9
    
    de_budget = max_time * 0.6
    generation = 0
    while elapsed() < de_budget:
        generation += 1
        for i in range(pop_size):
            if elapsed() >= de_budget:
                break
            
            # Select three random distinct indices different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)
            
            # Mutation
            mutant = population[a] + F * (population[b] - population[c])
            mutant = clip(mutant)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_fitness = eval_func(trial)
            if trial_fitness <= fitness[i]:
                population[i] = trial.copy()
                fitness[i] = trial_fitness
    
    # Phase 3: Nelder-Mead simplex on best solution found
    if best_x is not None and elapsed() < max_time * 0.95:
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        
        scale = 0.05 * (upper - lower)
        for i in range(n):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1][i] += scale[i]
            simplex[i + 1] = clip(simplex[i + 1])
        
        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if elapsed() >= max_time * 0.95:
                return best
            f_simplex[i] = eval_func(simplex[i])
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while elapsed() < max_time * 0.95:
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[n]))
            fr = eval_func(xr)
            
            if fr < f_simplex[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[n] = xe
                    f_simplex[n] = fe
                else:
                    simplex[n] = xr
                    f_simplex[n] = fr
            elif fr < f_simplex[n - 1]:
                simplex[n] = xr
                f_simplex[n] = fr
            else:
                # Contraction
                if fr < f_simplex[n]:
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = eval_func(xc)
                    if fc <= fr:
                        simplex[n] = xc
                        f_simplex[n] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
                else:
                    xc = clip(centroid + rho * (simplex[n] - centroid))
                    fc = eval_func(xc)
                    if fc <= f_simplex[n]:
                        simplex[n] = xc
                        f_simplex[n] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            f_simplex[i] = eval_func(simplex[i])
    
    return best
