#Here's a clean, self-contained implementation using CMA-ES-inspired search combined with differential evolution and local refinement:
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
    ranges = upper - lower
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # Phase 1: Latin Hypercube-style initial sampling
    pop_size = min(50, max(20, 10 * dim))
    population = np.zeros((pop_size, dim))
    fitness = np.zeros(pop_size)
    
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        population[i] = lower + np.random.rand(dim) * ranges
        fitness[i] = evaluate(population[i])
    
    # Phase 2: Differential Evolution
    F = 0.8
    CR = 0.9
    
    generation = 0
    while elapsed() < max_time * 0.7:
        for i in range(pop_size):
            if elapsed() >= max_time * 0.7:
                break
            
            # Select three distinct random indices different from i
            idxs = list(range(pop_size))
            idxs.remove(i)
            chosen = np.random.choice(idxs, 3, replace=False)
            a, b, c = chosen
            
            # Mutation
            mutant = population[a] + F * (population[b] - population[c])
            mutant = clip(mutant)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_fitness = evaluate(trial)
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
        
        generation += 1
        
        # Adaptive parameters
        F = 0.5 + 0.3 * np.random.rand()
        CR = 0.8 + 0.2 * np.random.rand()
    
    # Phase 3: Local search (Nelder-Mead style simplex) around best solution
    if best_x is not None:
        simplex_size = dim + 1
        simplex = np.zeros((simplex_size, dim))
        simplex_f = np.zeros(simplex_size)
        simplex[0] = best_x.copy()
        simplex_f[0] = best
        
        step = ranges * 0.05
        for i in range(1, simplex_size):
            if elapsed() >= max_time * 0.95:
                return best
            p = best_x.copy()
            p[i - 1 if i - 1 < dim else 0] += step[i - 1 if i - 1 < dim else 0]
            simplex[i] = clip(p)
            simplex_f[i] = evaluate(simplex[i])
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        while elapsed() < max_time * 0.95:
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            if elapsed() >= max_time * 0.95:
                return best
            
            if fr < simplex_f[0]:
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                xc = clip(centroid + rho * (simplex[-1] - centroid))
                fc = evaluate(xc)
                if fc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    for i in range(1, simplex_size):
                        if elapsed() >= max_time * 0.95:
                            return best
                        simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                        simplex_f[i] = evaluate(simplex[i])
    
    return best
