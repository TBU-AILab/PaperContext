#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    bounds_arr = np.array(bounds)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
        return f, x
    
    # Phase 1: Latin Hypercube-like initial sampling
    pop_size = min(20 * dim, 200)
    population = np.random.uniform(0, 1, (pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = (perm + population[:, i]) / pop_size
    population = lower + population * (upper - lower)
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i], population[i] = evaluate(population[i])
    
    best_idx = np.argmin(fitness)
    best_sol = population[best_idx].copy()
    best_fit = fitness[best_idx]
    
    # Phase 2: Differential Evolution with restarts and local search
    F = 0.8
    CR = 0.9
    generation = 0
    
    stagnation = 0
    prev_best = best_fit
    
    while elapsed() < max_time * 0.85:
        generation += 1
        
        # Adaptive parameters
        F_val = 0.5 + 0.3 * np.random.random()
        CR_val = 0.8 + 0.2 * np.random.random()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            # DE/current-to-best/1/bin
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            mutant = population[i] + F_val * (best_sol - population[i]) + F_val * (population[r1] - population[r2])
            mutant = clip(mutant)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR_val or j == j_rand:
                    trial[j] = mutant[j]
            
            ft, trial = evaluate(trial)
            
            if ft <= fitness[i]:
                population[i] = trial
                fitness[i] = ft
                if ft < best_fit:
                    best_fit = ft
                    best_sol = trial.copy()
        
        # Check stagnation
        if abs(prev_best - best_fit) < 1e-12:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best_fit
        
        # Restart worst half if stagnant
        if stagnation > 10 + dim:
            sorted_idx = np.argsort(fitness)
            half = pop_size // 2
            for idx in sorted_idx[half:]:
                population[idx] = np.random.uniform(lower, upper)
                fitness[idx], population[idx] = evaluate(population[idx])
                if elapsed() >= max_time * 0.85:
                    break
            stagnation = 0
    
    # Phase 3: Nelder-Mead-like local search around best solution
    simplex_size = dim + 1
    simplex = np.zeros((simplex_size, dim))
    simplex_fit = np.full(simplex_size, float('inf'))
    simplex[0] = best_sol.copy()
    simplex_fit[0] = best_fit
    
    scale = 0.05 * (upper - lower)
    for i in range(1, simplex_size):
        simplex[i] = best_sol.copy()
        simplex[i][((i - 1) % dim)] += scale[(i - 1) % dim]
        simplex[i] = clip(simplex[i])
        if elapsed() >= max_time * 0.95:
            return best
        simplex_fit[i], simplex[i] = evaluate(simplex[i])
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    while elapsed() < max_time * 0.95:
        order = np.argsort(simplex_fit)
        simplex = simplex[order]
        simplex_fit = simplex_fit[order]
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect
        xr = clip(centroid + alpha * (centroid - simplex[-1]))
        fr, xr = evaluate(xr)
        if elapsed() >= max_time * 0.95:
            return best
        
        if fr < simplex_fit[0]:
            xe = clip(centroid + gamma * (xr - centroid))
            fe, xe = evaluate(xe)
            if fe < fr:
                simplex[-1] = xe
                simplex_fit[-1] = fe
            else:
                simplex[-1] = xr
                simplex_fit[-1] = fr
        elif fr < simplex_fit[-2]:
            simplex[-1] = xr
            simplex_fit[-1] = fr
        else:
            xc = clip(centroid + rho * (simplex[-1] - centroid))
            fc, xc = evaluate(xc)
            if fc < simplex_fit[-1]:
                simplex[-1] = xc
                simplex_fit[-1] = fc
            else:
                for i in range(1, simplex_size):
                    simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                    simplex_fit[i], simplex[i] = evaluate(simplex[i])
                    if elapsed() >= max_time * 0.95:
                        return best
    
    return best
