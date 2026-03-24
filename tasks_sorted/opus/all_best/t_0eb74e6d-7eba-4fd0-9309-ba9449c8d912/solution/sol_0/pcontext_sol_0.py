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
    ranges = upper - lower
    
    # --- CMA-ES inspired approach combined with differential evolution and local search ---
    
    # Phase 1: Initial sampling with Latin Hypercube-like strategy
    n_init = min(100, max(20, dim * 10))
    population_size = min(50, max(15, 4 + int(3 * np.log(dim))))
    
    # Generate initial population
    pop = np.random.uniform(0, 1, (n_init, dim)) * ranges + lower
    fit = np.array([func(ind) for ind in pop])
    
    best_idx = np.argmin(fit)
    if fit[best_idx] < best:
        best = fit[best_idx]
        best_x = pop[best_idx].copy()
    
    # Sort and keep best as initial population
    sorted_idx = np.argsort(fit)
    pop = pop[sorted_idx[:population_size]].copy()
    fit = fit[sorted_idx[:population_size]].copy()
    
    def time_remaining():
        return max_time - (datetime.now() - start).total_seconds()
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    # Phase 2: Differential Evolution with adaptive parameters
    F = 0.8
    CR = 0.9
    generation = 0
    
    while time_remaining() > max_time * 0.3:
        if time_remaining() <= 0:
            return best
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        for i in range(population_size):
            if time_remaining() <= 0:
                return best
            
            # DE/current-to-best/1 with jitter
            idxs = list(range(population_size))
            idxs.remove(i)
            a, b = np.random.choice(idxs, 2, replace=False)
            
            # Adaptive F
            Fi = F + 0.1 * np.random.randn()
            Fi = np.clip(Fi, 0.1, 1.5)
            
            mutant = pop[i] + Fi * (best_x - pop[i]) + Fi * (pop[a] - pop[b])
            mutant = clip(mutant)
            
            # Binomial crossover
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            trial = clip(trial)
            
            trial_fit = func(trial)
            
            if trial_fit <= fit[i]:
                new_pop[i] = trial
                new_fit[i] = trial_fit
                
                if trial_fit < best:
                    best = trial_fit
                    best_x = trial.copy()
        
        pop = new_pop
        fit = new_fit
        generation += 1
        
        # Occasionally inject random individuals to avoid stagnation
        if generation % 20 == 0:
            worst_idx = np.argsort(fit)[-max(1, population_size // 5):]
            for idx in worst_idx:
                if time_remaining() <= 0:
                    return best
                new_ind = np.random.uniform(0, 1, dim) * ranges + lower
                new_f = func(new_ind)
                if new_f < fit[idx]:
                    pop[idx] = new_ind
                    fit[idx] = new_f
                    if new_f < best:
                        best = new_f
                        best_x = new_ind.copy()
    
    # Phase 3: Nelder-Mead-like local search around best solution
    simplex_size = dim + 1
    step = ranges * 0.05
    
    simplex = np.zeros((simplex_size, dim))
    simplex[0] = best_x.copy()
    for i in range(dim):
        simplex[i + 1] = best_x.copy()
        simplex[i + 1][i] += step[i] if best_x[i] + step[i] <= upper[i] else -step[i]
    
    simplex_fit = np.array([func(clip(s)) for s in simplex])
    
    for i in range(simplex_size):
        if simplex_fit[i] < best:
            best = simplex_fit[i]
            best_x = simplex[i].copy()
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    while time_remaining() > 0.1:
        if time_remaining() <= 0:
            return best
        
        order = np.argsort(simplex_fit)
        simplex = simplex[order]
        simplex_fit = simplex_fit[order]
        
        if simplex_fit[0] < best:
            best = simplex_fit[0]
            best_x = simplex[0].copy()
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect
        xr = clip(centroid + alpha * (centroid - simplex[-1]))
        fr = func(xr)
        if time_remaining() <= 0:
            return best
        
        if simplex_fit[0] <= fr < simplex_fit[-2]:
            simplex[-1] = xr
            simplex_fit[-1] = fr
            if fr < best:
                best = fr
                best_x = xr.copy()
            continue
        
        if fr < simplex_fit[0]:
            xe = clip(centroid + gamma * (xr - centroid))
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                simplex_fit[-1] = fe
                if fe < best:
                    best = fe
                    best_x = xe.copy()
            else:
                simplex[-1] = xr
                simplex_fit[-1] = fr
                if fr < best:
                    best = fr
                    best_x = xr.copy()
            continue
        
        # Contract
        xc = clip(centroid + rho * (simplex[-1] - centroid))
        fc = func(xc)
        if fc < simplex_fit[-1]:
            simplex[-1] = xc
            simplex_fit[-1] = fc
            if fc < best:
                best = fc
                best_x = xc.copy()
            continue
        
        # Shrink
        for i in range(1, simplex_size):
            if time_remaining() <= 0:
                return best
            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
            simplex_fit[i] = func(simplex[i])
            if simplex_fit[i] < best:
                best = simplex_fit[i]
                best_x = simplex[i].copy()
    
    return best
