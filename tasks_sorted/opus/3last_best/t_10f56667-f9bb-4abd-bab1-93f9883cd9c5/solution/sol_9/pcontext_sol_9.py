#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    bounds_arr = np.array(bounds)
    lb = bounds_arr[:, 0]
    ub = bounds_arr[:, 1]
    
    eval_count = 0
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def ev(x):
        nonlocal best, best_x, eval_count
        eval_count += 1
        x_clipped = np.clip(x, lb, ub)
        val = func(x_clipped)
        if val < best:
            best = val
            best_x = x_clipped.copy()
        return val
    
    def clip(x):
        return np.clip(x, lb, ub)
    
    # Phase 1: Latin Hypercube Sampling for initial exploration
    n_init = min(200, max(50, dim * 10))
    pop_size = n_init
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        for j in range(pop_size):
            population[j, i] = lb[i] + (perm[j] + np.random.random()) / pop_size * (ub[i] - lb[i])
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = ev(population[i])
    
    # Sort population by fitness
    order = np.argsort(fitness)
    population = population[order]
    fitness = fitness[order]
    
    # Phase 2: Differential Evolution with restarts
    de_pop_size = min(max(20, 4 * dim), pop_size)
    de_pop = population[:de_pop_size].copy()
    de_fit = fitness[:de_pop_size].copy()
    
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.85:
        generation += 1
        improved = False
        
        for i in range(de_pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            # Select 3 distinct indices different from i
            idxs = list(range(de_pop_size))
            idxs.remove(i)
            chosen = np.random.choice(idxs, 3, replace=False)
            a, b, c = de_pop[chosen[0]], de_pop[chosen[1]], de_pop[chosen[2]]
            
            # Current-to-best mutation
            best_idx = np.argmin(de_fit)
            if np.random.random() < 0.5:
                mutant = de_pop[i] + F * (de_pop[best_idx] - de_pop[i]) + F * (a - b)
            else:
                mutant = a + F * (b - c)
            
            # Crossover
            trial = de_pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial = clip(trial)
            trial_fit = ev(trial)
            
            if trial_fit <= de_fit[i]:
                de_pop[i] = trial
                de_fit[i] = trial_fit
                improved = True
        
        if best < prev_best:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        # Restart if stagnant
        if stagnation > 10:
            stagnation = 0
            keep = max(2, de_pop_size // 5)
            order2 = np.argsort(de_fit)
            de_pop = de_pop[order2]
            de_fit = de_fit[order2]
            for i in range(keep, de_pop_size):
                de_pop[i] = np.array([np.random.uniform(lb[j], ub[j]) for j in range(dim)])
                de_fit[i] = ev(de_pop[i])
                if elapsed() >= max_time * 0.85:
                    break
    
    # Phase 3: Nelder-Mead local search from best point
    if best_x is not None and elapsed() < max_time * 0.95:
        n = dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        s_fit = np.zeros(n + 1)
        s_fit[0] = best
        
        step = 0.05 * (ub - lb)
        for i in range(n):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1][i] += step[i]
            simplex[i + 1] = clip(simplex[i + 1])
            s_fit[i + 1] = ev(simplex[i + 1])
            if elapsed() >= max_time * 0.95:
                return best
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while elapsed() < max_time * 0.95:
            order3 = np.argsort(s_fit)
            simplex = simplex[order3]
            s_fit = s_fit[order3]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = ev(xr)
            if elapsed() >= max_time * 0.95:
                return best
            
            if fr < s_fit[0]:
                # Expand
                xe = clip(centroid + gamma * (xr - centroid))
                fe = ev(xe)
                if fe < fr:
                    simplex[-1] = xe
                    s_fit[-1] = fe
                else:
                    simplex[-1] = xr
                    s_fit[-1] = fr
            elif fr < s_fit[-2]:
                simplex[-1] = xr
                s_fit[-1] = fr
            else:
                if fr < s_fit[-1]:
                    # Outside contraction
                    xc = clip(centroid + rho * (xr - centroid))
                    fc = ev(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        s_fit[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            s_fit[i] = ev(simplex[i])
                            if elapsed() >= max_time * 0.95:
                                return best
                else:
                    # Inside contraction
                    xc = clip(centroid - rho * (centroid - simplex[-1]))
                    fc = ev(xc)
                    if fc < s_fit[-1]:
                        simplex[-1] = xc
                        s_fit[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                            s_fit[i] = ev(simplex[i])
                            if elapsed() >= max_time * 0.95:
                                return best
    
    return best
