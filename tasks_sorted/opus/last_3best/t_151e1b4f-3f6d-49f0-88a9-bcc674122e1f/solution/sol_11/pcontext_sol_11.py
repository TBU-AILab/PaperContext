import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_x = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def eval_func(x):
        nonlocal best, best_x
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_x = x.copy()
        return f
    
    # Phase 1: Latin Hypercube Sampling for initial exploration
    n_init = min(200, max(50, dim * 10))
    pop_size = min(60, max(20, dim * 4))
    
    init_xs = []
    init_fs = []
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        x = np.array([np.random.uniform(l, u) for l, u in bounds])
        f = eval_func(x)
        init_xs.append(x.copy())
        init_fs.append(f)
    
    # Phase 2: Differential Evolution
    # Initialize population with best from initial sampling
    indices = np.argsort(init_fs)[:pop_size]
    population = np.array([init_xs[i] for i in indices])
    fitness = np.array([init_fs[i] for i in indices])
    
    F = 0.8
    CR = 0.9
    generation = 0
    
    while elapsed() < max_time * 0.75:
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.75:
                break
            
            # Select three distinct random indices different from i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Current-to-best mutation
            best_idx = np.argmin(fitness)
            F_i = F + 0.1 * np.random.randn()
            F_i = np.clip(F_i, 0.1, 1.5)
            
            mutant = population[i] + F_i * (population[best_idx] - population[i]) + F_i * (population[r1] - population[r2])
            mutant = clip(mutant)
            
            # Crossover
            CR_i = np.clip(CR + 0.1 * np.random.randn(), 0.0, 1.0)
            cross_points = np.random.rand(dim) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial = clip(trial)
            
            f_trial = eval_func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        generation += 1
    
    # Phase 3: Nelder-Mead simplex on best solution found
    if best_x is not None and elapsed() < max_time * 0.95:
        n = dim
        step = 0.05 * (upper - lower)
        simplex = [best_x.copy()]
        simplex_f = [best]
        
        for i in range(n):
            if elapsed() >= max_time * 0.95:
                return best
            xi = best_x.copy()
            xi[i] = xi[i] + step[i]
            xi = clip(xi)
            fi = eval_func(xi)
            simplex.append(xi)
            simplex_f.append(fi)
        
        simplex = list(zip(simplex_f, [s.copy() for s in simplex]))
        
        while elapsed() < max_time * 0.95:
            simplex.sort(key=lambda p: p[0])
            simplex_f_sorted = [s[0] for s in simplex]
            simplex_x_sorted = [s[1] for s in simplex]
            
            c = np.mean(simplex_x_sorted[:-1], axis=0)
            
            # Reflect
            xr = clip(c + 1.0 * (c - simplex_x_sorted[-1]))
            fr = eval_func(xr)
            
            if fr < simplex_f_sorted[0]:
                xe = clip(c + 2.0 * (xr - c))
                fe = eval_func(xe)
                if fe < fr:
                    simplex[-1] = (fe, xe)
                else:
                    simplex[-1] = (fr, xr)
            elif fr < simplex_f_sorted[-2]:
                simplex[-1] = (fr, xr)
            else:
                xc = clip(c + 0.5 * (simplex_x_sorted[-1] - c))
                fc = eval_func(xc)
                if fc < simplex_f_sorted[-1]:
                    simplex[-1] = (fc, xc)
                else:
                    best_pt = simplex_x_sorted[0]
                    new_simplex = [(simplex[0][0], simplex[0][1])]
                    for j in range(1, len(simplex)):
                        if elapsed() >= max_time * 0.95:
                            return best
                        xs = clip(best_pt + 0.5 * (simplex_x_sorted[j] - best_pt))
                        fs = eval_func(xs)
                        new_simplex.append((fs, xs))
                    simplex = new_simplex
    
    return best