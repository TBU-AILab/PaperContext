#
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        val = func(x)
        if val < best:
            best = val
            best_params = x.copy()
        return val
    
    # Phase 1: Latin Hypercube-like initial sampling
    n_init = min(100, max(20, dim * 10))
    population = []
    fitness_list = []
    
    for i in range(n_init):
        if elapsed() >= max_time * 0.95:
            return best
        x = np.array([np.random.uniform(l, u) for l, u in bounds])
        val = evaluate(x)
        population.append(x.copy())
        fitness_list.append(val)
    
    population = np.array(population)
    fitness_list = np.array(fitness_list)
    
    # Phase 2: Differential Evolution with restarts and Nelder-Mead refinement
    pop_size = min(50, max(15, dim * 5))
    
    # Select best from initial sampling to seed population
    sorted_idx = np.argsort(fitness_list)
    if len(sorted_idx) >= pop_size:
        pop = population[sorted_idx[:pop_size]].copy()
        pop_fit = fitness_list[sorted_idx[:pop_size]].copy()
    else:
        pop = population.copy()
        pop_fit = fitness_list.copy()
        while len(pop) < pop_size:
            x = np.array([np.random.uniform(l, u) for l, u in bounds])
            val = evaluate(x)
            pop = np.vstack([pop, x.reshape(1, -1)])
            pop_fit = np.append(pop_fit, val)
    
    F = 0.8
    CR = 0.9
    
    generation = 0
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.85:
        generation += 1
        improved = False
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.85:
                break
            
            # Mutation: DE/best/1 with occasional DE/rand/1
            idxs = list(range(pop_size))
            idxs.remove(i)
            
            if np.random.random() < 0.8:
                # DE/best/1
                best_idx = np.argmin(pop_fit)
                a, b = pop[np.random.choice(idxs, 2, replace=False)]
                mutant = pop[best_idx] + F * (a - b)
            else:
                # DE/rand/1
                chosen = np.random.choice(idxs, 3, replace=False)
                r1, r2, r3 = pop[chosen[0]], pop[chosen[1]], pop[chosen[2]]
                mutant = r1 + F * (r2 - r3)
            
            mutant = clip(mutant)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_fit = evaluate(trial)
            
            if trial_fit <= pop_fit[i]:
                pop[i] = trial
                pop_fit[i] = trial_fit
                improved = True
        
        if best < prev_best:
            stagnation = 0
            prev_best = best
        else:
            stagnation += 1
        
        # If stagnating, reinitialize worst half
        if stagnation > 10:
            sorted_idx = np.argsort(pop_fit)
            half = pop_size // 2
            for k in range(half, pop_size):
                idx = sorted_idx[k]
                pop[idx] = np.array([np.random.uniform(l, u) for l, u in bounds])
                pop_fit[idx] = evaluate(pop[idx])
            stagnation = 0
    
    # Phase 3: Nelder-Mead refinement around best solution
    if best_params is not None and elapsed() < max_time * 0.95:
        n = dim
        step = 0.05 * (upper - lower)
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_params.copy()
        simplex_fit = np.zeros(n + 1)
        simplex_fit[0] = best
        
        for i in range(n):
            simplex[i + 1] = best_params.copy()
            simplex[i + 1][i] += step[i]
            simplex[i + 1] = clip(simplex[i + 1])
            simplex_fit[i + 1] = evaluate(simplex[i + 1])
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while elapsed() < max_time * 0.98:
            order = np.argsort(simplex_fit)
            simplex = simplex[order]
            simplex_fit = simplex_fit[order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = clip(centroid + alpha * (centroid - simplex[-1]))
            fr = evaluate(xr)
            
            if fr < simplex_fit[-2] and fr >= simplex_fit[0]:
                simplex[-1] = xr
                simplex_fit[-1] = fr
                continue
            
            if fr < simplex_fit[0]:
                # Expansion
                xe = clip(centroid + gamma * (xr - centroid))
                fe = evaluate(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_fit[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_fit[-1] = fr
                continue
            
            # Contraction
            xc = clip(centroid + rho * (simplex[-1] - centroid))
            fc = evaluate(xc)
            if fc < simplex_fit[-1]:
                simplex[-1] = xc
                simplex_fit[-1] = fc
                continue
            
            # Shrink
            for i in range(1, n + 1):
                simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                simplex_fit[i] = evaluate(simplex[i])
    
    return best
