#Looking at the current algorithm, I can see several areas for improvement:
#
#1. **Better DE strategy**: Use SHADE-like adaptive parameter control
#2. **More aggressive local search**: Use a proper pattern search / coordinate descent
#3. **Better population management**: Smaller population for faster convergence, with opposition-based learning
#4. **CMA-ES-inspired local search** around the best solution
#
import numpy as np
from datetime import datetime, timedelta

def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')
    best_params = None
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ranges = upper - lower
    
    evals = 0
    def eval_func(x):
        nonlocal evals, best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        evals += 1
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    def time_left():
        return (datetime.now() - start).total_seconds() < max_time * 0.95
    
    # Phase 1: Initial sampling with Sobol-like coverage
    pop_size = min(15 * dim, 150)
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        perm = np.random.permutation(pop_size)
        population[:, i] = lower[i] + (perm + np.random.rand(pop_size)) / pop_size * ranges[i]
    
    fitness = np.array([eval_func(population[i]) for i in range(pop_size) if time_left() or i == 0])
    if len(fitness) < pop_size:
        population = population[:len(fitness)]
    
    if not time_left():
        return best

    # SHADE memory
    mem_size = 10
    M_F = np.full(mem_size, 0.5)
    M_CR = np.full(mem_size, 0.5)
    mem_idx = 0
    
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best
    
    while time_left():
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        S_F, S_CR, S_w = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        p_best_size = max(2, int(0.1 * len(population)))
        
        for i in range(len(population)):
            if not time_left():
                return best
            
            ri = np.random.randint(mem_size)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.01, 1.5)
            CRi = np.clip(np.random.randn() * 0.1 + M_CR[ri], 0.0, 1.0)
            
            p_best = np.random.randint(p_best_size)
            idxs = [j for j in range(len(population)) if j != i]
            a = np.random.choice(idxs)
            
            combined = list(range(len(population))) + list(range(len(archive)))
            b_pool = [j for j in idxs]
            if archive:
                b_pool_ext = b_pool + list(range(len(population), len(population) + len(archive)))
            else:
                b_pool_ext = b_pool
            b = np.random.choice(len(b_pool_ext))
            if b < len(b_pool):
                xb = population[b_pool[b]]
            else:
                xb = archive[b_pool_ext[b] - len(population)]
            
            mutant = population[i] + Fi * (population[p_best] - population[i]) + Fi * (population[a] - xb)
            
            cross = np.random.rand(dim) < CRi
            if not np.any(cross):
                cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, population[i])
            
            for d in range(dim):
                if trial[d] < lower[d]:
                    trial[d] = (lower[d] + population[i][d]) / 2
                if trial[d] > upper[d]:
                    trial[d] = (upper[d] + population[i][d]) / 2
            
            t_fit = eval_func(trial)
            if t_fit <= fitness[i]:
                if t_fit < fitness[i]:
                    archive.append(population[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(np.random.randint(len(archive)))
                    w = abs(fitness[i] - t_fit)
                    S_F.append(Fi); S_CR.append(CRi); S_w.append(w)
                new_pop[i] = trial
                new_fit[i] = t_fit
        
        population, fitness = new_pop, new_fit
        
        if S_F:
            w = np.array(S_w); w /= w.sum() + 1e-30
            M_F[mem_idx] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[mem_idx] = np.sum(w * np.array(S_CR))
            mem_idx = (mem_idx + 1) % mem_size
        
        if abs(best - prev_best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0; prev_best = best
        
        if stagnation > 30:
            half = len(population) // 2
            for i in range(half, len(population)):
                population[i] = lower + np.random.rand(dim) * ranges
                if not time_left(): return best
                fitness[i] = eval_func(population[i])
            stagnation = 0
    
    return best
