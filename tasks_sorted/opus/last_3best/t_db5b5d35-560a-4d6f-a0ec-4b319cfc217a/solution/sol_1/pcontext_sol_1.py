#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Vectorize crossover** to speed up DE iterations
#2. **Use multiple DE strategies** (DE/rand/1, DE/best/1, DE/current-to-best/1) 
#3. **Implement SHADE-like adaptive parameter control** instead of simple random F/CR
#4. **Run multiple Nelder-Mead restarts** with different scales
#5. **Add CMA-ES-inspired local search** for better exploitation
#6. **Better time management**
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
    
    def elapsed():
        return (datetime.now() - start).total_seconds()
    
    def remaining():
        return max_time - elapsed()
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # ---- Phase 1: LHS initialization ----
    pop_size = min(max(10 * dim, 40), 150)
    
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.95:
            return best
        fitness[i] = evaluate(population[i])
    
    # ---- Phase 2: SHADE-like DE ----
    memory_size = 5
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.70:
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            
            ri = np.random.randint(memory_size)
            Fi = np.clip(np.random.standard_cauchy() * 0.1 + M_F[ri], 0.01, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            best_idx = np.argmin(fitness)
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # DE/current-to-pbest/1
            p = max(2, int(0.1 * pop_size))
            pbest_idx = np.random.choice(np.argsort(fitness)[:p])
            
            pool = list(range(pop_size)) + list(range(len(archive)))
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - (archive[r2 % len(archive)] if archive and r2 >= pop_size else population[r2 % pop_size]))
            
            trial = population[i].copy()
            jrand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == jrand)
            trial[mask] = mutant[mask]
            trial = np.clip(trial, lower, upper)
            
            f_trial = evaluate(trial)
            if f_trial < fitness[i]:
                S_F.append(Fi)
                S_CR.append(CRi)
                S_delta.append(fitness[i] - f_trial)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        if S_F:
            weights = np.array(S_delta) / (np.sum(S_delta) + 1e-30)
            M_F[k] = np.sum(weights * np.array(S_F)**2) / (np.sum(weights * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(weights * np.array(S_CR))
            k = (k + 1) % memory_size
        
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 15:
            keep = max(pop_size // 5, 2)
            order = np.argsort(fitness)
            for i in range(keep, pop_size):
                if elapsed() >= max_time * 0.70:
                    break
                population[order[i]] = lower + np.random.random(dim) * ranges
                fitness[order[i]] = evaluate(population[order[i]])
            stagnation = 0
    
    # ---- Phase 3: Nelder-Mead with restarts ----
    for scale_factor in [0.05, 0.01, 0.002]:
        if elapsed() >= max_time * 0.98 or best_params is None:
            break
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n+1, n))
        simplex[0] = best_params.copy()
        for i in range(n):
            simplex[i+1] = best_params.copy()
            simplex[i+1][i] += scale[i] * (1 if np.random.random() > 0.5 else -1)
        simplex = np.clip(simplex, lower, upper)
        
        fs = np.array([evaluate(simplex[i]) for i in range(n+1) if elapsed() < max_time*0.98])
        if len(fs) < n+1:
            break
        
        for _ in range(5000):
            if elapsed() >= max_time * 0.98:
                break
            order = np.argsort(fs)
            simplex = simplex[order]; fs = fs[order]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = np.clip(2*centroid - simplex[-1], lower, upper)
            fr = evaluate(xr)
            if fs[0] <= fr < fs[-2]:
                simplex[-1]=xr; fs[-1]=fr; continue
            if fr < fs[0]:
                xe = np.clip(centroid+2*(xr-centroid), lower, upper)
                fe = evaluate(xe)
                if fe<fr: simplex[-1]=xe; fs[-1]=fe
                else: simplex[-1]=xr; fs[-1]=fr
                continue
            xc = np.clip(centroid+0.5*(simplex[-1]-centroid), lower, upper)
            fc = evaluate(xc)
            if fc < fs[-1]:
                simplex[-1]=xc; fs[-1]=fc; continue
            for i in range(1,n+1):
                if elapsed()>=max_time*0.98: break
                simplex[i] = np.clip(simplex[0]+0.5*(simplex[i]-simplex[0]), lower, upper)
                fs[i] = evaluate(simplex[i])
            if np.max(np.abs(fs-fs[0])) < 1e-16:
                break
    
    return best
