#Looking at the previous algorithm, I can identify several areas for improvement:
#
#1. **Better DE variant**: Use SHADE (Success-History based Adaptive DE) with parameter adaptation
#2. **More efficient time usage**: Vectorize operations, reduce overhead
#3. **Multiple restarts of local search**: Run Nelder-Mead from multiple good points
#4. **CMA-ES inspired search**: Add a covariance-matrix adaptation component
#5. **Better stagnation handling**: More aggressive diversification
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
    
    def clip(x):
        return np.clip(x, lower, upper)
    
    def evaluate(x):
        nonlocal best, best_params
        x = clip(x)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f
    
    # --- Phase 1: Sobol-like initial sampling ---
    pop_size = min(max(30, 8 * dim), 300)
    
    # LHS initialization
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.array([evaluate(population[i]) for i in range(pop_size) if elapsed() < max_time * 0.9])
    if len(fitness) < pop_size:
        population = population[:len(fitness)]
        pop_size = len(fitness)
    
    # --- Phase 2: SHADE-like DE ---
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size
    
    stagnation = 0
    prev_best = best
    
    while elapsed() < max_time * 0.75:
        S_F = []
        S_CR = []
        S_delta = []
        
        new_pop = population.copy()
        new_fit = fitness.copy()
        
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.75:
                break
            
            ri = np.random.randint(memory_size)
            F_i = min(1.0, max(0.1, np.random.standard_cauchy() * 0.1 + M_F[ri]))
            CR_i = min(1.0, max(0.0, np.random.normal(M_CR[ri], 0.1)))
            
            # p-best
            p = max(2, int(0.15 * pop_size))
            pbest = sorted_idx[np.random.randint(p)]
            
            # r1 from population
            idxs = [j for j in range(pop_size) if j != i]
            r1 = np.random.choice(idxs)
            
            # r2 from population + archive
            combined = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
            combined = [j for j in combined if j != i and j != r1]
            r2c = np.random.choice(combined)
            if r2c < pop_size:
                xr2 = population[r2c]
            else:
                xr2 = archive[r2c - pop_size]
            
            mutant = population[i] + F_i * (population[pbest] - population[i]) + F_i * (population[r1] - xr2)
            
            # Binomial crossover
            mask = np.random.random(dim) < CR_i
            mask[np.random.randint(dim)] = True
            trial = np.where(mask, mutant, population[i])
            trial = clip(trial)
            
            # Bounce-back for out-of-bounds (already clipped)
            f_trial = evaluate(trial)
            
            if f_trial < fitness[i]:
                delta = fitness[i] - f_trial
                S_F.append(F_i)
                S_CR.append(CR_i)
                S_delta.append(delta)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial
                new_fit[i] = f_trial
        
        population = new_pop
        fitness = new_fit
        
        if S_F:
            w = np.array(S_delta) / (sum(S_delta) + 1e-30)
            M_F[k] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
            M_CR[k] = np.sum(w * np.array(S_CR))
            k = (k + 1) % memory_size
        
        if abs(best - prev_best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 20:
            half = pop_size // 2
            si = np.argsort(fitness)
            for i in range(half, pop_size):
                idx = si[i]
                population[idx] = lower + np.random.random(dim) * ranges
                if elapsed() >= max_time * 0.75:
                    break
                fitness[idx] = evaluate(population[idx])
            stagnation = 0
    
    # --- Phase 3: Multiple Nelder-Mead restarts from top solutions ---
    if best_params is None:
        return best
    
    def nelder_mead(x0, scale_factor, time_frac):
        nonlocal best, best_params
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += scale[i] if x0[i] + scale[i] <= upper[i] else -scale[i]
        simplex = clip(simplex)
        fs = np.array([evaluate(simplex[j]) for j in range(n+1) if elapsed() < max_time * time_frac])
        if len(fs) < n + 1:
            return
        
        for _ in range(10000):
            if elapsed() >= max_time * time_frac:
                break
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = clip(centroid + (centroid - simplex[-1]))
            fr = evaluate(xr)
            if fs[0] <= fr < fs[-2]:
                simplex[-1], fs[-1] = xr, fr
            elif fr < fs[0]:
                xe = clip(centroid + 2*(xr - centroid))
                fe = evaluate(xe)
                simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            else:
                xc = clip(centroid + 0.5*(simplex[-1] - centroid))
                fc = evaluate(xc)
                if fc < fs[-1]:
                    simplex[-1], fs[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = clip(simplex[0] + 0.5*(simplex[i] - simplex[0]))
                        if elapsed() >= max_time * time_frac:
                            return
                        fs[i] = evaluate(simplex[i])
            if np.std(fs) < 1e-16:
                break
    
    # Run NM from best and top solutions
    sorted_idx = np.argsort(fitness)
    top_k = min(3, pop_size)
    for attempt in range(top_k):
        if elapsed() >= max_time * 0.97:
            break
        x0 = population[sorted_idx[attempt]].copy()
        nelder_mead(x0, 0.05, max_time * 0.97)
    
    # Final fine NM
    if elapsed() < max_time * 0.99 and best_params is not None:
        nelder_mead(best_params, 0.005, max_time * 0.99)
    
    return best
#
#Key improvements:
#1. **SHADE adaptation**: F and CR are adapted based on success history with weighted Lehmer mean, much better than random adaptation
#2. **Archive**: Failed parents stored for diversity in mutation
#3. **Multiple NM restarts**: From top-3 solutions, not just the best
#4. **Fine-grained NM**: A final very small-scale NM for polishing
#5. **Better time management**: More time allocated to DE phase
