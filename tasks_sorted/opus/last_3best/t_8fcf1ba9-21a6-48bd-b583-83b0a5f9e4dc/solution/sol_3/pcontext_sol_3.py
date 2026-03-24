#Looking at the results, algorithm 1 (score 4.07) significantly outperforms algorithms 2 and 3 (scores ~31-33). The CMA-ES implementation in algorithm 3 performed poorly, likely due to bugs or poor time allocation. I'll build on algorithm 1's successful SHADE + Nelder-Mead approach but add key improvements:
#
#1. **Fix CMA-ES implementation** with better numerical stability
#2. **Better time allocation** - more time for the global search phase
#3. **Population size reduction** (L-SHADE style)
#4. **Improved restart strategy**
#
import numpy as np
from datetime import datetime

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
    
    # --- Phase 1: LHS initialization ---
    pop_size = min(max(30, 8 * dim), 300)
    init_pop_size = pop_size
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        if elapsed() >= max_time * 0.9:
            return best
        fitness[i] = evaluate(population[i])
    
    # --- Phase 2: L-SHADE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    ki = 0
    archive = []
    stag = 0
    prev_best = best
    gen_count = 0
    max_gen_approx = 500
    min_pop = max(4, dim)
    
    while elapsed() < max_time * 0.70:
        gen_count += 1
        SF, SCR, Sdelta = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        si = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * 0.70:
                break
            ri = np.random.randint(H)
            # Cauchy for F
            Fi = M_F[ri]
            while True:
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            pb = si[np.random.randint(p)]
            
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            pool_size = pop_size + len(archive)
            r2c = i
            while r2c == i or r2c == r1:
                r2c = np.random.randint(pool_size)
            xr2 = population[r2c] if r2c < pop_size else archive[r2c - pop_size]
            
            v = population[i] + Fi * (population[pb] - population[i]) + Fi * (population[r1] - xr2)
            mask = np.random.random(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = clip(np.where(mask, v, population[i]))
            ft = evaluate(trial)
            if ft < fitness[i]:
                d = fitness[i] - ft
                SF.append(Fi); SCR.append(CRi); Sdelta.append(d)
                if len(archive) < init_pop_size:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial; new_fit[i] = ft
        
        population, fitness = new_pop, new_fit
        if SF:
            w = np.array(Sdelta); w /= w.sum() + 1e-30
            M_F[ki] = np.sum(w * np.array(SF)**2) / (np.sum(w * np.array(SF)) + 1e-30)
            M_CR[ki] = np.sum(w * np.array(SCR))
            ki = (ki + 1) % H
        
        # L-SHADE population reduction
        new_size = max(min_pop, int(round(init_pop_size - (init_pop_size - min_pop) * gen_count / max_gen_approx)))
        if new_size < pop_size:
            si2 = np.argsort(fitness)
            population = population[si2[:new_size]]
            fitness = fitness[si2[:new_size]]
            pop_size = new_size
        
        if abs(best - prev_best) < 1e-14: stag += 1
        else: stag = 0
        prev_best = best
        if stag > 20:
            si2 = np.argsort(fitness)
            for i in range(pop_size // 2, pop_size):
                if elapsed() >= max_time * 0.70: break
                population[si2[i]] = lower + np.random.random(dim) * ranges
                fitness[si2[i]] = evaluate(population[si2[i]])
            stag = 0

    # --- Phase 3: Nelder-Mead from top solutions ---
    def nelder_mead(x0, scale_factor, time_frac):
        n = dim
        scale = scale_factor * ranges
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += scale[i] if x0[i] + scale[i] <= upper[i] else -scale[i]
        simplex = clip(simplex)
        fs = np.array([evaluate(simplex[j]) for j in range(n+1)])
        for _ in range(10000):
            if elapsed() >= time_frac: break
            order = np.argsort(fs); simplex = simplex[order]; fs = fs[order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = clip(centroid + (centroid - simplex[-1])); fr = evaluate(xr)
            if fs[0] <= fr < fs[-2]: simplex[-1], fs[-1] = xr, fr
            elif fr < fs[0]:
                xe = clip(centroid + 2*(xr - centroid)); fe = evaluate(xe)
                simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            else:
                xc = clip(centroid + 0.5*(simplex[-1] - centroid)); fc = evaluate(xc)
                if fc < fs[-1]: simplex[-1], fs[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = clip(simplex[0] + 0.5*(simplex[i] - simplex[0]))
                        if elapsed() >= time_frac: return
                        fs[i] = evaluate(simplex[i])
            if np.std(fs) < 1e-16: break

    si = np.argsort(fitness)
    for attempt in range(min(4, pop_size)):
        if elapsed() >= max_time * 0.95: break
        nelder_mead(population[si[attempt]], 0.05/(1+attempt), max_time * 0.95)
    
    if elapsed() < max_time * 0.99 and best_params is not None:
        nelder_mead(best_params, 0.002, max_time * 0.99)
    
    return best
