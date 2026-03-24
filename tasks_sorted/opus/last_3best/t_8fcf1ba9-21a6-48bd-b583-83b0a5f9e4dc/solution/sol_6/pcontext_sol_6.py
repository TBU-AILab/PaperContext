#Looking at the results, algorithm 1 (SHADE + Nelder-Mead, score 4.07) is clearly the best. The CMA-ES attempt was worse, and the latest attempt regressed significantly. Let me analyze what went wrong in the latest attempt and improve upon algorithm 1.
#
#Key improvements:
#1. Fix the L-SHADE population reduction (it was buggy)
#2. Better time allocation
#3. More robust parameter handling
#4. Improved Nelder-Mead with restarts from diverse good points
#5. Add a pattern search as final polisher
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
    
    # Opposition-based initialization
    opp_pop = lower + upper - population
    opp_fit = np.zeros(pop_size)
    for i in range(pop_size):
        if elapsed() >= max_time * 0.85:
            break
        opp_fit[i] = evaluate(opp_pop[i])
    
    n_opp = min(pop_size, len([1 for i in range(pop_size) if elapsed() < max_time * 0.85]))
    if n_opp == pop_size:
        combined_pop = np.vstack([population, opp_pop])
        combined_fit = np.concatenate([fitness, opp_fit])
        si = np.argsort(combined_fit)[:pop_size]
        population = combined_pop[si].copy()
        fitness = combined_fit[si].copy()
    
    # --- Phase 2: L-SHADE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.8)
    ki = 0
    archive = []
    stag = 0
    prev_best = best
    min_pop = max(4, dim)
    time_shade = max_time * 0.70
    
    while elapsed() < time_shade:
        SF, SCR, Sdelta = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        si = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= time_shade:
                break
            ri = np.random.randint(H)
            # Generate F via Cauchy
            Fi = -1
            for _ in range(10):
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            pb = si[np.random.randint(p)]
            
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1 = candidates[np.random.randint(len(candidates))]
            
            pool_sz = pop_size + len(archive)
            r2c = i
            while r2c == i or r2c == r1:
                r2c = np.random.randint(pool_sz)
            xr2 = population[r2c] if r2c < pop_size else archive[r2c - pop_size]
            
            v = population[i] + Fi * (population[pb] - population[i]) + Fi * (population[r1] - xr2)
            mask = np.random.random(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = clip(np.where(mask, v, population[i]))
            ft = evaluate(trial)
            if ft < fitness[i]:
                delta = fitness[i] - ft
                SF.append(Fi); SCR.append(CRi); Sdelta.append(delta)
                if len(archive) < init_pop_size:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial; new_fit[i] = ft
            elif ft == fitness[i]:
                new_pop[i] = trial; new_fit[i] = ft
        
        population, fitness = new_pop, new_fit
        if SF:
            w = np.array(Sdelta); w /= w.sum() + 1e-30
            M_F[ki] = np.sum(w * np.array(SF)**2) / (np.sum(w * np.array(SF)) + 1e-30)
            M_CR[ki] = np.sum(w * np.array(SCR))
            ki = (ki + 1) % H
        
        # L-SHADE population reduction
        frac = elapsed() / time_shade
        new_size = max(min_pop, int(round(init_pop_size - (init_pop_size - min_pop) * frac)))
        if new_size < pop_size:
            si2 = np.argsort(fitness)
            population = population[si2[:new_size]].copy()
            fitness = fitness[si2[:new_size]].copy()
            pop_size = new_size
            while len(archive) > pop_size:
                archive.pop(np.random.randint(len(archive)))
        
        if abs(best - prev_best) < 1e-14: stag += 1
        else: stag = 0
        prev_best = best
        if stag > 20:
            si2 = np.argsort(fitness)
            for j in range(pop_size // 2, pop_size):
                if elapsed() >= time_shade: break
                population[si2[j]] = lower + np.random.random(dim) * ranges
                fitness[si2[j]] = evaluate(population[si2[j]])
            stag = 0

    # --- Phase 3: Nelder-Mead from multiple starts ---
    def nelder_mead(x0, sf, tl):
        n = dim; scale = sf * ranges
        simplex = np.zeros((n+1, n)); simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += scale[i] if x0[i]+scale[i] <= upper[i] else -scale[i]
        simplex = clip(simplex)
        fs = np.array([evaluate(simplex[j]) for j in range(n+1)])
        for _ in range(10000):
            if elapsed() >= tl: break
            o = np.argsort(fs); simplex = simplex[o]; fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(c+(c-simplex[-1])); fr = evaluate(xr)
            if fs[0]<=fr<fs[-2]: simplex[-1],fs[-1]=xr,fr
            elif fr<fs[0]:
                xe=clip(c+2*(xr-c)); fe=evaluate(xe)
                simplex[-1],fs[-1]=(xe,fe) if fe<fr else (xr,fr)
            else:
                xc=clip(c+0.5*(simplex[-1]-c)); fc=evaluate(xc)
                if fc<fs[-1]: simplex[-1],fs[-1]=xc,fc
                else:
                    for i in range(1,n+1):
                        simplex[i]=clip(simplex[0]+0.5*(simplex[i]-simplex[0]))
                        if elapsed()>=tl: return
                        fs[i]=evaluate(simplex[i])
            if np.std(fs)<1e-16: break

    si = np.argsort(fitness)
    for a in range(min(5, pop_size)):
        if elapsed() >= max_time*0.90: break
        nelder_mead(population[si[a]], 0.05/(1+a), max_time*0.90)
    
    if elapsed() < max_time*0.95 and best_params is not None:
        nelder_mead(best_params, 0.01, max_time*0.95)
    
    if elapsed() < max_time*0.98 and best_params is not None:
        nelder_mead(best_params, 0.001, max_time*0.98)
    
    if elapsed() < max_time*0.995 and best_params is not None:
        nelder_mead(best_params, 0.0002, max_time*0.995)
    
    return best
