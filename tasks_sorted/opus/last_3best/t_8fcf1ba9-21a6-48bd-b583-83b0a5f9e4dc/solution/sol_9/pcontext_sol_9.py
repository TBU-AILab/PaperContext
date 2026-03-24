#Looking at the results, algorithm 1 (SHADE + Nelder-Mead, score 4.07) is the best. The CMA-ES version (score 32.69) and the latest attempt (score 208.62) regressed. Let me carefully analyze what works and improve:
#
#1. The SHADE + NM combo is strong - keep it
#2. The latest algorithm had timing issues and poor NM scheduling
#3. I'll add: better initialization, periodic local search during DE, improved NM with adaptive restarts, and a small CMA-ES polish
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
    pop_size = min(max(30, 8 * dim), 250)
    population = np.zeros((pop_size, dim))
    for d in range(dim):
        perm = np.random.permutation(pop_size)
        for i in range(pop_size):
            population[i, d] = lower[d] + (perm[i] + np.random.random()) / pop_size * ranges[d]
    
    fitness = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        fitness[i] = evaluate(population[i])
    
    # Opposition-based learning
    opp_pop = lower + upper - population
    opp_fit = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.80:
            break
        opp_fit[i] = evaluate(opp_pop[i])
    
    valid_count = np.sum(~np.isinf(opp_fit))
    if valid_count == pop_size:
        combined_pop = np.vstack([population, opp_pop])
        combined_fit = np.concatenate([fitness, opp_fit])
        si = np.argsort(combined_fit)[:pop_size]
        population = combined_pop[si].copy()
        fitness = combined_fit[si].copy()
    
    # --- Phase 2: SHADE ---
    H = 6
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    ki = 0
    archive = []
    max_archive = pop_size
    stagnation = 0
    prev_best = best
    
    de_time_limit = max_time * 0.70
    nm_during_de = max(1, pop_size // 15)
    gen_count = 0
    
    while elapsed() < de_time_limit:
        SF, SCR, Sdelta = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        sorted_idx = np.argsort(fitness)
        gen_count += 1
        
        for i in range(pop_size):
            if elapsed() >= de_time_limit:
                break
            
            ri = np.random.randint(H)
            Fi = -1
            for _ in range(20):
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p = max(2, int(0.11 * pop_size))
            pbest = sorted_idx[np.random.randint(p)]
            
            candidates = [j for j in range(pop_size) if j != i]
            r1 = candidates[np.random.randint(len(candidates))]
            
            pool_size = pop_size + len(archive)
            r2c = i
            for _ in range(50):
                r2c = np.random.randint(pool_size)
                if r2c != i and r2c != r1:
                    break
            xr2 = population[r2c] if r2c < pop_size else archive[r2c - pop_size]
            
            mutant = population[i] + Fi * (population[pbest] - population[i]) + Fi * (population[r1] - xr2)
            mask = np.random.random(dim) < CRi
            mask[np.random.randint(dim)] = True
            trial = clip(np.where(mask, mutant, population[i]))
            
            ft = evaluate(trial)
            if ft < fitness[i]:
                delta = fitness[i] - ft
                SF.append(Fi); SCR.append(CRi); Sdelta.append(delta)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial; new_fit[i] = ft
        
        population = new_pop; fitness = new_fit
        
        if SF:
            w = np.array(Sdelta); w = w / (w.sum() + 1e-30)
            sf_arr = np.array(SF); scr_arr = np.array(SCR)
            M_F[ki] = np.sum(w * sf_arr**2) / (np.sum(w * sf_arr) + 1e-30)
            M_CR[ki] = np.sum(w * scr_arr); ki = (ki + 1) % H
        
        if abs(best - prev_best) < 1e-14: stagnation += 1
        else: stagnation = 0
        prev_best = best
        
        if stagnation > 18:
            si2 = np.argsort(fitness)
            for idx in range(pop_size // 2, pop_size):
                if elapsed() >= de_time_limit: break
                j = si2[idx]
                population[j] = lower + np.random.random(dim) * ranges
                fitness[j] = evaluate(population[j])
            stagnation = 0
    
    # --- Phase 3: Nelder-Mead polishing ---
    def nelder_mead(x0, scale_factor, time_lim):
        n = dim; scale = scale_factor * ranges
        simplex = np.zeros((n+1, n)); simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += scale[i] if x0[i]+scale[i]<=upper[i] else -scale[i]
        simplex = clip(simplex)
        fs = np.array([evaluate(simplex[j]) for j in range(n+1)])
        for _ in range(20000):
            if elapsed() >= time_lim: break
            o = np.argsort(fs); simplex = simplex[o]; fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = clip(c + (c - simplex[-1])); fr = evaluate(xr)
            if fs[0] <= fr < fs[-2]: simplex[-1], fs[-1] = xr, fr
            elif fr < fs[0]:
                xe = clip(c + 2*(xr - c)); fe = evaluate(xe)
                simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            else:
                xc = clip(c + 0.5*(simplex[-1] - c)); fc = evaluate(xc)
                if fc < fs[-1]: simplex[-1], fs[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = clip(simplex[0] + 0.5*(simplex[i] - simplex[0]))
                        if elapsed() >= time_lim: return
                        fs[i] = evaluate(simplex[i])
            if np.std(fs) < 1e-16: break
    
    si = np.argsort(fitness)
    for a in range(min(4, pop_size)):
        if elapsed() >= max_time * 0.92: break
        nelder_mead(population[si[a]], 0.05/(1+a*0.5), max_time*(0.78+a*0.035))
    
    for sc, tf in [(0.01, 0.95), (0.002, 0.98), (0.0004, 0.995)]:
        if elapsed() < max_time * tf and best_params is not None:
            nelder_mead(best_params, sc, max_time * tf)
    
    return best
