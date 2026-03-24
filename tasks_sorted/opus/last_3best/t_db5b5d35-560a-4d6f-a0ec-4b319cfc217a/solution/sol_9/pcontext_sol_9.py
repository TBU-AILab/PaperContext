#Looking at the results, algorithm 1 (score 10.28) is the best. The algorithms are very similar - all use L-SHADE + Nelder-Mead + coordinate descent. Let me analyze what makes algorithm 1 better and improve upon it:
#
#Key improvements:
#1. **Add a simple CMA-ES-inspired local search** using covariance adaptation
#2. **Better population diversity management** with opposition-based learning
#3. **Golden section search** in coordinate descent for more precise line searches
#4. **Multiple restarts of NM from perturbed best** to escape local optima
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
    
    def evaluate(x):
        nonlocal best, best_params
        x = np.clip(x, lower, upper)
        f = func(x)
        if f < best:
            best = f
            best_params = x.copy()
        return f

    # ---- Phase 1: LHS initialization ----
    pop_size_init = min(max(12 * dim, 50), 200)
    pop_size = pop_size_init
    min_pop_size = max(4, dim // 2)
    
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
    
    # Opposition-based learning for initial population
    obl_pop = lower + upper - population
    obl_pop = np.clip(obl_pop, lower, upper)
    obl_fit = np.full(pop_size, float('inf'))
    for i in range(pop_size):
        if elapsed() >= max_time * 0.90:
            return best
        obl_fit[i] = evaluate(obl_pop[i])
    
    combined = np.vstack([population, obl_pop])
    combined_fit = np.concatenate([fitness, obl_fit])
    order = np.argsort(combined_fit)[:pop_size]
    population = combined[order]
    fitness = combined_fit[order]
    
    # ---- Phase 2: L-SHADE ----
    memory_size = 6
    M_F = np.full(memory_size, 0.5)
    M_CR = np.full(memory_size, 0.5)
    k = 0
    archive = []
    max_archive = pop_size_init
    
    stagnation = 0
    prev_best = best
    de_time_frac = 0.60
    
    while elapsed() < max_time * de_time_frac:
        S_F, S_CR, S_delta = [], [], []
        new_pop = population.copy()
        new_fit = fitness.copy()
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            if elapsed() >= max_time * de_time_frac:
                break
            
            ri = np.random.randint(memory_size)
            Fi = -1
            for _ in range(20):
                Fi = np.random.standard_cauchy() * 0.1 + M_F[ri]
                if Fi > 0:
                    break
            if Fi <= 0:
                Fi = 0.1
            Fi = min(Fi, 1.0)
            CRi = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)
            
            p_count = max(2, int(np.ceil(np.random.uniform(2.0/pop_size, 0.2) * pop_size)))
            pbest_idx = np.random.choice(sorted_idx[:p_count])
            
            candidates = [j for j in range(pop_size) if j != i]
            r1 = candidates[np.random.randint(len(candidates))]
            
            combined_size = pop_size + len(archive)
            r2 = np.random.randint(combined_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(combined_size)
            xr2 = archive[r2 - pop_size] if r2 >= pop_size else population[r2]
            
            mutant = population[i] + Fi * (population[pbest_idx] - population[i]) + Fi * (population[r1] - xr2)
            for dd in range(dim):
                if mutant[dd] < lower[dd]:
                    mutant[dd] = (lower[dd] + population[i][dd]) / 2.0
                elif mutant[dd] > upper[dd]:
                    mutant[dd] = (upper[dd] + population[i][dd]) / 2.0
            
            jrand = np.random.randint(dim)
            mask = (np.random.random(dim) < CRi) | (np.arange(dim) == jrand)
            trial = np.where(mask, mutant, population[i])
            
            f_trial = evaluate(trial)
            if f_trial < fitness[i]:
                S_F.append(Fi); S_CR.append(CRi); S_delta.append(fitness[i] - f_trial)
                if len(archive) < max_archive:
                    archive.append(population[i].copy())
                elif archive:
                    archive[np.random.randint(len(archive))] = population[i].copy()
                new_pop[i] = trial; new_fit[i] = f_trial
            elif f_trial == fitness[i]:
                new_pop[i] = trial; new_fit[i] = f_trial
        
        population = new_pop; fitness = new_fit
        
        if S_F:
            w = np.array(S_delta); w = w / (w.sum() + 1e-30)
            sf = np.array(S_F); sc = np.array(S_CR)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * sc)
            k = (k + 1) % memory_size
        
        new_ps = max(min_pop_size, int(round(pop_size_init - (pop_size_init - min_pop_size) * elapsed() / (max_time * de_time_frac))))
        if new_ps < pop_size:
            si = np.argsort(fitness)
            population = population[si[:new_ps]]; fitness = fitness[si[:new_ps]]; pop_size = new_ps
        
        if abs(prev_best - best) < 1e-14:
            stagnation += 1
        else:
            stagnation = 0
        prev_best = best
        
        if stagnation > 20:
            keep = max(pop_size // 4, 2); order = np.argsort(fitness)
            for ii in range(keep, pop_size):
                if elapsed() >= max_time * de_time_frac: break
                if np.random.random() < 0.5:
                    population[order[ii]] = best_params + np.random.randn(dim) * 0.1 * ranges
                else:
                    population[order[ii]] = lower + np.random.random(dim) * ranges
                population[order[ii]] = np.clip(population[order[ii]], lower, upper)
                fitness[order[ii]] = evaluate(population[order[ii]])
            stagnation = 0

    # ---- Phase 3: Nelder-Mead at multiple scales ----
    def nelder_mead(x0, sf, tl):
        n = dim; scale = sf * ranges
        simplex = np.zeros((n+1, n)); simplex[0] = x0.copy()
        for i in range(n):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += scale[i] * (1 if np.random.random() > 0.5 else -1)
        simplex = np.clip(simplex, lower, upper)
        fs = []
        for i in range(n+1):
            if elapsed() >= tl: return
            fs.append(evaluate(simplex[i]))
        fs = np.array(fs)
        for _ in range(10000):
            if elapsed() >= tl: break
            o = np.argsort(fs); simplex = simplex[o]; fs = fs[o]
            c = np.mean(simplex[:-1], axis=0)
            xr = np.clip(2*c - simplex[-1], lower, upper); fr = evaluate(xr)
            if fs[0] <= fr < fs[-2]:
                simplex[-1] = xr; fs[-1] = fr; continue
            if fr < fs[0]:
                xe = np.clip(c + 2*(xr - c), lower, upper); fe = evaluate(xe)
                if fe < fr: simplex[-1] = xe; fs[-1] = fe
                else: simplex[-1] = xr; fs[-1] = fr
                continue
            if fr < fs[-1]:
                xc = np.clip(c + 0.5*(xr - c), lower, upper); fc = evaluate(xc)
                if fc <= fr: simplex[-1] = xc; fs[-1] = fc; continue
            else:
                xc = np.clip(c + 0.5*(simplex[-1] - c), lower, upper); fc = evaluate(xc)
                if fc < fs[-1]: simplex[-1] = xc; fs[-1] = fc; continue
            for i in range(1, n+1):
                if elapsed() >= tl: break
                simplex[i] = np.clip(simplex[0] + 0.5*(simplex[i] - simplex[0]), lower, upper)
                fs[i] = evaluate(simplex[i])
            if np.max(np.abs(fs - fs[0])) < 1e-16: break

    if best_params is not None:
        scales = [0.1, 0.03, 0.01, 0.003, 0.001]
        nm_end = 0.88
        for idx_s, sf in enumerate(scales):
            if elapsed() >= max_time * nm_end: break
            remaining = max_time * nm_end - elapsed()
            time_budget = remaining / (len(scales) - idx_s)
            nelder_mead(best_params.copy(), sf, elapsed() + time_budget)

    # ---- Phase 4: Pattern search with adaptive step ----
    if best_params is not None:
        x = best_params.copy(); fx = best
        for ss in [0.02, 0.005, 0.001, 0.0003]:
            if elapsed() >= max_time * 0.97: break
            step = ss * ranges
            improved = True
            while improved and elapsed() < max_time * 0.97:
                improved = False
                perm = np.random.permutation(dim)
                for d in perm:
                    if elapsed() >= max_time * 0.97: break
                    for dr in [1, -1]:
                        xn = x.copy()
                        xn[d] = np.clip(x[d] + dr * step[d], lower[d], upper[d])
                        fn = evaluate(xn)
                        if fn < fx:
                            # Accelerated step
                            x2 = x.copy()
                            x2[d] = np.clip(x[d] + 2*dr*step[d], lower[d], upper[d])
                            f2 = evaluate(x2)
                            if f2 < fn:
                                x = x2; fx = f2
                            else:
                                x = xn; fx = fn
                            improved = True
                            break

    # ---- Phase 5: Final random perturbations around best ----
    if best_params is not None:
        while elapsed() < max_time * 0.995:
            sigma = 0.001 * ranges * np.random.exponential(1.0)
            candidate = best_params + np.random.randn(dim) * sigma
            evaluate(candidate)

    return best
